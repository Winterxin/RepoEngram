"""Interrupted validation resumes with bounded, isolated diagnostics, not answers."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import WorkflowModel
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import complete_identity
from test_semantic_workflows import _fixture

from skillforge.errors import SemanticDecisionValidationError
from skillforge.productization.semantic_assembly import _evaluate_candidate, _ObservedModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_batch import request_id
from skillforge.repository.semantic_workflows import (
    _validate_decision,
    build_semantic_workflow_request,
)


def request_fixture():
    history, static, candidates = _fixture()
    return build_semantic_workflow_request(history, static, candidates, candidates.candidates[0])


class Interrupted(BaseException):
    pass


class ResumeModel(WorkflowModel):
    def __init__(self, interrupted=False):
        super().__init__()
        self.semantic_cache_identity = complete_identity()
        self.interrupted = interrupted

    def complete(self, request):
        if self.interrupted and request.validation_feedback:
            raise Interrupted()
        response = super().complete(request)
        if self.interrupted:
            return replace(response, reason="References invented.py")
        return response

    def complete_batch(self, requests):
        return {request_id(request): self.complete(request) for request in requests}


def test_interrupted_validation_resumes_and_success_reuses_without_feedback(
    historical_repository, tmp_path,
):
    directory = tmp_path.parent / (tmp_path.name + "-diagnostics-cache")
    request = request_fixture()
    first = ResumeModel(interrupted=True)
    cache = RepoSnapshotSemanticCache(historical_repository, directory)
    observed = _ObservedModel(cache.wrap(first))

    def evaluate():
        response = observed.complete(request)
        try:
            _validate_decision(request, response)
        except ValueError as error:
            raise SemanticDecisionValidationError(str(error)) from error
        return response

    with pytest.raises(Interrupted):
        _evaluate_candidate(evaluate, observed, retry=True)
    assert len(first.requests) == 1
    assert not list(directory.glob("*.json"))
    records = list((directory / "validation-diagnostics").glob("*.json"))
    assert len(records) == 1
    diagnostic = json.loads(records[0].read_text("utf-8"))
    assert "unknown path: invented.py" in diagnostic["validation_feedback"]
    assert "response" not in diagnostic

    second = ResumeModel()
    resumed = RepoSnapshotSemanticCache(historical_repository, directory)
    wrapped = resumed.wrap(second)
    result = wrapped.complete(request)
    _validate_decision(request, result)
    assert len(second.requests) == 1
    assert second.requests[0].validation_feedback == diagnostic["validation_feedback"]
    assert second.requests[0].evidence == request.evidence
    wrapped.checkpoint_candidate(request.candidate_id)
    assert not list((directory / "validation-diagnostics").glob("*.json"))
    third = ResumeModel()
    warm = RepoSnapshotSemanticCache(historical_repository, directory).wrap(third)
    assert warm.complete(request) == result
    assert not third.requests


@pytest.mark.parametrize("change", ["source", "request", "model", "corrupt", "oversized"])
def test_diagnostic_never_crosses_source_request_model_or_integrity_boundary(
    historical_repository, tmp_path, change,
):
    directory = tmp_path.parent / (tmp_path.name + "-isolated-cache")
    request = request_fixture()
    first = RepoSnapshotSemanticCache(historical_repository, directory).wrap(ResumeModel())
    first.remember_validation_feedback(request, "unknown path: invented.py")
    diagnostic = next((directory / "validation-diagnostics").glob("*.json"))
    model = ResumeModel()
    if change == "source":
        (historical_repository / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    elif change == "request":
        request = replace(request, evidence=())
    elif change == "model":
        model.semantic_cache_identity = deepcopy(model.semantic_cache_identity)
        model.semantic_cache_identity["model_selector"] += "-different"
    elif change == "corrupt":
        record = json.loads(diagnostic.read_text("utf-8"))
        record["feedback_hash"] = "wrong"
        diagnostic.write_text(json.dumps(record), encoding="utf-8")
    else:
        diagnostic.write_text(" " * 16385, encoding="utf-8")
    wrapped = RepoSnapshotSemanticCache(historical_repository, directory).wrap(model)
    assert wrapped.resume_request(request) == request


def test_batch_resume_is_bounded_and_preserves_candidate_identity(historical_repository, tmp_path):
    request = request_fixture()
    directory = tmp_path.parent / (tmp_path.name + "-batch-cache")
    first = RepoSnapshotSemanticCache(historical_repository, directory).wrap(ResumeModel())
    first.remember_validation_feedback(request, "错误" * 1000)
    model = ResumeModel()
    wrapped = RepoSnapshotSemanticCache(historical_repository, directory).wrap(model)
    result = wrapped.complete_batch((request,))
    assert set(result) == {request_id(request)}
    assert 0 < len(model.requests[0].validation_feedback.encode("utf-8")) <= 512
    assert request_id(model.requests[0]) == request_id(request)
    assert model.requests[0].evidence == request.evidence


def test_cohort_resumes_diagnostic_without_an_extra_transport(eleven_inputs, tmp_path):
    from test_semantic_batch_transport import recorded_models
    from test_semantic_cohort import assemble, enable

    expected, runner, models = recorded_models(eleven_inputs)
    enable(models)
    root, facts, history, candidates, _, selected, _, _ = eleven_inputs
    directory = tmp_path.parent / (tmp_path.name + "-cohort-cache")
    request = build_semantic_workflow_request(history, facts, candidates, selected[0])
    # The exact-wire recorder includes this diagnostic in its expected wire;
    # all source facts, responses and original equality assertions stay intact.
    from skillforge.repository.semantic_workflow_prompt import build_semantic_workflow_prompt

    records = {}
    for prompt, (original, response) in runner.records.items():
        if request_id(original) == request_id(request):
            original = replace(original, validation_feedback="unknown path: invented.py")
            prompt = build_semantic_workflow_prompt(original)
        records[prompt] = original, response
    runner.records = records
    wrapped = RepoSnapshotSemanticCache(root, directory).wrap(models[1])
    wrapped.remember_validation_feedback(request, "unknown path: invented.py")
    result, _ = assemble(eleven_inputs, models, directory)
    assert len(runner.calls) == 1
    assert "unknown path: invented.py" in runner.calls[0]
    assert not result.failures
    assert result.workflows == expected[0].workflows
    assert len(list(directory.glob("*.json"))) == 11
    assert not list((directory / "validation-diagnostics").glob("*.json"))
    warm, cache = assemble(eleven_inputs, models, directory)
    assert warm.workflows == result.workflows
    assert len(runner.calls) == 1 and cache.hits == 11
