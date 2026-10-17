"""Incremental semantic cache publication after unchanged evaluator gates."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_prepare import (
    FakeModel,
    historical_repository as historical_repository_fixture,
    repository as repository_fixture,
)
from test_repo_snapshot_semantic_cache import RecordingModel, complete_identity

from skillforge.cost_control import BudgetExceeded
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner
from skillforge.productization.semantic_assembly import _checkpoint_validated_candidate


@pytest.fixture
def historical_repository(tmp_path):
    root = repository_fixture.__wrapped__(tmp_path)
    return historical_repository_fixture.__wrapped__(root)


class AbortAfterOne(RecordingModel):
    def __init__(self, decision):
        super().__init__()
        self.decision = decision
        self.seen = []
        self.abort = True

    def complete(self, request):
        self.seen.append(request.candidate_id)
        if self.abort and len(self.seen) == 2:
            raise BudgetExceeded("fixture later-candidate budget abort")
        return FakeModel.complete(self, request)


def _mined(root):
    facts = PythonStaticAnalyzer().analyze(root)
    return facts, StructuralRuleMiner().mine(facts)


def _add_reverse_import_pattern(root):
    worker = root / "src/app/worker"
    worker.mkdir(parents=True, exist_ok=True)
    (worker / "__init__.py").write_text("", encoding="utf-8")
    for index in range(3):
        (worker / f"unit{index}.py").write_text(
            f"import app.service.unit{index % 2}\n", encoding="utf-8",
        )


def test_accepted_tendency_checkpoints_response_candidate_id():
    class CheckpointSpy:
        def __init__(self):
            self.ids = []

        def checkpoint_candidate(self, candidate_id):
            self.ids.append(candidate_id)

    candidate_id = "tendency-candidate"
    result = SimpleNamespace(
        accepted=(SimpleNamespace(response=SimpleNamespace(candidate_id=candidate_id)),),
        rejections=(),
    )
    spy = CheckpointSpy()
    _checkpoint_validated_candidate(
        SimpleNamespace(model=spy), candidate_id, result, ("accepted", "rejections"),
    )
    assert spy.ids == [candidate_id]


@pytest.mark.parametrize("decision", (
    SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
    SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE,
))
def test_validated_checkpoint_survives_later_budget_abort(
        historical_repository, tmp_path, decision):
    directory = tmp_path.parent / (tmp_path.name + "-cache")
    _add_reverse_import_pattern(historical_repository)
    model = AbortAfterOne(decision)
    cache = RepoSnapshotSemanticCache(historical_repository, directory)
    facts, mined = _mined(historical_repository)
    with pytest.raises(BudgetExceeded, match="later-candidate"):
        from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge

        evaluate_semantic_knowledge(
            mined.repository_id, facts, mined, cache.wrap(model),
        )

    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    entry = json.loads(paths[0].read_text(encoding="utf-8"))
    assert entry["identity"]["candidate_input_hash"]
    assert entry["response"]["candidate_id"] == model.seen[0]

    # Exact replay still runs through the evaluator and reproduces its canonical result.
    candidate = next(item for item in mined.candidates if item.candidate_id == model.seen[0])
    scoped = replace(mined, candidates=(candidate,), evidence=tuple(
        item for item in mined.evidence if item.evidence_id in candidate.rule.evidence_ids
    ))
    resumed_model = AbortAfterOne(decision)
    resumed_model.abort = False
    resumed_model.semantic_cache_identity = deepcopy(model.semantic_cache_identity)
    resumed_cache = RepoSnapshotSemanticCache(historical_repository, directory)
    resumed = SemanticStructuralRuleEvaluator().evaluate(scoped, resumed_cache.wrap(resumed_model))
    direct_model = RecordingModel()
    direct_model.decision = decision
    expected = SemanticStructuralRuleEvaluator().evaluate(scoped, direct_model)
    assert resumed == expected
    assert resumed_cache.hits == 1 and resumed_model.seen == []


def test_invalid_evaluator_response_is_not_checkpointed(historical_repository, tmp_path):
    class Invalid(RecordingModel):
        def complete(self, request):
            self.calls += 1
            return replace(FakeModel.complete(self, request),
                           decision_evidence_ids=("unknown",))

    model = Invalid()
    model.invalid = False
    cache = RepoSnapshotSemanticCache(
        historical_repository, tmp_path.parent / (tmp_path.name + "-cache"),
    )
    facts, mined = _mined(historical_repository)
    from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge

    result = evaluate_semantic_knowledge(mined.repository_id, facts, mined, cache.wrap(model))
    assert result.structural.failures
    assert not list(cache.directory.glob("*.json"))


def test_checkpoint_keeps_unvalidated_entries_and_respects_model_config(
        historical_repository, tmp_path):
    _add_reverse_import_pattern(historical_repository)
    cache = RepoSnapshotSemanticCache(
        historical_repository, tmp_path.parent / (tmp_path.name + "-cache"),
    )
    model = RecordingModel()
    cached = cache.wrap(model)
    _, mined = _mined(historical_repository)
    requests = [build_semantic_structural_request(mined, item)
                for item in mined.candidates[:2]]
    assert len(requests) == 2
    for request in requests:
        cached.complete(request)
    cached.checkpoint_candidate(requests[0].candidate_id)
    assert len(list(cache.directory.glob("*.json"))) == 1
    assert len(cache.pending) == 1
    assert next(iter(cache.pending.values()))[0] == requests[1].candidate_id

    # Another adapter configuration for the same candidate cannot publish this entry.
    alternate = RecordingModel()
    alternate.semantic_cache_identity = complete_identity(model="different-selector")
    other_cache = RepoSnapshotSemanticCache(historical_repository, cache.directory)
    other = other_cache.wrap(alternate)
    request = requests[0]
    response = other.complete(request)
    assert response.candidate_id == request.candidate_id
    assert other_cache.hits == 0 and alternate.calls == 1


@pytest.mark.parametrize("drift", ("snapshot", "config"))
def test_checkpoint_does_not_hit_after_snapshot_or_config_drift(
        historical_repository, tmp_path, drift):
    directory = tmp_path.parent / (tmp_path.name + "-cache")
    original = RecordingModel()
    cache = RepoSnapshotSemanticCache(historical_repository, directory)
    _, mined = _mined(historical_repository)
    request = build_semantic_structural_request(mined, mined.candidates[0])
    result = SemanticStructuralRuleEvaluator().evaluate(
        replace(mined, candidates=(mined.candidates[0],), evidence=tuple(
            item for item in mined.evidence
            if item.evidence_id in mined.candidates[0].rule.evidence_ids
        )), cache.wrap(original),
    )
    cache.checkpoint_candidate(result.accepted[0].candidate_id,
                               cache.wrap(original).config)
    assert list(directory.glob("*.json"))
    if drift == "snapshot":
        (historical_repository / "new-source.txt").write_text("changed", encoding="utf-8")
    changed = RecordingModel()
    if drift == "config":
        changed.semantic_cache_identity = complete_identity(model="different-selector")
    fresh = RepoSnapshotSemanticCache(historical_repository, directory)
    fresh.wrap(changed).complete(request)
    assert fresh.hits == 0 and changed.calls == 1
