"""The existing single retry receives bounded diagnostics, never new evidence."""
from dataclasses import dataclass

import pytest

from skillforge.errors import SemanticDecisionValidationError
from skillforge.productization.semantic_assembly import _evaluate_candidate, _ObservedModel
from skillforge.repository.semantic_batch import PrefetchedSemanticModel


@dataclass(frozen=True)
class Request:
    repository_id: str = "repo"
    candidate_id: str = "candidate"
    evidence: tuple = ("frozen-source",)
    validation_feedback: str = ""


@pytest.mark.parametrize("preliminary", [False, True])
def test_existing_retry_receives_diagnostic_and_keeps_evidence(preliminary):
    seen = []

    class Model:
        def complete(self, request):
            seen.append(request)
            return "corrected" if request.validation_feedback else "invalid"

    observed = _ObservedModel(Model())
    request = Request()
    error = SemanticDecisionValidationError("unknown path: invented.py")

    def evaluate():
        response = observed.complete(request)
        if response == "invalid":
            raise error
        return response

    result, retries = _evaluate_candidate(evaluate, observed, retry=True,
        preliminary=(None, error) if preliminary else None)
    assert (result, retries) == ("corrected", 1)
    assert len(seen) == (1 if preliminary else 2)
    assert seen[-1].validation_feedback == str(error)
    assert seen[-1].evidence == request.evidence
    assert request.validation_feedback == ""


def test_retry_remains_once_and_feedback_is_bounded():
    seen = []

    class Model:
        def complete(self, request):
            seen.append(request)
            return "invalid"

    observed = _ObservedModel(Model())

    def evaluate():
        observed.complete(Request())
        raise SemanticDecisionValidationError("错误" * 1000)

    with pytest.raises(SemanticDecisionValidationError):
        _evaluate_candidate(evaluate, observed, retry=True)
    assert len(seen) == 2
    assert 0 < len(seen[-1].validation_feedback.encode("utf-8")) <= 512


def test_prefetched_failure_feedback_reaches_single_transport():
    from skillforge.repository.semantic_batch import request_id
    request = Request()
    seen = []

    class Model:
        def complete(self, request):
            seen.append(request)
            return "corrected"

    observed = _ObservedModel(PrefetchedSemanticModel(Model(),
        {request_id(request): "invalid"}))

    def evaluate():
        result = observed.complete(request)
        if result == "invalid":
            raise SemanticDecisionValidationError("unknown path: invented.py")
        return result

    assert _evaluate_candidate(evaluate, observed, retry=True) == ("corrected", 1)
    assert len(seen) == 1
    assert seen[0].validation_feedback == "unknown path: invented.py"


def test_diagnostics_keep_candidate_identity_without_erasing_evidence():
    from dataclasses import replace

    from test_semantic_workflows import _fixture

    from skillforge.repository.semantic_batch import request_id
    from skillforge.repository.semantic_workflows import build_semantic_workflow_request

    history, facts, candidates = _fixture()
    request = build_semantic_workflow_request(history, facts, candidates, candidates.candidates[0])
    repair = replace(request, validation_feedback="unknown path: invented.py")
    assert request_id(request) == request_id(repair)
    assert request_id(request) != request_id(replace(repair, evidence=()))
    with pytest.raises(ValueError, match="512 bytes"):
        replace(request, validation_feedback="x" * 513)
    # A custom request's similarly named field is not silently normalized.
    assert request_id(Request()) != request_id(Request(validation_feedback="custom fact"))


def test_empty_diagnostics_preserve_original_wire_and_fingerprint():
    import hashlib
    import json
    from dataclasses import fields, replace

    from test_semantic_workflows import _fixture

    from skillforge.domain.evolution import fingerprint
    from skillforge.domain.serialization import to_json_value
    from skillforge.repository.semantic_workflows import build_semantic_workflow_request

    history, facts, candidates = _fixture()
    request = build_semantic_workflow_request(history, facts, candidates, candidates.candidates[0])
    old = {f.name: to_json_value(getattr(request, f.name)) for f in fields(request)
           if f.name != "validation_feedback"}
    assert to_json_value(request) == old
    canonical = json.dumps(old, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert fingerprint(request) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    feedback = to_json_value(replace(request, validation_feedback="unknown path: invented.py"))
    assert feedback.pop("validation_feedback") == "unknown path: invented.py"
    assert feedback == old
