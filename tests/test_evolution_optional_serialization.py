from dataclasses import dataclass, field

import pytest

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.evidence import Support
from skillforge.domain.semantic_antipattern import SemanticAntiPatternRequest
from skillforge.domain.semantic_workflow import SemanticWorkflowRequest
from skillforge.domain.serialization import to_json_value
from skillforge.productization.evolution_artifacts import _restore


def _workflow_request(feedback: str = "") -> SemanticWorkflowRequest:
    return SemanticWorkflowRequest(
        repository_id="repo",
        candidate_id="candidate",
        kind=ChangePatternKind.MODULE_CO_CHANGE,
        members=(),
        current_modules=(),
        support=Support(0, 0),
        support_ratio=0.0,
        pair_associations=(),
        structural_links=(),
        structural_coherence=None,
        noise_hints=(),
        source_patterns=(),
        supporting_commits=(),
        family_supporting_commits=(),
        evidence=(),
        validation_feedback=feedback,
    )


def _antipattern_request(feedback: str = "") -> SemanticAntiPatternRequest:
    return SemanticAntiPatternRequest(
        repository_id="repo",
        candidate_id="candidate",
        affected_paths=(),
        signal_count=0,
        involved_commit_count=0,
        examined_commits=0,
        signals=(),
        commits=(),
        evidence=(),
        validation_feedback=feedback,
    )


@pytest.mark.parametrize(
    "semantic_request",
    [
        _workflow_request(),
        _workflow_request("review note"),
        _antipattern_request(),
        _antipattern_request("review note"),
    ],
)
def test_validation_feedback_round_trips_when_empty_or_present(semantic_request):
    payload = to_json_value(semantic_request)
    if semantic_request.validation_feedback:
        assert payload["validation_feedback"] == semantic_request.validation_feedback
    else:
        assert "validation_feedback" not in payload
    assert _restore(type(semantic_request), payload) == semantic_request


@dataclass(frozen=True)
class StrictRecord:
    required: str
    ordinary_default: str = "ordinary"


def test_missing_field_without_omit_empty_metadata_is_rejected():
    with pytest.raises(ValueError, match="invalid evolution object fields"):
        _restore(StrictRecord, {"required": "present"})


def test_unknown_object_field_is_rejected():
    with pytest.raises(ValueError, match="invalid evolution object fields"):
        _restore(StrictRecord, {"required": "present", "ordinary_default": "x",
                                "extra": "unknown"})


@dataclass(frozen=True)
class NonEmptyOmittableDefault:
    required: str
    label: str = field(default="kept", metadata={"omit_empty": True})


def test_omitting_nonempty_omit_empty_default_fails_canonical_round_trip():
    with pytest.raises(ValueError, match="evolution object identity mismatch"):
        _restore(NonEmptyOmittableDefault, {"required": "present"})
