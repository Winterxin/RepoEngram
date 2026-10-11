"""History observations are conventions, never ordered workflow obligations."""

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.change import StructuralRoleEvent
from skillforge.domain.evidence import Evidence
from skillforge.domain.project_dna import ProjectRule
from skillforge.domain.semantic_workflow import (
    SemanticEvidenceRef,
    SemanticSupportingCommit,
    SemanticWorkflowRequest,
)
from skillforge.domain.serialization import to_json_value

TENDENCY_METHOD = "llm_semantic_history_tendency_v1"


class TendencyDecisionKind(StrEnum):
    SUPPORTED_TENDENCY = "SUPPORTED_TENDENCY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCOHERENT = "INCOHERENT"


@dataclass(frozen=True, slots=True)
class SemanticTendencyRequest:
    context: SemanticWorkflowRequest
    normalized_role: str
    parent_package: str
    structural_scope: str
    structural_events: tuple[StructuralRoleEvent, ...]
    examined_changes: tuple[SemanticSupportingCommit, ...]
    evidence: tuple[SemanticEvidenceRef, ...]

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticTendencyResponse:
    repository_id: str
    candidate_id: str
    decision: TendencyDecisionKind
    reason: str
    evidence_ids: tuple[str, ...]
    audit_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TendencyTransportDecision:
    """Semantic fields only; any model identity echo is untrusted audit data."""

    decision: TendencyDecisionKind
    reason: str
    evidence_ids: tuple[str, ...]
    echoed_repository_id: str | None = None
    echoed_candidate_id: str | None = None
    citation_choices: tuple[tuple[str, str | None], ...] | None = None


@dataclass(frozen=True, slots=True)
class ScopedTendencyDecision:
    """The transport stamps the exact single request it actually evaluated."""

    request_fingerprint: str
    decision: TendencyTransportDecision


@dataclass(frozen=True, slots=True)
class SemanticTendencyAcceptance:
    request: SemanticTendencyRequest
    response: SemanticTendencyResponse
    rule: ProjectRule


@dataclass(frozen=True, slots=True)
class SemanticTendencyResult:
    repository_id: str
    accepted: tuple[SemanticTendencyAcceptance, ...]
    rejections: tuple[SemanticTendencyResponse, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.rule.rule_id for item in self.accepted)
        candidates = tuple(item.response.candidate_id for item in self.accepted)
        candidates += tuple(item.candidate_id for item in self.rejections)
        if len(set(ids)) != len(ids) or len(set(candidates)) != len(candidates):
            raise ValueError("duplicate tendency decisions")
        if any(item.rule.repository_id != self.repository_id for item in self.accepted):
            raise ValueError("tendency repository mismatch")
        if any(item.repository_id != self.repository_id or
               item.decision == TendencyDecisionKind.SUPPORTED_TENDENCY
               for item in self.rejections):
            raise ValueError("invalid tendency rejection")
        refs = {key for item in self.accepted for key in item.rule.evidence_ids}
        refs.update(key for item in self.rejections for key in item.evidence_ids)
        if tuple(item.evidence_id for item in self.evidence) != tuple(sorted(refs)) or any(
            item.repository_id != self.repository_id for item in self.evidence
        ):
            raise ValueError("tendency Evidence closure mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
