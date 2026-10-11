"""Evidence-bounded decisions about one structural observation."""

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.evidence import Evidence, Support
from skillforge.domain.serialization import to_json_value
from skillforge.domain.static_analysis import PythonTestLinkage
from skillforge.domain.structural_rules import StructuralRuleCandidate

SEMANTIC_METHOD = "llm_semantic_structural_v1"


class SemanticStructuralDecisionKind(StrEnum):
    ARCHITECTURE_RULE = "ARCHITECTURE_RULE"
    INCOHERENT = "INCOHERENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class StructuralEvidenceRef:
    evidence_id: str
    path: str
    line: int | None
    description: str
    observed_import_relation: str


@dataclass(frozen=True, slots=True)
class StructuralEvidenceGroup:
    """Candidate-local structured fact with deterministic source provenance."""

    fact_id: str
    subject: str
    evidence_ids: tuple[str, ...]
    scope_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticStructuralRequest:
    repository_id: str
    candidate_id: str
    canonical_observation: str
    source_package: str
    target_package: str
    support: Support
    examined_modules: tuple[str, ...]
    source_modules: tuple[str, ...]
    target_modules: tuple[str, ...]
    evidence: tuple[StructuralEvidenceRef, ...]
    observation_kind: str = "imports"
    operation_symbols: tuple[str, ...] = ()
    delegation_topology: tuple[str, ...] = ()
    operation_family: tuple[str, ...] = ()
    test_linkages: tuple[PythonTestLinkage, ...] = ()
    evidence_groups: tuple[StructuralEvidenceGroup, ...] = ()
    examined_operation_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.repository_id or not self.candidate_id:
            raise ValueError("semantic request identity is required")
        if self.observation_kind not in {"imports", "static_delegation", "static_testing"}:
            raise ValueError("unknown structural observation kind")
        population = (self.operation_symbols if self.observation_kind != "imports"
                      else self.source_modules)
        if self.support.supporting != len(population):
            raise ValueError("semantic request support mismatch")
        examined = (self.operation_symbols if self.observation_kind != "imports"
                    else self.examined_modules)
        if self.examined_operation_symbols:
            if (self.observation_kind != "static_delegation"
                    or len(self.delegation_topology) != 2 or self.operation_family
                    or self.examined_operation_symbols
                    != tuple(sorted(set(self.examined_operation_symbols)))
                    or not set(self.operation_symbols) <= set(self.examined_operation_symbols)):
                raise ValueError("invalid repeated owner examined operations")
            examined = self.examined_operation_symbols
        if self.support.examined != len(examined):
            raise ValueError("semantic request examined module mismatch")
        ids = tuple(item.evidence_id for item in self.evidence)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError("semantic request Evidence IDs must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        result = to_json_value(self)
        if not self.examined_operation_symbols:
            # Keep legacy candidate prompts byte-for-byte compatible.
            del result["examined_operation_symbols"]
        return result


@dataclass(frozen=True, slots=True)
class SemanticStructuralException:
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticStructuralResponse:
    repository_id: str
    candidate_id: str
    decision: SemanticStructuralDecisionKind
    reason: str
    decision_evidence_ids: tuple[str, ...]
    source_package: str
    target_package: str
    scope_modules: tuple[str, ...]
    scope_evidence_ids: tuple[str, ...]
    exceptions: tuple[SemanticStructuralException, ...]
    supporting_fact_ids: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticStructuralAcceptance:
    """Validated interpretation; the ProjectRule remains the mined candidate's rule."""

    candidate: StructuralRuleCandidate
    candidate_id: str
    decision: SemanticStructuralDecisionKind
    reason: str
    decision_evidence_ids: tuple[str, ...]
    scope_modules: tuple[str, ...]
    scope_evidence_ids: tuple[str, ...]
    method: str = SEMANTIC_METHOD

    def __post_init__(self) -> None:
        if self.candidate_id != self.candidate.candidate_id:
            raise ValueError("accepted artifact candidate identity mismatch")
        if self.decision != SemanticStructuralDecisionKind.ARCHITECTURE_RULE:
            raise ValueError("accepted artifact requires ARCHITECTURE_RULE")
        if self.method != SEMANTIC_METHOD:
            raise ValueError("accepted artifact semantic method mismatch")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("accepted artifact reason is required")
        if self.scope_modules != self.candidate.source_modules:
            raise ValueError("accepted artifact scope mismatch")
        allowed = set(self.candidate.rule.evidence_ids)
        for label, refs in (
            ("decision", self.decision_evidence_ids),
            ("scope", self.scope_evidence_ids),
        ):
            if (not isinstance(refs, tuple) or not refs
                    or refs != tuple(sorted(set(refs))) or not set(refs) <= allowed):
                raise ValueError(f"accepted artifact {label} Evidence mismatch")
        if not set(self.scope_evidence_ids) <= set(self.decision_evidence_ids):
            raise ValueError("accepted artifact scope Evidence is not decision Evidence")


@dataclass(frozen=True, slots=True)
class SemanticStructuralRejection:
    candidate_id: str
    decision: SemanticStructuralDecisionKind
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticStructuralFailure:
    candidate_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class SemanticStructuralRuleResult:
    repository_id: str
    accepted: tuple[SemanticStructuralAcceptance, ...]
    rejections: tuple[SemanticStructuralRejection, ...]
    failures: tuple[SemanticStructuralFailure, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.candidate_id for item in self.accepted)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("accepted candidate IDs must be sorted and unique")
        all_ids = (*ids, *(item.candidate_id for item in self.rejections),
                   *(item.candidate_id for item in self.failures))
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("semantic candidate decisions must be unique")
        if any(item.decision not in {
            SemanticStructuralDecisionKind.INCOHERENT,
            SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE,
        } for item in self.rejections):
            raise ValueError("semantic rejection decision is invalid")
        if any(item.candidate.rule.repository_id != self.repository_id for item in self.accepted):
            raise ValueError("accepted candidate repository_id mismatch")
        refs = {ref for item in self.accepted for ref in item.candidate.rule.evidence_ids}
        refs.update(ref for item in self.accepted for ref in item.decision_evidence_ids)
        refs.update(ref for item in self.accepted for ref in item.scope_evidence_ids)
        refs.update(ref for item in self.rejections for ref in item.evidence_ids)
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(refs)) or any(
            item.repository_id != self.repository_id for item in self.evidence
        ):
            raise ValueError("semantic Evidence closure mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
