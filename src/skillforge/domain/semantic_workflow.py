"""Evidence-bounded semantic workflow request, response, and draft contracts."""

from dataclasses import dataclass, field
from enum import StrEnum

from skillforge.domain.change import ChangePatternKind, NoiseSignal
from skillforge.domain.evidence import Confidence, Evidence, Support
from skillforge.domain.serialization import to_json_value
from skillforge.domain.workflow import (
    CandidatePairAssociation,
    StructuralCoherence,
    StructuralLink,
    WorkflowCandidateMember,
)


@dataclass(frozen=True, slots=True)
class SemanticEvidenceRef:
    evidence_id: str
    kind: str
    path: str
    commit_id: str | None
    line: int | None
    description: str
    provenance: str


@dataclass(frozen=True, slots=True)
class SemanticChangedFile:
    path: str
    kind: str
    previous_path: str | None
    additions: int | None
    deletions: int | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticSupportingCommit:
    commit_id: str
    message: str
    relevant_changed_files: tuple[SemanticChangedFile, ...]
    evidence_ids: tuple[str, ...]
    source_pattern_ids: tuple[str, ...]
    supports_seed: bool


@dataclass(frozen=True, slots=True)
class SemanticSourcePattern:
    pattern_id: str
    kind: ChangePatternKind
    items: tuple[str, ...]
    support: Support
    support_ratio: float
    item_occurrences: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    supporting_commits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticWorkflowRequest:
    repository_id: str
    candidate_id: str
    kind: ChangePatternKind
    members: tuple[WorkflowCandidateMember, ...]
    current_modules: tuple[str, ...]
    support: Support
    support_ratio: float
    pair_associations: tuple[CandidatePairAssociation, ...]
    structural_links: tuple[StructuralLink, ...]
    structural_coherence: StructuralCoherence | None
    noise_hints: tuple[NoiseSignal, ...]
    source_patterns: tuple[SemanticSourcePattern, ...]
    supporting_commits: tuple[SemanticSupportingCommit, ...]
    family_supporting_commits: tuple[SemanticSupportingCommit, ...]
    evidence: tuple[SemanticEvidenceRef, ...]
    validation_feedback: str = field(default="", metadata={"omit_empty": True})

    def __post_init__(self) -> None:
        if (not isinstance(self.validation_feedback, str)
                or len(self.validation_feedback.encode("utf-8")) > 512):
            raise ValueError("validation feedback must be text within 512 bytes")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticWorkflowRole:
    role_id: str
    name: str
    responsibility: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticWorkflowStep:
    step_id: str
    action: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticStepClassification:
    step_id: str
    reason: str
    evidence_ids: tuple[str, ...]
    condition: str | None = None


class SemanticWorkflowScopeKind(StrEnum):
    UNKNOWN = "unknown"
    REPOSITORY = "repository"
    DIRECTORY = "directory"
    MODULE = "module"
    FILE = "file"


@dataclass(frozen=True, slots=True)
class SemanticWorkflowScope:
    kind: SemanticWorkflowScopeKind
    selectors: tuple[str, ...]
    description: str
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticWorkflowException:
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticEntityReferences:
    """Explicit repository entities named by the model's natural-language claims."""

    commit_ids: tuple[str, ...] = ()
    module_names: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticWorkflowResponse:
    """Structured model output before it is accepted as a workflow draft."""

    repository_id: str
    candidate_id: str
    name: str
    name_evidence_ids: tuple[str, ...]
    intent: str
    intent_evidence_ids: tuple[str, ...]
    roles: tuple[SemanticWorkflowRole, ...]
    ordered_steps: tuple[SemanticWorkflowStep, ...]
    required_steps: tuple[SemanticStepClassification, ...]
    optional_steps: tuple[SemanticStepClassification, ...]
    conditional_steps: tuple[SemanticStepClassification, ...]
    scope: SemanticWorkflowScope
    exceptions: tuple[SemanticWorkflowException, ...]
    confidence_score: float
    references: SemanticEntityReferences = SemanticEntityReferences()

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


class SemanticDecisionKind(StrEnum):
    WORKFLOW = "WORKFLOW"
    INCOHERENT = "INCOHERENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class SemanticDecisionResponse:
    """Model decision before SkillForge validates the claim and optional workflow."""

    repository_id: str
    candidate_id: str
    decision: SemanticDecisionKind
    reason: str
    decision_evidence_ids: tuple[str, ...]
    confidence_score: float
    support: Support
    workflow: SemanticWorkflowResponse | None

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticWorkflowRejection:
    repository_id: str
    candidate_id: str
    decision: SemanticDecisionKind
    reason: str
    evidence_ids: tuple[str, ...]
    confidence: Confidence
    support: Support


@dataclass(frozen=True, slots=True)
class SemanticWorkflowDraft:
    draft_id: str
    repository_id: str
    candidate_id: str
    name: str
    name_evidence_ids: tuple[str, ...]
    intent: str
    intent_evidence_ids: tuple[str, ...]
    roles: tuple[SemanticWorkflowRole, ...]
    ordered_steps: tuple[SemanticWorkflowStep, ...]
    required_steps: tuple[SemanticStepClassification, ...]
    optional_steps: tuple[SemanticStepClassification, ...]
    conditional_steps: tuple[SemanticStepClassification, ...]
    scope: SemanticWorkflowScope
    exceptions: tuple[SemanticWorkflowException, ...]
    evidence_ids: tuple[str, ...]
    support: Support
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class SemanticWorkflowResult:
    repository_id: str
    workflows: tuple[SemanticWorkflowDraft, ...]
    evidence: tuple[Evidence, ...]
    rejections: tuple[SemanticWorkflowRejection, ...] = ()

    def __post_init__(self) -> None:
        draft_ids = tuple(item.draft_id for item in self.workflows)
        if len(draft_ids) != len(set(draft_ids)):
            raise ValueError("semantic workflow draft IDs must be unique")
        if any(item.repository_id != self.repository_id for item in self.workflows):
            raise ValueError("semantic workflow repository_id mismatch")
        candidate_ids = tuple(item.candidate_id for item in (*self.workflows, *self.rejections))
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("semantic decision candidate IDs must be unique")
        if any(item.repository_id != self.repository_id for item in self.rejections):
            raise ValueError("semantic rejection repository_id mismatch")
        if any(item.decision == SemanticDecisionKind.WORKFLOW for item in self.rejections):
            raise ValueError("semantic rejection must have a rejection decision")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("semantic workflow Evidence must be unique and sorted")
        referenced = {
            evidence_id for item in (*self.workflows, *self.rejections)
            for evidence_id in item.evidence_ids
        }
        if referenced != set(evidence_ids):
            raise ValueError("semantic workflow Evidence must match draft references")
        if any(item.repository_id != self.repository_id for item in self.evidence):
            raise ValueError("semantic workflow Evidence repository_id mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
