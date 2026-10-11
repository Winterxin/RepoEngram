"""Evidence-bounded semantic Anti-pattern contracts."""

from dataclasses import dataclass, field
from enum import StrEnum

from skillforge.domain.evidence import Confidence, Evidence
from skillforge.domain.negative_patterns import NegativeSignalKind
from skillforge.domain.serialization import to_json_value


class SemanticAntiPatternDecisionKind(StrEnum):
    ANTI_PATTERN = "ANTI_PATTERN"
    INCOHERENT = "INCOHERENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SemanticAntiPatternScopeKind(StrEnum):
    UNKNOWN = "UNKNOWN"
    REPOSITORY = "REPOSITORY"
    FILE = "FILE"


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternEvidenceRef:
    evidence_id: str
    kind: str
    path: str
    commit_id: str | None
    line: int | None
    description: str
    provenance: str


@dataclass(frozen=True, slots=True)
class SemanticNegativeSignal:
    signal_id: str
    kind: NegativeSignalKind
    commit_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    method: str
    heuristic: bool
    matched_terms: tuple[str, ...]
    referenced_commit_id: str | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticNegativeChangedFile:
    path: str
    kind: str | None
    previous_path: str | None
    additions: int | None
    deletions: int | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticNegativeCommit:
    commit_id: str
    message: str | None
    relevant_changed_files: tuple[SemanticNegativeChangedFile, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternRequest:
    """One candidate's facts, restricted to its negative-signal Evidence closure."""

    repository_id: str
    candidate_id: str
    affected_paths: tuple[str, ...]
    signal_count: int
    involved_commit_count: int
    examined_commits: int
    signals: tuple[SemanticNegativeSignal, ...]
    commits: tuple[SemanticNegativeCommit, ...]
    evidence: tuple[SemanticAntiPatternEvidenceRef, ...]
    validation_feedback: str = field(default="", metadata={"omit_empty": True})

    def __post_init__(self) -> None:
        if (not isinstance(self.validation_feedback, str)
                or len(self.validation_feedback.encode("utf-8")) > 512):
            raise ValueError("validation feedback must be text within 512 bytes")
        _required_text(self.repository_id, "repository_id")
        _required_text(self.candidate_id, "candidate_id")
        _sorted_unique(self.affected_paths, "affected_paths")
        for name, value in (
            ("signal_count", self.signal_count),
            ("involved_commit_count", self.involved_commit_count),
            ("examined_commits", self.examined_commits),
        ):
            _nonnegative(value, name)
        signal_ids = tuple(item.signal_id for item in self.signals)
        if signal_ids != tuple(sorted(set(signal_ids))):
            raise ValueError("request signals must be unique and sorted")
        if self.signal_count != len(self.signals):
            raise ValueError("request signal_count must match its signals")
        commit_ids = tuple(item.commit_id for item in self.commits)
        if commit_ids != tuple(sorted(set(commit_ids))):
            raise ValueError("request commits must be unique and sorted")
        expected_commits = {
            commit_id for signal in self.signals for commit_id in signal.commit_ids
        }
        if expected_commits != set(commit_ids):
            raise ValueError("request commits must match its signal commit closure")
        if self.involved_commit_count != len(expected_commits):
            raise ValueError("request involved_commit_count must match its commit closure")
        if self.examined_commits < self.involved_commit_count:
            raise ValueError("request examined_commits must include involved commits")

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("request Evidence must be unique and sorted")
        signal_evidence = {
            evidence_id for signal in self.signals for evidence_id in signal.evidence_ids
        }
        if signal_evidence != set(evidence_ids):
            raise ValueError("request Evidence must match its signal closure")
        allowed_evidence = set(evidence_ids)
        allowed_paths = set(self.affected_paths)
        for signal in self.signals:
            if not signal.evidence_ids or not set(signal.evidence_ids) <= allowed_evidence:
                raise ValueError(f"request signal has invalid Evidence: {signal.signal_id}")
            if not set(signal.commit_ids) <= expected_commits:
                raise ValueError(f"request signal has an unknown commit: {signal.signal_id}")
            if signal.affected_paths and not set(signal.affected_paths) <= allowed_paths:
                raise ValueError(f"request signal has an unknown path: {signal.signal_id}")
        for commit in self.commits:
            if not set(commit.evidence_ids) <= allowed_evidence:
                raise ValueError(f"request commit has unknown Evidence: {commit.commit_id}")
            for changed_file in commit.relevant_changed_files:
                if not set(changed_file.evidence_ids) <= allowed_evidence:
                    raise ValueError(
                        f"request changed file has unknown Evidence: {changed_file.path}"
                    )

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternScope:
    kind: SemanticAntiPatternScopeKind
    selectors: tuple[str, ...]
    description: str
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternException:
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternResponse:
    """The model's proposed Anti-pattern before provenance validation."""

    name: str
    name_evidence_ids: tuple[str, ...]
    description: str
    description_evidence_ids: tuple[str, ...]
    avoid_behavior: str
    avoid_behavior_evidence_ids: tuple[str, ...]
    negative_evidence_summary: str
    negative_evidence_ids: tuple[str, ...]
    scope: SemanticAntiPatternScope
    exceptions: tuple[SemanticAntiPatternException, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternDecisionResponse:
    """Untrusted semantic decision envelope returned by a model port."""

    repository_id: str
    candidate_id: str
    decision: SemanticAntiPatternDecisionKind
    reason: str
    decision_evidence_ids: tuple[str, ...]
    confidence_score: float
    anti_pattern: SemanticAntiPatternResponse | None

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternRejection:
    repository_id: str
    candidate_id: str
    decision: SemanticAntiPatternDecisionKind
    reason: str
    evidence_ids: tuple[str, ...]
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternDraft:
    draft_id: str
    repository_id: str
    candidate_id: str
    name: str
    name_evidence_ids: tuple[str, ...]
    description: str
    description_evidence_ids: tuple[str, ...]
    avoid_behavior: str
    avoid_behavior_evidence_ids: tuple[str, ...]
    negative_evidence_summary: str
    negative_evidence_ids: tuple[str, ...]
    scope: SemanticAntiPatternScope
    exceptions: tuple[SemanticAntiPatternException, ...]
    decision_reason: str
    decision_evidence_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class SemanticAntiPatternResult:
    repository_id: str
    anti_patterns: tuple[SemanticAntiPatternDraft, ...]
    rejections: tuple[SemanticAntiPatternRejection, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        draft_ids = tuple(item.draft_id for item in self.anti_patterns)
        if len(draft_ids) != len(set(draft_ids)):
            raise ValueError("semantic Anti-pattern draft IDs must be unique")
        if any(item.repository_id != self.repository_id for item in self.anti_patterns):
            raise ValueError("semantic Anti-pattern repository_id mismatch")
        candidate_ids = tuple(
            item.candidate_id for item in (*self.anti_patterns, *self.rejections)
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("semantic decision candidate IDs must be unique")
        if any(item.repository_id != self.repository_id for item in self.rejections):
            raise ValueError("semantic rejection repository_id mismatch")
        if any(
            item.decision not in {
                SemanticAntiPatternDecisionKind.INCOHERENT,
                SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
            }
            for item in self.rejections
        ):
            raise ValueError("semantic rejection must have a rejection decision")
        decisions = (*self.anti_patterns, *self.rejections)
        for item in decisions:
            if item.confidence.method != "llm_semantic_antipattern_v1":
                raise ValueError("semantic Anti-pattern confidence method mismatch")
            if item.confidence.support is not None:
                raise ValueError("semantic Anti-pattern confidence must not contain Support")
            if item.evidence_ids != tuple(sorted(set(item.evidence_ids))):
                raise ValueError("semantic decision Evidence IDs must be unique and sorted")
        for item in self.anti_patterns:
            actual_references = {
                *item.name_evidence_ids,
                *item.description_evidence_ids,
                *item.avoid_behavior_evidence_ids,
                *item.negative_evidence_ids,
                *item.scope.evidence_ids,
                *item.decision_evidence_ids,
                *(
                    evidence_id
                    for exception in item.exceptions
                    for evidence_id in exception.evidence_ids
                ),
            }
            if actual_references != set(item.evidence_ids):
                raise ValueError("semantic Anti-pattern Evidence must match output references")
        if any(not item.evidence_ids for item in self.rejections):
            raise ValueError("semantic rejection must cite Evidence")

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("semantic Anti-pattern Evidence must be unique and sorted")
        referenced = {
            evidence_id for item in decisions for evidence_id in item.evidence_ids
        }
        if referenced != set(evidence_ids):
            raise ValueError("semantic Anti-pattern Evidence must match output references")
        if any(item.repository_id != self.repository_id for item in self.evidence):
            raise ValueError("semantic Anti-pattern Evidence repository_id mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _sorted_unique(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must contain non-empty strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and sorted")
