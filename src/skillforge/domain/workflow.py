"""Evidence-backed, semantic-neutral workflow candidate contracts."""

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.change import (
    ChangePattern,
    ChangePatternKind,
    NoiseSignal,
    StructuralRoleEvent,
)
from skillforge.domain.evidence import Confidence, Evidence, Support
from skillforge.domain.project_dna import ProjectScope
from skillforge.domain.serialization import to_json_value


@dataclass(frozen=True, slots=True)
class WorkflowCandidateMember:
    value: str
    current_module: str | None = None


class StructuralLinkKind(StrEnum):
    INTERNAL_IMPORT = "internal_import"


@dataclass(frozen=True, slots=True)
class StructuralLink:
    source: str
    target: str
    kind: StructuralLinkKind
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuralCoherence:
    linked_pairs: int
    possible_pairs: int
    link_ratio: float


@dataclass(frozen=True, slots=True)
class CandidatePairAssociation:
    first: str
    second: str
    source_pattern_id: str
    first_occurrences: int
    second_occurrences: int
    jaccard: float
    second_given_first: float
    first_given_second: float


@dataclass(frozen=True, slots=True)
class WorkflowCandidate:
    candidate_id: str
    repository_id: str
    kind: ChangePatternKind
    members: tuple[WorkflowCandidateMember, ...]
    seed_pattern_id: str
    source_pattern_ids: tuple[str, ...]
    supporting_commits: tuple[str, ...]
    support: Support
    support_ratio: float
    pair_associations: tuple[CandidatePairAssociation, ...]
    current_modules: tuple[str, ...]
    structural_links: tuple[StructuralLink, ...]
    structural_coherence: StructuralCoherence | None
    noise_signals: tuple[NoiseSignal, ...]
    evidence_ids: tuple[str, ...]
    structural_events: tuple[StructuralRoleEvent, ...] = ()
    observation: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowCandidateResult:
    repository_id: str
    candidates: tuple[WorkflowCandidate, ...]
    source_patterns: tuple[ChangePattern, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        patterns_by_id = {pattern.pattern_id: pattern for pattern in self.source_patterns}
        if len(patterns_by_id) != len(self.source_patterns):
            raise ValueError("source pattern IDs must be unique")
        if tuple(patterns_by_id) != tuple(sorted(patterns_by_id)):
            raise ValueError("source patterns must be sorted by pattern ID")
        if any(pattern.repository_id != self.repository_id for pattern in self.source_patterns):
            raise ValueError("source pattern repository_id mismatch")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("result evidence IDs must be unique and sorted")
        referenced: set[str] = set()
        referenced_patterns: set[str] = set()
        for candidate in self.candidates:
            if candidate.repository_id != self.repository_id:
                raise ValueError("candidate repository_id mismatch")
            static = candidate.kind == ChangePatternKind.STATIC_DELEGATION
            if static and (candidate.seed_pattern_id or candidate.source_pattern_ids
                           or candidate.supporting_commits or candidate.pair_associations
                           or candidate.support.supporting < 2 or not candidate.observation):
                raise ValueError("static delegation requires repeated source-only observations")
            if not static and candidate.seed_pattern_id not in candidate.source_pattern_ids:
                raise ValueError("seed pattern missing from candidate source family")
            seed = patterns_by_id.get(candidate.seed_pattern_id)
            if candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE:
                if (seed is None or candidate.structural_events != seed.structural_events
                        or candidate.supporting_commits != seed.supporting_commits
                        or candidate.support != seed.support or candidate.pair_associations):
                    raise ValueError("structural candidate must preserve seed observations")
            if candidate.source_pattern_ids != tuple(sorted(set(candidate.source_pattern_ids))):
                raise ValueError("candidate source pattern IDs must be unique and sorted")
            if candidate.current_modules != tuple(sorted(set(candidate.current_modules))):
                raise ValueError("candidate current modules must be unique and sorted")
            mapped_modules = tuple(sorted({
                member.current_module for member in candidate.members
                if member.current_module is not None
            }))
            if candidate.current_modules != mapped_modules:
                raise ValueError("candidate current modules must match member mappings")
            family_ids = set(candidate.source_pattern_ids)
            if not {pair.source_pattern_id for pair in candidate.pair_associations} <= family_ids:
                raise ValueError("pair association source pattern missing from candidate family")
            referenced_patterns.update(family_ids)
            referenced.update(candidate.evidence_ids)
            for link in candidate.structural_links:
                if not set(link.evidence_ids) <= set(candidate.evidence_ids):
                    raise ValueError("structural link evidence missing from candidate")
            for signal in candidate.noise_signals:
                if not set(signal.evidence_ids) <= set(candidate.evidence_ids):
                    raise ValueError("noise evidence missing from candidate")
            for pattern_id in family_ids & patterns_by_id.keys():
                if not set(patterns_by_id[pattern_id].evidence_ids) <= set(candidate.evidence_ids):
                    raise ValueError("source pattern evidence missing from candidate")
        if referenced_patterns != patterns_by_id.keys():
            raise ValueError("source patterns must match candidate references")
        if referenced != set(evidence_ids):
            raise ValueError("result evidence must match candidate references")
        if any(item.repository_id != self.repository_id for item in self.evidence):
            raise ValueError("Evidence repository_id mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    step_id: str
    instruction: str
    evidence_ids: tuple[str, ...]
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class Workflow:
    workflow_id: str
    repository_id: str
    intent: str
    scope: ProjectScope
    required_steps: tuple[WorkflowStep, ...]
    optional_steps: tuple[WorkflowStep, ...]
    evidence_ids: tuple[str, ...]
    support: Support
    confidence: Confidence
