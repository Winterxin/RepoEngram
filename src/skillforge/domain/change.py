"""Observed repository changes and separately derived change patterns."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from skillforge.domain.evidence import Evidence, Support
from skillforge.domain.serialization import to_json_value


class FileChangeKind(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    kind: FileChangeKind
    previous_path: str | None = None
    additions: int | None = None
    deletions: int | None = None
    evidence_ids: tuple[str, ...] = ()
    parent_package_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChangeSet:
    repository_id: str
    commit_id: str
    committed_at: str  # ISO 8601 with offset
    files: tuple[FileChange, ...]
    evidence_ids: tuple[str, ...]
    labels: tuple[str, ...] = ()
    noise_score: float | None = None
    parent_ids: tuple[str, ...] = ()
    message: str = ""
    is_merge: bool = False


@dataclass(frozen=True, slots=True)
class GitHistoryResult:
    """A mined history page and the evidence records referenced by its changesets."""

    repository_id: str
    changesets: tuple[ChangeSet, ...]
    evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, object]:
        from skillforge.domain.serialization import to_json_value

        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class ChangePatternItem:
    """One canonical repository path or Python module in a co-change pattern."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("change pattern item value is required")


class ChangePatternKind(StrEnum):
    STATIC_DELEGATION = "static_delegation"
    FILE_CO_CHANGE = "file_co_change"
    MODULE_CO_CHANGE = "module_co_change"
    ADDED_PYTHON_MODULE_ROLE = "added_python_module_role"


@dataclass(frozen=True, slots=True)
class StructuralRoleEvent:
    commit_id: str
    path: str
    evidence_ids: tuple[str, ...]


class NoiseSignalKind(StrEnum):
    LOCK_FILE = "lock_file"
    LARGE_CHANGESET = "large_changeset"
    UBIQUITOUS_ITEM = "ubiquitous_item"


@dataclass(frozen=True, slots=True)
class NoiseSignal:
    kind: NoiseSignalKind
    target: str
    method: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairAssociation:
    first_occurrences: int
    second_occurrences: int
    jaccard: float
    second_given_first: float
    first_given_second: float


@dataclass(frozen=True, slots=True)
class ChangePattern:
    pattern_id: str
    repository_id: str
    kind: ChangePatternKind
    items: tuple[ChangePatternItem, ...]
    supporting_commits: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    support: Support
    support_ratio: float
    item_occurrences: tuple[int, ...]
    pair_association: PairAssociation | None
    structural_events: tuple[StructuralRoleEvent, ...] = ()

    def __post_init__(self) -> None:
        item_values = tuple(item.value for item in self.items)
        role = self.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
        if role:
            if len(item_values) != 1 or not self.structural_events:
                raise ValueError("structural role requires one signature and raw events")
            event_commits = {event.commit_id for event in self.structural_events}
            if event_commits != set(self.supporting_commits) or len(event_commits) < 3:
                raise ValueError("structural events must match supporting commits")
            if len({event.path for event in self.structural_events}) < 3:
                raise ValueError("structural role requires distinct observed paths")
            if not all(set(event.evidence_ids) <= set(self.evidence_ids)
                       for event in self.structural_events):
                raise ValueError("structural event evidence missing from pattern")
        elif self.structural_events:
            raise ValueError("co-change patterns cannot carry structural role events")
        if len(item_values) < 2 and not role:
            raise ValueError("change patterns require at least two items")
        if item_values != tuple(sorted(set(item_values))):
            raise ValueError("change pattern items must be unique and canonical")
        if self.supporting_commits != tuple(sorted(set(self.supporting_commits))):
            raise ValueError("supporting commits must be unique and sorted")
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("evidence IDs must be unique and sorted")
        if self.support.supporting != len(self.supporting_commits):
            raise ValueError("supporting count must match supporting commits")
        if len(self.item_occurrences) != len(self.items):
            raise ValueError("item occurrences must match items")
        if any(count < self.support.supporting for count in self.item_occurrences):
            raise ValueError("item occurrences cannot be less than pattern support")
        expected_ratio = (
            self.support.supporting / self.support.examined if self.support.examined else 0.0
        )
        if not isfinite(self.support_ratio) or self.support_ratio != expected_ratio:
            raise ValueError("support ratio must match raw support")
        if (len(self.items) == 2) != (self.pair_association is not None):
            raise ValueError("pair association is required exactly for pair patterns")


@dataclass(frozen=True, slots=True)
class ChangePatternResult:
    repository_id: str
    patterns: tuple[ChangePattern, ...]
    evidence: tuple[Evidence, ...]
    noise_signals: tuple[NoiseSignal, ...] = ()

    def __post_init__(self) -> None:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("result evidence IDs must be unique and sorted")
        referenced_ids = {
            evidence_id for pattern in self.patterns for evidence_id in pattern.evidence_ids
        }
        referenced_ids.update(
            evidence_id for signal in self.noise_signals
            for evidence_id in signal.evidence_ids
        )
        if referenced_ids != set(evidence_ids):
            raise ValueError("result evidence must match pattern evidence references")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
