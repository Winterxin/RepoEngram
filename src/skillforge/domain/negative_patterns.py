"""Evidence-backed negative-signal and candidate contracts."""

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.evidence import Evidence
from skillforge.domain.serialization import to_json_value


class NegativeSignalKind(StrEnum):
    EXPLICIT_REVERT_HINT = "explicit_revert_hint"
    CORRECTION_MESSAGE_HINT = "correction_message_hint"
    SHORT_WINDOW_REPEAT_EDIT = "short_window_repeat_edit"
    REVERT_SHA_REFERENCE = "revert_sha_reference"


@dataclass(frozen=True, slots=True)
class NegativeSignal:
    signal_id: str
    repository_id: str
    kind: NegativeSignalKind
    commit_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    method: str
    heuristic: bool
    matched_terms: tuple[str, ...] = ()
    referenced_commit_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.signal_id, "signal_id")
        _required_text(self.repository_id, "repository_id")
        _required_text(self.method, "method")
        if not isinstance(self.kind, NegativeSignalKind):
            raise ValueError("kind must be a NegativeSignalKind")
        if not isinstance(self.heuristic, bool):
            raise ValueError("heuristic must be a bool")
        _canonical_values(self.commit_ids, "commit_ids", minimum=1)
        _canonical_values(self.affected_paths, "affected_paths")
        _canonical_values(self.evidence_ids, "evidence_ids", minimum=1)
        _canonical_values(self.matched_terms, "matched_terms")
        expected_heuristic = self.kind in {
            NegativeSignalKind.EXPLICIT_REVERT_HINT,
            NegativeSignalKind.CORRECTION_MESSAGE_HINT,
        }
        if self.heuristic != expected_heuristic:
            raise ValueError("heuristic must match the signal kind")
        if self.kind == NegativeSignalKind.REVERT_SHA_REFERENCE:
            if self.referenced_commit_id not in self.commit_ids or len(self.commit_ids) < 2:
                raise ValueError("revert SHA reference must identify a referenced input commit")
        elif self.referenced_commit_id is not None:
            raise ValueError("referenced_commit_id is only valid for a revert SHA reference")


@dataclass(frozen=True, slots=True)
class NegativePatternCandidate:
    candidate_id: str
    repository_id: str
    signal_ids: tuple[str, ...]
    commit_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    signal_count: int
    involved_commit_count: int
    examined_commits: int

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _required_text(self.repository_id, "repository_id")
        _canonical_values(self.signal_ids, "signal_ids", minimum=1)
        _canonical_values(self.commit_ids, "commit_ids", minimum=1)
        _canonical_values(self.affected_paths, "affected_paths")
        _canonical_values(self.evidence_ids, "evidence_ids", minimum=1)
        _nonnegative_count(self.signal_count, "signal_count")
        _nonnegative_count(self.involved_commit_count, "involved_commit_count")
        _nonnegative_count(self.examined_commits, "examined_commits")
        if self.signal_count != len(self.signal_ids):
            raise ValueError("signal_count must equal the number of candidate signals")
        if self.involved_commit_count != len(self.commit_ids):
            raise ValueError("involved_commit_count must equal the number of involved commits")
        if self.examined_commits < self.involved_commit_count:
            raise ValueError("examined_commits must include the involved commits")


@dataclass(frozen=True, slots=True)
class NegativePatternCandidateResult:
    repository_id: str
    signals: tuple[NegativeSignal, ...]
    candidates: tuple[NegativePatternCandidate, ...]
    evidence: tuple[Evidence, ...]
    examined_commits: int

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        _nonnegative_count(self.examined_commits, "examined_commits")
        signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("duplicate NegativeSignal ID")
        if self.signals != tuple(sorted(self.signals, key=_signal_order)):
            raise ValueError("signals must use stable ordering")
        for signal in self.signals:
            if signal.repository_id != self.repository_id:
                raise ValueError(f"NegativeSignal repository_id mismatch: {signal.signal_id}")

        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("duplicate NegativePatternCandidate ID")
        if self.candidates != tuple(sorted(self.candidates, key=_candidate_order)):
            raise ValueError("candidates must use stable ordering")

        signal_by_id = {signal.signal_id: signal for signal in self.signals}
        referenced_signal_ids: set[str] = set()
        for candidate in self.candidates:
            if candidate.repository_id != self.repository_id:
                raise ValueError(
                    f"NegativePatternCandidate repository_id mismatch: {candidate.candidate_id}"
                )
            selected_signals = []
            for signal_id in candidate.signal_ids:
                signal = signal_by_id.get(signal_id)
                if signal is None:
                    raise ValueError(f"Missing candidate signal: {signal_id}")
                selected_signals.append(signal)
                referenced_signal_ids.add(signal_id)

            expected_commits = tuple(sorted({
                commit_id for signal in selected_signals for commit_id in signal.commit_ids
            }))
            if candidate.signal_count != len(selected_signals):
                raise ValueError("candidate signal_count must match its signal closure")
            if candidate.commit_ids != expected_commits:
                raise ValueError("candidate commit_ids must match its signal closure")
            if candidate.involved_commit_count != len(expected_commits):
                raise ValueError("candidate involved_commit_count must match its signal closure")
            if candidate.examined_commits != self.examined_commits:
                raise ValueError("candidate examined_commits must match result examined_commits")
            expected_evidence = tuple(sorted({
                evidence_id for signal in selected_signals for evidence_id in signal.evidence_ids
            }))
            if candidate.evidence_ids != expected_evidence:
                raise ValueError("candidate evidence_ids must match its signal closure")
            if candidate.affected_paths:
                if len(candidate.affected_paths) != 1 or any(
                    candidate.affected_paths[0] not in signal.affected_paths
                    for signal in selected_signals
                ):
                    raise ValueError("candidate signals must share its exact affected path")
            elif len(selected_signals) != 1 or selected_signals[0].affected_paths:
                raise ValueError("pathless signals must remain separate candidates")
        if referenced_signal_ids != set(signal_ids):
            raise ValueError("every signal must be represented by a candidate")

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            evidence_by_id: dict[str, Evidence] = {}
            for item in self.evidence:
                previous = evidence_by_id.setdefault(item.evidence_id, item)
                if previous != item:
                    raise ValueError(f"Conflicting Evidence ID: {item.evidence_id}")
            raise ValueError("duplicate Evidence ID")
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("result evidence IDs must be unique and sorted")
        for item in self.evidence:
            if item.repository_id != self.repository_id:
                raise ValueError(f"Evidence repository_id mismatch: {item.evidence_id}")

        referenced_evidence = {
            evidence_id for signal in self.signals for evidence_id in signal.evidence_ids
        }
        referenced_evidence.update(
            evidence_id for candidate in self.candidates
            for evidence_id in candidate.evidence_ids
        )
        if referenced_evidence != set(evidence_ids):
            missing = referenced_evidence - set(evidence_ids)
            extra = set(evidence_ids) - referenced_evidence
            if missing:
                raise ValueError(f"Missing result Evidence: {', '.join(sorted(missing))}")
            raise ValueError(f"Unreferenced result Evidence: {', '.join(sorted(extra))}")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _nonnegative_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _canonical_values(values: tuple[str, ...], name: str, minimum: int = 0) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(values) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} value(s)")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and sorted")


def _signal_order(signal: NegativeSignal) -> tuple[object, ...]:
    return (
        signal.kind.value,
        signal.commit_ids,
        signal.affected_paths,
        signal.signal_id,
    )


def _candidate_order(candidate: NegativePatternCandidate) -> tuple[object, ...]:
    return candidate.affected_paths, candidate.signal_ids, candidate.candidate_id
