"""Mine evidence-backed negative signals without inferring anti-pattern meaning."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from math import isfinite
from pathlib import PurePosixPath

from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind, GitHistoryResult
from skillforge.domain.evidence import Evidence, EvidenceType
from skillforge.domain.negative_patterns import (
    NegativePatternCandidate,
    NegativePatternCandidateResult,
    NegativeSignal,
    NegativeSignalKind,
)
from skillforge.repository.bounded_git_patch_evidence import validate_git_patch_evidence

_COMMIT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", re.IGNORECASE)
_REVERT_SUBJECT = re.compile(r'^\s*Revert\s+"[^"\r\n]+"(?:\s|$)', re.IGNORECASE | re.MULTILINE)
_REVERT_REFERENCE = re.compile(
    r"^\s*This reverts commit\s+([0-9a-f]{7,64})\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CORRECTION_WORD = re.compile(
    r"\b(?:bug[\s_-]*fix(?:es|ed|ing)?|hot[\s_-]*fix(?:es|ed|ing)?|"
    r"fix(?:es|ed|ing)?|regression(?:s)?|work[\s_-]*around)\b",
    re.IGNORECASE,
)
_FILE_PROVENANCES = frozenset({"git_diff", "git_numstat"})
_ALLOWED_PROVENANCES = _FILE_PROVENANCES | {"git_log", "git_parent_package", "git_patch"}


class NegativePatternMiner:
    """Derive bounded message and repeated-edit signals from Git history only."""

    def __init__(self, window_commits: int = 5, window_days: float = 14.0) -> None:
        if (
            isinstance(window_commits, bool)
            or not isinstance(window_commits, int)
            or window_commits < 2
        ):
            raise ValueError("window_commits must be an integer >= 2")
        if (
            isinstance(window_days, bool)
            or not isinstance(window_days, (int, float))
            or not isfinite(window_days)
            or window_days <= 0
        ):
            raise ValueError("window_days must be a positive finite number")
        self.window_commits = window_commits
        self.window_days = float(window_days)

    def mine(self, history: GitHistoryResult) -> NegativePatternCandidateResult:
        evidence_by_id, file_evidence = self._validate_history(history)
        changesets = tuple(history.changesets)
        commit_ids = {changeset.commit_id for changeset in changesets}
        log_evidence = {
            changeset.commit_id: next(
                evidence_id for evidence_id in changeset.evidence_ids
                if evidence_by_id[evidence_id].provenance == "git_log"
            )
            for changeset in changesets
        }

        signals_by_id: dict[str, NegativeSignal] = {}
        for changeset in changesets:
            paths = tuple(sorted(change.path for change in changeset.files))
            commit_path_evidence = {
                change.path: file_evidence[(changeset.commit_id, change.path)]
                for change in changeset.files
            }
            message = changeset.message
            revert_terms = []
            if _REVERT_SUBJECT.search(message):
                revert_terms.append("revert_subject")
            if _REVERT_REFERENCE.search(message):
                revert_terms.append("reverts_commit_line")
            if revert_terms:
                for path in paths or (None,):
                    evidence_ids = {log_evidence[changeset.commit_id]}
                    signal_paths = (path,) if path is not None else ()
                    if path is not None:
                        evidence_ids.update(
                            evidence_id
                            for evidence_id in commit_path_evidence[path]
                            if evidence_by_id[evidence_id].provenance in {"git_diff", "git_patch"}
                        )
                    signal = self._make_signal(
                        history.repository_id,
                        NegativeSignalKind.EXPLICIT_REVERT_HINT,
                        (changeset.commit_id,),
                        signal_paths,
                        evidence_ids,
                        method="commit_message_revert_form_v1; patch reversal not verified",
                        heuristic=True,
                        matched_terms=tuple(sorted(revert_terms)),
                    )
                    signals_by_id[signal.signal_id] = signal

            correction_terms = tuple(sorted({
                re.sub(r"[\s_-]+", " ", match.group(0).casefold())
                for match in _CORRECTION_WORD.finditer(message)
            }))
            if correction_terms:
                for path in paths or (None,):
                    evidence_ids = {log_evidence[changeset.commit_id]}
                    signal_paths = (path,) if path is not None else ()
                    if path is not None:
                        evidence_ids.update(
                            evidence_id
                            for evidence_id in commit_path_evidence[path]
                            if evidence_by_id[evidence_id].provenance in {"git_diff", "git_patch"}
                        )
                    signal = self._make_signal(
                        history.repository_id,
                        NegativeSignalKind.CORRECTION_MESSAGE_HINT,
                        (changeset.commit_id,),
                        signal_paths,
                        evidence_ids,
                        method="case_insensitive_commit_keyword_match_v1; heuristic hint",
                        heuristic=True,
                        matched_terms=correction_terms,
                    )
                    signals_by_id[signal.signal_id] = signal

            for match in _REVERT_REFERENCE.finditer(message):
                referenced_id = self._resolve_reference(match.group(1), commit_ids)
                if referenced_id is None or referenced_id == changeset.commit_id:
                    continue
                for path in paths or (None,):
                    reference_evidence = {
                        log_evidence[changeset.commit_id], log_evidence[referenced_id]
                    }
                    signal_paths = (path,) if path is not None else ()
                    if path is not None:
                        reference_evidence.update(
                            evidence_id
                            for evidence_id in commit_path_evidence[path]
                            if evidence_by_id[evidence_id].provenance == "git_diff"
                        )
                    signal = self._make_signal(
                        history.repository_id,
                        NegativeSignalKind.REVERT_SHA_REFERENCE,
                        (changeset.commit_id, referenced_id),
                        signal_paths,
                        reference_evidence,
                        method="message_sha_reference_resolved_in_input_v1",
                        heuristic=False,
                        referenced_commit_id=referenced_id,
                    )
                    signals_by_id[signal.signal_id] = signal

        self._mine_repeated_edits(
            history.repository_id,
            changesets,
            evidence_by_id,
            file_evidence,
            signals_by_id,
        )
        signals = tuple(sorted(signals_by_id.values(), key=_signal_order))
        candidates = self._build_candidates(
            history.repository_id, signals, len(changesets)
        )
        referenced_evidence = {
            evidence_id for signal in signals for evidence_id in signal.evidence_ids
        }
        result_evidence = tuple(
            evidence_by_id[evidence_id] for evidence_id in sorted(referenced_evidence)
        )
        return NegativePatternCandidateResult(
            history.repository_id, signals, candidates, result_evidence, len(changesets)
        )

    def _validate_history(
        self, history: GitHistoryResult
    ) -> tuple[dict[str, Evidence], dict[tuple[str, str], tuple[str, ...]]]:
        repository_id = history.repository_id
        if not isinstance(repository_id, str) or not repository_id.strip():
            raise ValueError("Git history repository_id is required")

        changeset_by_id: dict[str, ChangeSet] = {}
        for changeset in history.changesets:
            if changeset.repository_id != repository_id:
                raise ValueError(f"Git history changeset repository_id mismatch: "
                                 f"{changeset.commit_id}")
            self._validate_commit_id(changeset.commit_id)
            if changeset.commit_id in changeset_by_id:
                raise ValueError(f"Duplicate commit_id in Git history: {changeset.commit_id}")
            if len(changeset.parent_ids) != len(set(changeset.parent_ids)):
                raise ValueError(f"Duplicate parent id in Git history: {changeset.commit_id}")
            for parent_id in changeset.parent_ids:
                self._validate_commit_id(parent_id)
                if parent_id == changeset.commit_id:
                    raise ValueError(f"Commit cannot parent itself: {changeset.commit_id}")
            if changeset.is_merge != (len(changeset.parent_ids) > 1):
                raise ValueError(f"is_merge does not match parent ids: {changeset.commit_id}")
            if not isinstance(changeset.message, str):
                raise ValueError(f"Invalid commit message: {changeset.commit_id}")
            try:
                committed_at = datetime.fromisoformat(changeset.committed_at)
                if committed_at.utcoffset() is None:
                    raise ValueError("timestamp must include a timezone offset")
                committed_at.astimezone(UTC)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid commit timestamp: {changeset.commit_id}") from error
            self._validate_changeset_files(changeset)
            changeset_by_id[changeset.commit_id] = changeset

        evidence_by_id: dict[str, Evidence] = {}
        for evidence in history.evidence:
            if not isinstance(evidence.evidence_id, str) or not evidence.evidence_id.strip():
                raise ValueError("Git history Evidence requires an evidence_id")
            previous = evidence_by_id.get(evidence.evidence_id)
            if previous is not None:
                if previous != evidence:
                    raise ValueError(f"Conflicting Evidence ID: {evidence.evidence_id}")
                raise ValueError(f"Duplicate Evidence ID: {evidence.evidence_id}")
            evidence_by_id[evidence.evidence_id] = evidence
        referenced_input_ids = {
            evidence_id
            for changeset in history.changesets
            for evidence_id in changeset.evidence_ids
        }
        missing = referenced_input_ids - evidence_by_id.keys()
        if missing:
            raise ValueError(f"Missing Git history Evidence: {', '.join(sorted(missing))}")
        extra = evidence_by_id.keys() - referenced_input_ids
        if extra:
            raise ValueError(f"Unreferenced Git history Evidence: {', '.join(sorted(extra))}")

        file_evidence: dict[tuple[str, str], tuple[str, ...]] = {}
        for evidence in evidence_by_id.values():
            if evidence.repository_id != repository_id:
                raise ValueError(f"Evidence repository_id mismatch: {evidence.evidence_id}")
            if evidence.kind != EvidenceType.GIT:
                raise ValueError(f"Non-Git Evidence in Git history: {evidence.evidence_id}")
            commit_id = evidence.location.commit
            if commit_id not in changeset_by_id:
                raise ValueError(f"Evidence references a missing commit: {evidence.evidence_id}")
            if evidence.provenance not in _ALLOWED_PROVENANCES:
                raise ValueError(f"Invalid Git Evidence provenance: {evidence.evidence_id}")
            if evidence.location.path != ".":
                _validate_path(evidence.location.path)

        for changeset in history.changesets:
            evidence_ids = changeset.evidence_ids
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(f"Duplicate changeset Evidence ID: {changeset.commit_id}")
            commit_evidence = [evidence_by_id[item] for item in evidence_ids]
            for evidence in commit_evidence:
                if evidence.location.commit != changeset.commit_id:
                    raise ValueError(
                        f"Evidence commit mismatch: {changeset.commit_id}:{evidence.evidence_id}"
                    )
            log_items = [item for item in commit_evidence if item.provenance == "git_log"]
            if len(log_items) != 1 or log_items[0].location.path != ".":
                raise ValueError(f"Missing or duplicate git_log Evidence: {changeset.commit_id}")

            expected_ids = {log_items[0].evidence_id}
            for change in changeset.files:
                if len(change.evidence_ids) != len(set(change.evidence_ids)):
                    raise ValueError(
                        f"Duplicate FileChange Evidence ID: {changeset.commit_id}:{change.path}"
                    )
                if not set(change.evidence_ids) <= set(evidence_ids):
                    raise ValueError(
                        f"FileChange Evidence not in ChangeSet: {changeset.commit_id}:{change.path}"
                    )
                linked = [evidence_by_id[item] for item in change.evidence_ids]
                if any(
                    item.repository_id != repository_id
                    or item.location.commit != changeset.commit_id
                    or item.location.path != change.path
                    for item in linked
                ):
                    raise ValueError(
                        f"FileChange Evidence mismatch: {changeset.commit_id}:{change.path}"
                    )
                metadata = [item for item in linked if item.provenance != "git_patch"]
                provenances = {item.provenance for item in metadata}
                if provenances != _FILE_PROVENANCES or len(metadata) != 2:
                    raise ValueError(
                        f"Missing or duplicate git_diff/git_numstat Evidence: "
                        f"{changeset.commit_id}:{change.path}"
                    )
                patches = [item for item in linked if item.provenance == "git_patch"]
                if len(patches) > 1 or any(not validate_git_patch_evidence(
                    item, changeset, change,
                ) for item in patches):
                    raise ValueError("Invalid or duplicate historical patch Evidence")
                expected_ids.update(change.evidence_ids)
                for evidence_id in change.parent_package_evidence_ids:
                    package = evidence_by_id.get(evidence_id)
                    parent_path = str(PurePosixPath(change.path).parent / "__init__.py")
                    if (package is None or evidence_id not in evidence_ids
                            or package.provenance != "git_parent_package"
                            or package.location.commit != changeset.commit_id
                            or package.location.path != parent_path
                            or change.kind != FileChangeKind.ADDED
                            or not change.path.endswith(".py")
                            or PurePosixPath(change.path).name == "__init__.py"
                            or not changeset.parent_ids):
                        raise ValueError("Invalid historical parent package Evidence")
                    expected_ids.add(evidence_id)
                file_evidence[(changeset.commit_id, change.path)] = tuple(
                    item.evidence_id for item in linked
                )
            if set(evidence_ids) != expected_ids:
                raise ValueError(f"Unassociated changeset Evidence: {changeset.commit_id}")

        return evidence_by_id, file_evidence

    @staticmethod
    def _validate_commit_id(commit_id: str) -> None:
        if not isinstance(commit_id, str) or not _COMMIT_ID.fullmatch(commit_id):
            raise ValueError(f"Malformed Git commit id: {commit_id!r}")

    @staticmethod
    def _validate_changeset_files(changeset: ChangeSet) -> None:
        seen_paths: set[str] = set()
        for change in changeset.files:
            _validate_path(change.path)
            if change.path in seen_paths:
                raise ValueError(f"Duplicate changed path: {changeset.commit_id}:{change.path}")
            seen_paths.add(change.path)
            if change.kind == FileChangeKind.RENAMED:
                if change.previous_path is None:
                    raise ValueError(f"Rename is missing previous_path: {change.path}")
                _validate_path(change.previous_path)
                if change.previous_path == change.path:
                    raise ValueError(f"Rename paths must differ: {change.path}")
            elif change.previous_path is not None:
                raise ValueError(f"Non-rename has previous_path: {change.path}")

    def _mine_repeated_edits(
        self,
        repository_id: str,
        changesets: tuple[ChangeSet, ...],
        evidence_by_id: dict[str, Evidence],
        file_evidence: dict[tuple[str, str], tuple[str, ...]],
        signals_by_id: dict[str, NegativeSignal],
    ) -> None:
        non_merge = sorted(
            (changeset for changeset in changesets if not changeset.is_merge),
            key=lambda item: (self._timestamp(item), item.commit_id),
        )
        edits_by_path: dict[str, list[tuple[int, ChangeSet, FileChange]]] = {}
        for position, changeset in enumerate(non_merge):
            for change in changeset.files:
                if change.kind == FileChangeKind.MODIFIED:
                    edits_by_path.setdefault(change.path, []).append(
                        (position, changeset, change)
                    )

        for path, edits in sorted(edits_by_path.items()):
            for first_index, (first_position, first_commit, _first_change) in enumerate(edits):
                for last_position, last_commit, _last_change in edits[first_index + 1:]:
                    if last_position - first_position >= self.window_commits:
                        break
                    elapsed_days = (
                        self._timestamp(last_commit) - self._timestamp(first_commit)
                    ).total_seconds() / 86400
                    if elapsed_days > self.window_days:
                        break
                    commit_ids = tuple(sorted({first_commit.commit_id, last_commit.commit_id}))
                    related_evidence = {
                        evidence_id
                        for commit in (first_commit, last_commit)
                        for evidence_id in file_evidence[(commit.commit_id, path)]
                    }
                    signal = self._make_signal(
                        repository_id,
                        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
                        commit_ids,
                        (path,),
                        related_evidence,
                        method=(
                            f"modified path pair within {self.window_commits} non-merge commits "
                            f"and {self.window_days:g} days; no cause inferred"
                        ),
                        heuristic=False,
                    )
                    existing = signals_by_id.setdefault(signal.signal_id, signal)
                    if existing != signal:
                        raise ValueError(f"Conflicting NegativeSignal ID: {signal.signal_id}")

    @staticmethod
    def _timestamp(changeset: ChangeSet) -> datetime:
        return datetime.fromisoformat(changeset.committed_at).astimezone(UTC)

    @staticmethod
    def _resolve_reference(reference: str, commit_ids: set[str]) -> str | None:
        normalized = reference.lower()
        matching = sorted(
            commit_id for commit_id in commit_ids
            if commit_id.lower() == normalized
            or (len(normalized) >= 7 and commit_id.lower().startswith(normalized))
        )
        return matching[0] if len(matching) == 1 else None

    @staticmethod
    def _make_signal(
        repository_id: str,
        kind: NegativeSignalKind,
        commit_ids: tuple[str, ...],
        affected_paths: tuple[str, ...],
        evidence_ids: set[str],
        *,
        method: str,
        heuristic: bool,
        matched_terms: tuple[str, ...] = (),
        referenced_commit_id: str | None = None,
    ) -> NegativeSignal:
        canonical_commits = tuple(sorted(set(commit_ids)))
        canonical_paths = tuple(sorted(set(affected_paths)))
        canonical_evidence = tuple(sorted(evidence_ids))
        canonical_terms = tuple(sorted(set(matched_terms)))
        identity = "\0".join((
            "negative-signal-v1",
            repository_id,
            kind.value,
            method,
            *canonical_commits,
            "affected_paths",
            *canonical_paths,
            "matched_terms",
            *canonical_terms,
            "referenced_commit_id",
            referenced_commit_id or "",
        ))
        signal_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return NegativeSignal(
            signal_id=signal_id,
            repository_id=repository_id,
            kind=kind,
            commit_ids=canonical_commits,
            affected_paths=canonical_paths,
            evidence_ids=canonical_evidence,
            method=method,
            heuristic=heuristic,
            matched_terms=canonical_terms,
            referenced_commit_id=referenced_commit_id,
        )

    @staticmethod
    def _build_candidates(
        repository_id: str,
        signals: tuple[NegativeSignal, ...],
        examined_commits: int,
    ) -> tuple[NegativePatternCandidate, ...]:
        by_path: dict[str, list[NegativeSignal]] = {}
        pathless: list[NegativeSignal] = []
        for signal in signals:
            if signal.affected_paths:
                for path in signal.affected_paths:
                    by_path.setdefault(path, []).append(signal)
            else:
                pathless.append(signal)

        grouped: list[tuple[tuple[str, ...], tuple[NegativeSignal, ...]]] = [
            ((path,), tuple(sorted(path_signals, key=lambda item: item.signal_id)))
            for path, path_signals in by_path.items()
        ]
        grouped.extend(((), (signal,)) for signal in pathless)

        candidates = []
        for paths, selected_signals in grouped:
            signal_ids = tuple(sorted(signal.signal_id for signal in selected_signals))
            commit_ids = tuple(sorted({
                commit_id for signal in selected_signals for commit_id in signal.commit_ids
            }))
            evidence_ids = tuple(sorted({
                evidence_id for signal in selected_signals for evidence_id in signal.evidence_ids
            }))
            identity = "\0".join((
                "negative-pattern-candidate-v1",
                repository_id,
                "affected_paths",
                *paths,
                "signals",
                *signal_ids,
            ))
            candidates.append(NegativePatternCandidate(
                candidate_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                repository_id=repository_id,
                signal_ids=signal_ids,
                commit_ids=commit_ids,
                affected_paths=paths,
                evidence_ids=evidence_ids,
                signal_count=len(signal_ids),
                involved_commit_count=len(commit_ids),
                examined_commits=examined_commits,
            ))
        return tuple(sorted(candidates, key=_candidate_order))


def _validate_path(path: str) -> None:
    if (
        not isinstance(path, str)
        or not path
        or path == "."
        or path.startswith("/")
        or "\\" in path
        or ":" in path.split("/")[0]
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError(f"Invalid repository-relative POSIX path: {path!r}")


def _signal_order(signal: NegativeSignal) -> tuple[object, ...]:
    return (
        signal.kind.value,
        signal.commit_ids,
        signal.affected_paths,
        signal.signal_id,
    )


def _candidate_order(candidate: NegativePatternCandidate) -> tuple[object, ...]:
    return candidate.affected_paths, candidate.signal_ids, candidate.candidate_id
