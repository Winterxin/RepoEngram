"""Deterministically mine bounded frequent file and current-module itemsets."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import PurePosixPath

from skillforge.domain.change import (
    ChangePattern,
    ChangePatternItem,
    ChangePatternKind,
    ChangePatternResult,
    ChangeSet,
    FileChangeKind,
    GitHistoryResult,
    NoiseSignal,
    NoiseSignalKind,
    PairAssociation,
    StructuralRoleEvent,
)
from skillforge.domain.evidence import Evidence, EvidenceType, Support
from skillforge.domain.static_analysis import PythonModuleAnalysis, StaticAnalysisResult

PatternKey = tuple[ChangePatternKind, tuple[str, ...]]
Transaction = tuple[str, ChangePatternKind, dict[str, set[str]], set[str]]
LOCK_FILES = frozenset({
    "poetry.lock", "Pipfile.lock", "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "Cargo.lock",
})


class ChangePatternMiner:
    """Mine raw historical transactions with exact current module mappings."""

    def __init__(
        self, min_support: int = 2, max_pattern_size: int = 5,
        large_changeset_threshold: int = 50,
        ubiquitous_threshold: float = 0.8,
        ubiquitous_min_occurrences: int = 5,
    ) -> None:
        if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        if (isinstance(max_pattern_size, bool) or not isinstance(max_pattern_size, int)
                or max_pattern_size < 2):
            raise ValueError("max_pattern_size must be an integer >= 2")
        if (isinstance(large_changeset_threshold, bool)
                or not isinstance(large_changeset_threshold, int)
                or large_changeset_threshold < 1):
            raise ValueError("large_changeset_threshold must be a positive integer")
        if not 0 < ubiquitous_threshold <= 1:
            raise ValueError("ubiquitous_threshold must be in (0, 1]")
        if (isinstance(ubiquitous_min_occurrences, bool)
                or not isinstance(ubiquitous_min_occurrences, int)
                or ubiquitous_min_occurrences < 1):
            raise ValueError("ubiquitous_min_occurrences must be a positive integer")
        self.min_support = min_support
        self.max_pattern_size = max_pattern_size
        self.large_changeset_threshold = large_changeset_threshold
        self.ubiquitous_threshold = ubiquitous_threshold
        self.ubiquitous_min_occurrences = ubiquitous_min_occurrences

    def mine(
        self, history: GitHistoryResult, static_analysis: StaticAnalysisResult
    ) -> ChangePatternResult:
        if history.repository_id != static_analysis.repository_id:
            raise ValueError("Git history and static analysis must describe the same repository")
        commit_ids = [changeset.commit_id for changeset in history.changesets]
        if len(commit_ids) != len(set(commit_ids)):
            raise ValueError("Duplicate commit_id in Git history changesets")

        module_by_path = {
            module.path: module for module in static_analysis.modules if module.module is not None
        }
        git_evidence = {item.evidence_id: item for item in history.evidence}
        static_evidence = {item.evidence_id: item for item in static_analysis.evidence}
        eligible = tuple(
            changeset for changeset in history.changesets
            if changeset.files and not changeset.is_merge
        )
        log_ids_by_commit = {
            changeset.commit_id: self._validate_evidence(
                changeset, git_evidence, static_evidence, module_by_path,
                history.repository_id,
            )
            for changeset in eligible
        }
        transactions: list[Transaction] = []
        occurrence: dict[tuple[ChangePatternKind, str], int] = {}
        for changeset in eligible:
            log_ids = log_ids_by_commit[changeset.commit_id]
            file_items: dict[str, set[str]] = {}
            module_items: dict[str, set[str]] = {}
            for change in changeset.files:
                file_items.setdefault(change.path, set()).update(change.evidence_ids)
                module = module_by_path.get(change.path)
                if module is not None:
                    module_items.setdefault(module.module, set()).update(
                        (*change.evidence_ids, *module.evidence_ids)
                    )
            for kind, items in (
                (ChangePatternKind.FILE_CO_CHANGE, file_items),
                (ChangePatternKind.MODULE_CO_CHANGE, module_items),
            ):
                transactions.append((changeset.commit_id, kind, items, log_ids))
                for value in items:
                    key = (kind, value)
                    occurrence[key] = occurrence.get(key, 0) + 1

        commits_by_pattern: dict[PatternKey, set[str]] = {}
        evidence_by_pattern: dict[PatternKey, set[str]] = {}
        for kind in ChangePatternKind:
            frequent_previous = {
                (value,) for (item_kind, value), count in occurrence.items()
                if item_kind == kind and count >= self.min_support
            }
            for size in range(2, self.max_pattern_size + 1):
                if not frequent_previous:
                    break
                if size == 2:
                    singleton_values = sorted(item[0] for item in frequent_previous)
                    candidates = {
                        pair for pair in combinations(singleton_values, 2)
                    }
                else:
                    candidates = {
                        candidate for candidate in combinations(
                            sorted({value for item in frequent_previous for value in item}), size
                        )
                        if all(subset in frequent_previous
                               for subset in combinations(candidate, size - 1))
                    }
                frequent_current: set[tuple[str, ...]] = set()
                for values in sorted(candidates):
                    matching = [
                        transaction for transaction in transactions
                        if transaction[1] == kind and set(values) <= transaction[2].keys()
                    ]
                    if len(matching) < self.min_support:
                        continue
                    key = (kind, values)
                    commits_by_pattern[key] = {commit for commit, _, _, _ in matching}
                    member_evidence = evidence_by_pattern.setdefault(key, set())
                    for _, _, items, log_ids in matching:
                        member_evidence.update(log_ids)
                        for value in values:
                            member_evidence.update(items[value])
                    frequent_current.add(values)
                frequent_previous = frequent_current

        examined = len(eligible)
        patterns = []
        for (kind, values), commits in commits_by_pattern.items():
            together = len(commits)
            counts = tuple(occurrence[(kind, value)] for value in values)
            association = None
            if len(values) == 2:
                first, second = counts
                association = PairAssociation(
                    first, second, together / (first + second - together),
                    together / first, together / second,
                )
            identity = "\0".join((history.repository_id, kind.value, *values))
            patterns.append(ChangePattern(
                pattern_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                repository_id=history.repository_id,
                kind=kind,
                items=tuple(ChangePatternItem(value) for value in values),
                supporting_commits=tuple(sorted(commits)),
                evidence_ids=tuple(sorted(evidence_by_pattern[(kind, values)])),
                support=Support(together, examined),
                support_ratio=together / examined if examined else 0.0,
                item_occurrences=counts,
                pair_association=association,
            ))
        patterns.extend(self._structural_patterns(
            history, module_by_path, eligible, log_ids_by_commit, git_evidence, static_evidence,
        ))
        patterns.sort(key=lambda pattern: (
            -pattern.support.supporting, pattern.kind.value,
            tuple(item.value for item in pattern.items),
        ))
        signals: list[NoiseSignal] = []
        for changeset in eligible:
            if len({change.path for change in changeset.files}) > self.large_changeset_threshold:
                ids = set(log_ids_by_commit[changeset.commit_id])
                ids.update(
                    evidence_id for change in changeset.files
                    for evidence_id in change.evidence_ids
                )
                signals.append(NoiseSignal(
                    NoiseSignalKind.LARGE_CHANGESET, changeset.commit_id,
                    f"distinct changed paths > {self.large_changeset_threshold}",
                    tuple(sorted(ids)),
                ))
        for kind in ChangePatternKind:
            for (item_kind, value), count in sorted(occurrence.items()):
                if item_kind != kind:
                    continue
                item_transactions = [
                    transaction for transaction in transactions
                    if transaction[1] == kind and value in transaction[2]
                ]
                item_evidence = tuple(sorted({
                    evidence_id for _, _, items, _ in item_transactions
                    for evidence_id in items[value]
                }))
                if kind == ChangePatternKind.FILE_CO_CHANGE and value.split("/")[-1] in LOCK_FILES:
                    signals.append(NoiseSignal(
                        NoiseSignalKind.LOCK_FILE, value,
                        "exact lock filename", item_evidence,
                    ))
                if (count >= self.ubiquitous_min_occurrences and examined
                        and count / examined >= self.ubiquitous_threshold):
                    signals.append(NoiseSignal(
                        NoiseSignalKind.UBIQUITOUS_ITEM, f"{kind.value}:{value}",
                        f"occurrence / eligible commits >= {self.ubiquitous_threshold}; "
                        f"occurrence >= {self.ubiquitous_min_occurrences}",
                        item_evidence,
                    ))
        signals.sort(key=lambda signal: (signal.kind.value, signal.target))
        referenced = {evidence_id for pattern in patterns for evidence_id in pattern.evidence_ids}
        referenced.update(
            evidence_id for signal in signals for evidence_id in signal.evidence_ids
        )
        evidence_by_id: dict[str, Evidence] = {}
        for evidence in (*history.evidence, *static_analysis.evidence):
            if evidence.evidence_id not in referenced:
                continue
            if evidence.repository_id != history.repository_id:
                raise ValueError(f"Evidence repository_id mismatch: {evidence.evidence_id}")
            existing = evidence_by_id.setdefault(evidence.evidence_id, evidence)
            if existing != evidence:
                raise ValueError(f"Conflicting evidence ID: {evidence.evidence_id}")
        missing = referenced - evidence_by_id.keys()
        if missing:
            raise ValueError(f"Missing pattern evidence IDs: {', '.join(sorted(missing))}")
        result_evidence = tuple(evidence_by_id[key] for key in sorted(referenced))
        return ChangePatternResult(
            history.repository_id, tuple(patterns), result_evidence, tuple(signals)
        )

    def _structural_patterns(
        self, history, module_by_path, eligible, log_ids_by_commit, git_evidence, static_evidence,
    ) -> list[ChangePattern]:
        """Observe additions to existing regular packages, with no business interpretation."""
        grouped: dict[str, list[StructuralRoleEvent]] = {}
        for changeset in eligible:
            for change in changeset.files:
                module = module_by_path.get(change.path)
                if (change.kind != FileChangeKind.ADDED or module is None
                        or module.is_package or not change.path.endswith(".py")
                        or not change.parent_package_evidence_ids):
                    continue
                parent_path = PurePosixPath(change.path).parent / "__init__.py"
                package = module_by_path.get(str(parent_path))
                if (package is None or not package.is_package
                        or module.module.rpartition(".")[0] != package.module):
                    continue
                package_ids = set(change.parent_package_evidence_ids)
                for evidence_id in package_ids:
                    evidence = git_evidence.get(evidence_id)
                    if (evidence is None or evidence.kind != EvidenceType.GIT
                            or evidence.repository_id != history.repository_id
                            or evidence.provenance != "git_parent_package"
                            or evidence.location.commit != changeset.commit_id
                            or evidence.location.path != str(parent_path)
                            or evidence_id not in changeset.evidence_ids):
                        raise ValueError("Invalid historical parent package Evidence")
                for evidence_id in package.evidence_ids:
                    evidence = static_evidence.get(evidence_id)
                    if (evidence is None or evidence.repository_id != history.repository_id
                            or evidence.location.path != package.path):
                        raise ValueError("Invalid current parent package Evidence")
                if not any(static_evidence[key].kind == EvidenceType.SOURCE
                           for key in package.evidence_ids):
                    raise ValueError("Missing current parent package SOURCE Evidence")
                signature = f"ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER({package.module})"
                ids = set(change.evidence_ids) | package_ids
                ids.update(log_ids_by_commit[changeset.commit_id])
                ids.update(module.evidence_ids)
                ids.update(package.evidence_ids)
                grouped.setdefault(signature, []).append(StructuralRoleEvent(
                    changeset.commit_id, change.path, tuple(sorted(ids)),
                ))
        patterns = []
        for signature, events in sorted(grouped.items()):
            # A path added again after deletion cannot supply independent new-path support.
            commits = {event.commit_id for event in events}
            paths = {event.path for event in events}
            threshold = max(3, self.min_support)
            if len(commits) < threshold or len(paths) < threshold:
                continue
            # Require a distinct path for every counted supporting commit (matching).
            owner: dict[str, str] = {}
            paths_by_commit = {
                commit: sorted(event.path for event in events if event.commit_id == commit)
                for commit in commits
            }

            def assign(commit: str, visited: set[str], available, assigned) -> bool:
                for path in available[commit]:
                    if path in visited:
                        continue
                    visited.add(path)
                    if path not in assigned or assign(assigned[path], visited, available, assigned):
                        assigned[path] = commit
                        return True
                return False

            supported = {
                commit for commit in sorted(commits)
                if assign(commit, set(), paths_by_commit, owner)
            }
            if len(supported) < threshold:
                continue
            selected = tuple(sorted(
                (event for event in events if event.commit_id in supported),
                key=lambda event: (event.commit_id, event.path),
            ))
            kind = ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
            identity = "\0".join((history.repository_id, kind.value, signature))
            patterns.append(ChangePattern(
                hashlib.sha256(identity.encode()).hexdigest()[:24], history.repository_id,
                kind, (ChangePatternItem(signature),), tuple(sorted(supported)),
                tuple(sorted({key for event in selected for key in event.evidence_ids})),
                Support(len(supported), len(eligible)), len(supported) / len(eligible),
                (len(supported),), None, selected,
            ))
        return patterns

    @staticmethod
    def _validate_evidence(
        changeset: ChangeSet,
        git_evidence: dict[str, Evidence],
        static_evidence: dict[str, Evidence],
        module_by_path: dict[str, PythonModuleAnalysis],
        repository_id: str,
    ) -> set[str]:
        log_ids = {
            evidence_id for evidence_id in changeset.evidence_ids
            if (evidence := git_evidence.get(evidence_id)) is not None
            and evidence.kind == EvidenceType.GIT
            and evidence.provenance == "git_log"
            and evidence.location.commit == changeset.commit_id
            and evidence.repository_id == repository_id
        }
        if not log_ids:
            raise ValueError(f"Missing valid git_log Evidence for commit {changeset.commit_id}")
        changeset_ids = set(changeset.evidence_ids)
        for change in changeset.files:
            if not set(change.evidence_ids) <= changeset_ids:
                raise ValueError(
                    f"FileChange Evidence not in ChangeSet for {changeset.commit_id}:{change.path}"
                )
            provenances = set()
            for evidence_id in change.evidence_ids:
                evidence = git_evidence.get(evidence_id)
                if (evidence is None or evidence.kind != EvidenceType.GIT
                        or evidence.provenance not in {"git_diff", "git_numstat", "git_patch"}
                        or evidence.repository_id != repository_id
                        or evidence.location.commit != changeset.commit_id
                        or evidence.location.path != change.path):
                    raise ValueError(
                        f"Invalid FileChange Evidence for {changeset.commit_id}:{change.path}"
                    )
                provenances.add(evidence.provenance)
                if evidence.provenance == "git_patch":
                    from skillforge.repository.bounded_git_patch_evidence import (
                        validate_git_patch_evidence,
                    )

                    validate_git_patch_evidence(evidence, changeset, change)
            if not {"git_diff", "git_numstat"} <= provenances:
                raise ValueError(
                    f"Missing git_diff or git_numstat Evidence for "
                    f"{changeset.commit_id}:{change.path}"
                )
            module = module_by_path.get(change.path)
            if module is None:
                continue
            if not module.evidence_ids:
                raise ValueError(f"Missing Static module Evidence for {module.path}")
            has_source = False
            for evidence_id in module.evidence_ids:
                evidence = static_evidence.get(evidence_id)
                if (evidence is None or evidence.repository_id != repository_id
                        or evidence.location.path != module.path):
                    raise ValueError(f"Invalid Static module Evidence for {module.path}")
                has_source |= evidence.kind == EvidenceType.SOURCE
            if not has_source:
                raise ValueError(f"Missing Static module SOURCE Evidence for {module.path}")
        return log_ids
