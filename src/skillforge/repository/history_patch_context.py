"""Share bounded historical observations across the selected semantic channels."""

from dataclasses import replace

from skillforge.domain.change import ChangePatternKind, FileChangeKind
from skillforge.repository.bounded_candidate_selection import scoped_candidates
from skillforge.repository.bounded_git_patch_evidence import (
    add_bounded_git_patch_evidence,
    validate_git_patch_evidence,
)
from skillforge.repository.semantic_antipattern_evaluation import (
    select_semantic_antipattern_candidates,
)
from skillforge.repository.semantic_evaluation import select_semantic_candidates


def strip_git_patch_evidence(history):
    """Rebuild a fixed observation window rather than accumulate prior patch context."""
    indexed = {item.evidence_id: item for item in history.evidence}
    if len(indexed) != len(history.evidence):
        raise ValueError("Duplicate history Evidence")
    patch_ids = {key for key, item in indexed.items() if item.provenance == "git_patch"}
    if not patch_ids:
        return history
    bound = set()
    for commit in history.changesets:
        for change in commit.files:
            for key in patch_ids.intersection(change.evidence_ids):
                if key not in commit.evidence_ids or not validate_git_patch_evidence(
                    indexed[key], commit, change,
                ):
                    raise ValueError("Invalid saved historical patch Evidence")
                bound.add(key)
    if bound != patch_ids:
        raise ValueError("Unassociated historical patch Evidence")
    return replace(
        history,
        changesets=tuple(replace(
            commit,
            files=tuple(replace(change, evidence_ids=tuple(
                key for key in change.evidence_ids if key not in patch_ids
            )) for change in commit.files),
            evidence_ids=tuple(key for key in commit.evidence_ids if key not in patch_ids),
        ) for commit in history.changesets),
        evidence=tuple(item for item in history.evidence if item.evidence_id not in patch_ids),
    )


def enrich_selected_history(root, history, facts, workflow_candidates, negative_candidates,
                            *, workflow_limit=5, anti_pattern_limit=5, current_module=None):
    """Keep quotas unchanged and share finite patch examples across existing candidates."""
    workflow = select_semantic_candidates(
        scoped_candidates(workflow_candidates, facts, current_module), limit=workflow_limit)
    negative = select_semantic_antipattern_candidates(
        scoped_candidates(negative_candidates, facts, current_module), limit=anti_pattern_limit)
    paths_by_module = {module.module: module.path for module in facts.modules}
    groups = []

    def pairs(commits, paths):
        eligible = tuple(
            (commit.commit_id, change.path)
            for commit in history.changesets if commit.commit_id in commits
            and not commit.is_merge and len(commit.parent_ids) == 1
            for change in sorted(commit.files, key=lambda item: (
                "/tests/" in item.path or item.path.startswith("tests/"), item.path,
            ))
            if change.kind in {FileChangeKind.MODIFIED, FileChangeKind.RENAMED}
            and (change.path in paths or change.previous_path in paths)
        )
        return _distinct_commit_examples_first(eligible)

    for index in range(max(len(workflow), len(negative))):
        if index < len(workflow):
            candidate = workflow[index]
            if candidate.kind in {
                ChangePatternKind.FILE_CO_CHANGE, ChangePatternKind.MODULE_CO_CHANGE,
            }:
                paths = {member.value for member in candidate.members}
                paths.update(paths_by_module[name] for name in candidate.current_modules
                             if name in paths_by_module)
                groups.append(pairs(set(candidate.supporting_commits), paths))
        if index < len(negative):
            candidate = negative[index]
            groups.append(pairs(set(candidate.commit_ids), set(candidate.affected_paths)))
    requests = []
    # Reserve the leading ranked candidate's comparison example; then keep the
    # existing breadth-first channel interleaving for every remaining group.
    # Do not reserve two examples for all groups at the expense of coverage.
    leading = next((group for group in groups if group), ())
    requests.extend(leading[:2])
    for position in range(2):
        for group in groups:
            if position < len(group) and group[position] not in requests:
                requests.append(group[position])
    return add_bounded_git_patch_evidence(root, history, tuple(requests))


def _distinct_commit_examples_first(pairs):
    """Use the existing two slots for separate revisions when available.

    Each revision keeps its first eligible file (implementation before tests).
    Remaining files retain their order as fallback for a single revision. This
    changes retrieval only: separate revisions do not prove a shared policy.
    """
    representatives = []
    remaining = []
    seen = set()
    for pair in pairs:
        commit_id, _ = pair
        if commit_id in seen:
            remaining.append(pair)
        else:
            seen.add(commit_id)
            representatives.append(pair)
    return tuple((*representatives, *remaining))
