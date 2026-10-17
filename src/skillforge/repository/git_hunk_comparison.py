"""Conservative, ancestry-aware pairing of complete Git hunk observations.

Function anchors and ancestry only help retrieve related observations. They do
not establish that one change semantically replaces another.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HunkCandidate:
    evidence: Any
    serialized_size: int
    priority: int
    hunk_order: int
    anchor: str | None


_HEADER = re.compile(r"^@@ .*? @@(?P<context>.*)$")
_DEF_LINE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
_HUNK_RANGE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def function_anchor(hunk_text: str) -> str | None:
    """Return a unique explicit Python function name, or no usable anchor."""
    lines = hunk_text.splitlines()
    if not lines:
        return None
    header = _HEADER.match(lines[0])
    contexts = [header.group("context")] if header else []
    for line in lines[1:]:
        if line.startswith((" ", "+", "-")):
            contexts.append(line[1:])
    # Anchoring at the declaration line avoids matching string/comment contents.
    names = {match.group(1) for text in contexts
             if (match := _DEF_LINE.match(text)) is not None}
    return f"def:{next(iter(names))}" if len(names) == 1 else None


def is_ancestor(
    ancestor_id: str,
    descendant_id: str,
    changesets: Mapping[str, Any],
) -> bool:
    """Use only recorded parent links; missing graph edges mean unknown."""
    if ancestor_id == descendant_id:
        return False
    seen: set[str] = set()
    pending = list(getattr(changesets.get(descendant_id), "parent_ids", ()))
    while pending:
        commit_id = pending.pop()
        if commit_id == ancestor_id:
            return True
        if commit_id in seen:
            continue
        seen.add(commit_id)
        item = changesets.get(commit_id)
        if item is not None:
            pending.extend(item.parent_ids)
    return False


def descendants_first_order(
    requests: Sequence[tuple[str, str]],
    changesets: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Stable topological request order, preferring descendants on same paths."""
    edges: dict[int, set[int]] = {index: set() for index in range(len(requests))}
    indegree = [0] * len(requests)
    for older_index, (older_commit, older_path) in enumerate(requests):
        for newer_index, (newer_commit, newer_path) in enumerate(requests):
            if (older_path == newer_path and older_index != newer_index
                    and is_ancestor(older_commit, newer_commit, changesets)
                    and older_index not in edges[newer_index]):
                edges[newer_index].add(older_index)
                indegree[older_index] += 1
    ready = [index for index, count in enumerate(indegree) if count == 0]
    ordered_indices: list[int] = []
    while ready:
        current = min(ready)  # request order breaks only incomparable ties
        ready.remove(current)
        ordered_indices.append(current)
        for target in sorted(edges[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ordered_indices) != len(requests):
        return list(requests)
    return [requests[index] for index in ordered_indices]


def compact_context_requests(
    requests: Sequence[tuple[str, str]],
    changesets: Mapping[str, Any],
    candidates: Mapping[tuple[str, str], Sequence[HunkCandidate]],
    max_reference_bytes: int,
) -> list[tuple[str, str]]:
    """Find unique later same-anchor hunks worth one compact-context retry."""
    chosen: dict[tuple[str, str], HunkCandidate] = {}
    ordered = descendants_first_order(requests, changesets)
    for key in ordered:
        fitting = [item for item in candidates.get(key, ())
                   if item.serialized_size <= max_reference_bytes]
        if fitting:
            chosen[key] = min(fitting, key=lambda item: (item.priority, item.hunk_order))

    retry: set[tuple[str, str]] = set()
    for older_key, old in chosen.items():
        if old.anchor is None:
            continue
        children = [key for key in ordered if key[1] == older_key[1]
                    and tuple(getattr(changesets.get(key[0]), "parent_ids", ()))
                    == (older_key[0],)]
        related = {
            key: [candidate for candidate in candidates.get(key, ())
                  if _related_candidates(older_key, old, key, candidate, changesets)]
            for key in children
        }
        matching_keys = [key for key, items in related.items() if items]
        if len(matching_keys) != 1:
            continue
        latest = matching_keys[0]
        aligned = related[latest]
        if aligned and all(
            candidate.serialized_size > max_reference_bytes for candidate in aligned
        ):
            retry.add(latest)
    return [key for key in ordered if key in retry]


def select_compared_hunks(
    requests: Sequence[tuple[str, str]],
    changesets: Mapping[str, Any],
    candidates: Mapping[tuple[str, str], Sequence[HunkCandidate]],
    unavailable: Mapping[tuple[str, str], str],
    max_reference_bytes: int,
) -> tuple[dict[tuple[str, str], HunkCandidate], set[tuple[str, str]]]:
    """Choose legacy-best hunks, aligning direct-parent anchors and line ranges.

    The returned blocked requests are incomplete comparisons, not semantic
    supersession claims. Incomparable branches and unanchored hunks fall back.
    """
    chosen: dict[tuple[str, str], HunkCandidate] = {}
    blocked: set[tuple[str, str]] = set()
    ordered = descendants_first_order(requests, changesets)
    for key in ordered:
        if key in unavailable and unavailable[key] != "compact_context_read_budget":
            continue
        fitting = [item for item in candidates.get(key, ())
                   if item.serialized_size <= max_reference_bytes]
        if fitting:
            chosen[key] = min(fitting, key=lambda item: (item.priority, item.hunk_order))

    # Compare only direct-parent hunks whose old/new line ranges overlap.
    for older_key in ordered:
        old = chosen.get(older_key)
        if old is None or old.anchor is None:
            continue
        older_commit, path = older_key
        children = [key for key in ordered if key[1] == path
                    and tuple(getattr(changesets.get(key[0]), "parent_ids", ()))
                    == (older_commit,)]
        if any(key in unavailable or key in blocked for key in children):
            chosen.pop(older_key, None)
            blocked.add(older_key)
            continue
        related = {
            key: [candidate for candidate in candidates.get(key, ())
                  if _related_candidates(older_key, old, key, candidate, changesets)]
            for key in children
        }
        matching_keys = [key for key, items in related.items() if items]
        if len(matching_keys) != 1:
            continue
        latest = matching_keys[0]
        aligned = related[latest]
        fitting_aligned = [candidate for candidate in aligned
                           if candidate.serialized_size <= max_reference_bytes]
        if fitting_aligned:
            chosen[latest] = min(
                fitting_aligned, key=lambda item: (item.priority, item.hunk_order),
            )
        else:
            # Remove only the older observation for this same anchor. If a
            # separate fitting hunk exists in that request, retain the fallback.
            alternatives = [candidate for candidate in candidates.get(older_key, ())
                            if candidate.serialized_size <= max_reference_bytes
                            and not any(_related_candidates(
                                older_key, candidate, latest, later, changesets,
                            ) for later in aligned)]
            if alternatives:
                chosen[older_key] = min(
                    alternatives, key=lambda item: (item.priority, item.hunk_order),
                )
            else:
                chosen.pop(older_key, None)
                blocked.add(older_key)
    return chosen, blocked


def _related_candidates(
    older_key: tuple[str, str],
    older: HunkCandidate,
    newer_key: tuple[str, str],
    newer: HunkCandidate,
    changesets: Mapping[str, Any],
) -> bool:
    """Pair only same-anchor direct-parent hunks touching the same line range."""
    if (older_key[1] != newer_key[1] or older.anchor is None
            or older.anchor != newer.anchor):
        return False
    newer_set = changesets.get(newer_key[0])
    if tuple(getattr(newer_set, "parent_ids", ())) != (older_key[0],):
        return False
    old_range, new_range = _candidate_ranges(older), _candidate_ranges(newer)
    if old_range is None or new_range is None:
        return False
    old_after, new_before = old_range[1], new_range[0]
    return max(old_after[0], new_before[0]) < min(old_after[1], new_before[1])


def _candidate_ranges(
    candidate: HunkCandidate,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    description = getattr(candidate.evidence, "description", None)
    if not isinstance(description, str) or "\n" not in description:
        return None
    hunk_header = description.split("\n", 1)[1].splitlines()[0]
    match = _HUNK_RANGE.match(hunk_header)
    if match is None:
        return None
    old_start = int(match.group("old_start"))
    new_start = int(match.group("new_start"))
    old_count = int(match.group("old_count") or 1)
    new_count = int(match.group("new_count") or 1)
    return ((old_start, old_start + old_count), (new_start, new_start + new_count))
