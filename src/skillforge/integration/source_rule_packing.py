"""Pure, bounded subset selection for already validated source rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

Binding = tuple[str, str]


def prioritize_rule_baseline(
    eligible: Sequence[int], baseline: tuple[int, ...],
    bindings: Mapping[int, frozenset[Binding]],
    measure: Callable[[tuple[int, ...]], tuple[int, int]], *,
    max_bytes: int, preferred: frozenset[Binding],
) -> tuple[int, ...]:
    """Prefer evidenced explicit entrypoints only when their coverage improves."""
    if not preferred:
        return baseline
    order = sorted(eligible, key=lambda i: (-len(bindings[i] & preferred), i))
    candidate: tuple[int, ...] = ()
    for index in order:
        if max(measure((*candidate, index))) <= max_bytes:
            candidate = (*candidate, index)

    def coverage(indexes: tuple[int, ...]) -> int:
        return len(frozenset().union(*(bindings[i] for i in indexes)) & preferred)

    return candidate if coverage(candidate) > coverage(baseline) else baseline


def _primary_rules(
    baseline: Sequence[int], bindings: Mapping[int, frozenset[Binding]],
) -> frozenset[int]:
    covered: set[Binding] = set()
    primary = set()
    for index in baseline:
        if bindings[index] - covered:
            primary.add(index)
            covered.update(bindings[index])
    return frozenset(primary)


def choose_rule_subset(
    eligible: Sequence[int],
    baseline: tuple[int, ...],
    bindings: Mapping[int, frozenset[Binding]],
    measure: Callable[[tuple[int, ...]], tuple[int, int]],
    *,
    max_bytes: int,
) -> tuple[int, ...]:
    """Preserve primary baseline rules, then maximize distinct source bindings.

    At most eight validated candidates mean at most 256 subsets. Coverage counts
    (path, symbol), not chunk IDs. It is a structural proxy, not semantic utility.
    """
    if len(eligible) > 8:
        raise ValueError("rule packing supports at most eight candidates")
    primary = _primary_rules(baseline, bindings)
    baseline_set = frozenset(baseline)

    def score(indexes: tuple[int, ...], sizes: tuple[int, int]) -> tuple:
        covered = set().union(*(bindings[index] for index in indexes))
        return (
            len(covered), len({path for path, _ in covered}), len(indexes),
            len(baseline_set.intersection(indexes)), -max(sizes),
            tuple(-index for index in indexes),
        )

    best = baseline
    best_score = score(baseline, measure(baseline))
    baseline_coverage = best_score[0]
    for count in range(len(primary), len(eligible) + 1):
        for indexes in combinations(eligible, count):
            if not primary.issubset(indexes):
                continue
            sizes = measure(indexes)
            if max(sizes) > max_bytes:
                continue
            candidate_score = score(indexes, sizes)
            if candidate_score > best_score:
                best, best_score = indexes, candidate_score
    return best if best_score[0] > baseline_coverage else baseline
