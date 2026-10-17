"""Deterministic, budget-bounded distribution of complete source units."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from itertools import zip_longest

Span = tuple[str, str]
SpanUnit = tuple[int, Sequence[Span]]
FileUnits = tuple[str, str, Sequence[Sequence[SpanUnit]]]
SelectedSpan = tuple[str, str, str, str]


def _spread_indices(count: int) -> list[int]:
    """Visit the beginning, middle and end before filling intervening positions."""
    if count <= 0:
        return []
    order: list[int] = []
    for index in (0, count // 2, count - 1):
        if index not in order:
            order.append(index)
    intervals = [(1, count // 2), (count // 2 + 1, count - 1)]
    while intervals:
        following = []
        for start, stop in intervals:
            if start >= stop:
                continue
            middle = (start + stop) // 2
            if middle not in order:
                order.append(middle)
            following.extend(((start, middle), (middle + 1, stop)))
        intervals = following
    return order


def _ordered_units(tier: Sequence[SpanUnit]) -> list[SpanUnit]:
    ordered: list[SpanUnit] = []
    def quality(unit: SpanUnit) -> tuple[Fraction, int]:
        benefit, spans = unit
        cost = sum(len(text.encode("utf-8")) for _, text in spans)
        return Fraction(benefit, max(1, cost)), benefit

    groups: dict[tuple[Fraction, int], list[SpanUnit]] = {}
    for unit in tier:
        groups.setdefault(quality(unit), []).append(unit)
    for priority in sorted(groups, reverse=True):
        group = groups[priority]
        ordered.extend(group[index] for index in _spread_indices(len(group)))
    return ordered


def _selection_rounds(
    files: Sequence[FileUnits], relevance: Mapping[tuple[str, str], int],
) -> Iterator[tuple[str, str, Sequence[Span]]]:
    """Related units precede unranked units; equal scores interleave files."""
    tier_count = max((len(tiers) for _, _, tiers in files), default=0)
    for tier_index in range(tier_count):
        tiers = [tiers[tier_index] if tier_index < len(tiers) else []
                 for _, _, tiers in files]
        scored = [[(relevance.get((path, spans[0][0]), 0), (benefit, spans))
                   for benefit, spans in tier if spans]
                  for (path, _, _), tier in zip(files, tiers, strict=True)]
        levels = sorted({score for units in scored for score, _ in units}, reverse=True)
        for level in levels:
            orders = [_ordered_units([unit for score, unit in units if score == level])
                      for units in scored]
            for round_units in zip_longest(*orders):
                for (path, sha, _), unit in zip(files, round_units, strict=True):
                    if unit is not None:
                        yield path, sha, unit[1]


def select_distributed_spans(
    files: Sequence[FileUnits],
    *,
    max_spans: int,
    max_spans_per_file: int = 32,
    raw_byte_budget: int,
    serialized_fits: Callable[[Sequence[SelectedSpan]], bool],
    relevance: Mapping[tuple[str, str], int] | None = None,
    allowed_symbols: frozenset[tuple[str, str]] | None = None,
    dependencies: Mapping[tuple[str, str], Sequence[str] | None] | None = None,
) -> tuple[list[SelectedSpan], list[tuple[str, str, str]]]:
    """Select complete units, interleaved by file and spread by source position.

    Each unit contains all numbered spans belonging to one safe function/method (or
    one lower-priority auxiliary segment). Oversized units are skipped atomically.
    ``serialized_fits`` is a pure caller-supplied check for its existing wire formats.
    ``allowed_symbols`` narrows eligible units; None preserves unrestricted selection.
    Skip records contain only path, symbol and a bounded reason code.
    """
    if max_spans < 0 or max_spans_per_file < 0 or raw_byte_budget < 0:
        raise ValueError("selection limits must be nonnegative")
    selected: list[SelectedSpan] = []
    skipped: list[tuple[str, str, str]] = []
    units_by_key = {(path, spans[0][0]): spans for path, _, tiers in files
                    for tier in tiers for _, spans in tier if spans}
    dependency_keys = {(path, name) for (path, _), names in (dependencies or {}).items()
                       for name in names or ()}
    selected_keys: set[tuple[str, str]] = set()
    for path, sha, spans in _selection_rounds(files, relevance or {}):
        symbol = spans[0][0]
        if (path, symbol) in selected_keys and (path, symbol) in dependency_keys:
            continue
        if allowed_symbols is not None and (path, symbol) not in allowed_symbols:
            continue
        required = (dependencies or {}).get((path, symbol), ())
        if required is None or any((path, name) not in units_by_key for name in required):
            skipped.append((path, symbol, "unit_dependency_unavailable"))
            continue
        additions = list(spans)
        for name in required:
            if (path, name) not in selected_keys and name != symbol:
                additions.extend(units_by_key[(path, name)])
        candidate = [*selected, *((path, sha, name, text) for name, text in additions)]
        raw_bytes = sum(len(row[3].encode("utf-8")) for row in candidate)
        if len(candidate) > max_spans:
            skipped.append((path, symbol, "unit_exceeds_span_limit"))
        elif sum(row[0] == path for row in candidate) > max_spans_per_file:
            skipped.append((path, symbol, "unit_exceeds_file_span_limit"))
        elif raw_bytes > raw_byte_budget:
            skipped.append((path, symbol, "unit_exceeds_raw_budget"))
        elif not serialized_fits(candidate):
            skipped.append((path, symbol, "unit_exceeds_serialized_budget"))
        else:
            selected = candidate
            selected_keys.update((path, name) for name, _ in additions)
    return selected, skipped
