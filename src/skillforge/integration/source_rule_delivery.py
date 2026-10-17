"""Lossless compact rendering and whole-rule budgeting for source candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from skillforge.domain.project_dna import ProjectScope
from skillforge.integration.source_evidence_binding import MAX_RULE_FIELD_CHARACTERS, MAX_RULES
from skillforge.integration.source_rule_packing import choose_rule_subset, prioritize_rule_baseline

Rule = Mapping[str, Any]
Source = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SourceRuleDelivery:
    body: str
    evidence: str
    retained_rule_indexes: tuple[int, ...]
    dropped_rules: tuple[tuple[int, str], ...]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _candidate_lines(
    rules: Sequence[Rule],
) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
    evidence_ids: dict[tuple[str, str], str] = {}
    evidence_rows: list[tuple[str, str, str]] = []
    source_order: list[str] = []
    candidate_rows: list[str] = []

    for candidate_index, rule in enumerate(rules, 1):
        refs: list[str] = []
        for item in rule["evidence"]:
            source_id = item["source_id"]
            quote = _normalize_newlines(item["quote"])
            key = (source_id, quote)
            evidence_id = evidence_ids.get(key)
            if evidence_id is None:
                evidence_id = f"E{len(evidence_ids) + 1}"
                evidence_ids[key] = evidence_id
                evidence_rows.append((evidence_id, source_id, quote))
            if source_id not in source_order:
                source_order.append(source_id)
            if evidence_id not in refs:
                refs.append(evidence_id)

        entrypoints = [
            {"source_id": item["source_id"], "symbol": item["symbol"]}
            for item in rule["entrypoints"]
        ]
        exceptions = list(rule["exceptions"])
        ordering = list(rule["ordering"])
        fields = [("condition", rule["condition"]), ("action", rule["action"])]
        if exceptions:
            fields.append(("exceptions", exceptions))
        if ordering:
            fields.append(("ordering", ordering))
        fields.extend((
            ("entrypoints", [[item["source_id"], item["symbol"]]
                             for item in entrypoints]),
            ("evidence_refs", refs),
        ))
        candidate_rows.extend((
            f"## Candidate {candidate_index}",
            "",
            *[f"- {name}: {_json(value)}" for name, value in fields],
            "",
        ))
    return candidate_rows, evidence_rows, source_order


def _render_evidence(
    evidence_rows: Sequence[tuple[str, str, str]],
    source_order: Sequence[str],
    sources: Sequence[Source],
) -> str:
    source_paths = {source["source_id"]: source["path"] for source in sources}
    paths = list(dict.fromkeys(source_paths[source_id] for source_id in source_order))
    path_ids = {path: f"P{index}" for index, path in enumerate(paths, 1)}
    lines = [
        "# Exact source evidence", "",
        "Quotes are complete; line endings normalized to LF.", "",
    ]
    lines.extend(f"- {_json(source_id)} -> {path_ids[source_paths[source_id]]}"
                 for source_id in source_order)
    lines.extend(("", "## Paths", ""))
    lines.extend(f"- {path_ids[path]} -> {_json(path)}" for path in paths)
    lines.extend(("", "## Evidence", ""))
    lines.extend(
        f"- {evidence_id} ({_json(source_id)}): {_json(quote)}"
        for evidence_id, source_id, quote in evidence_rows
    )
    return "\n".join(lines).rstrip() + "\n"


def render_source_rules(
    rules: Sequence[Rule], sources: Sequence[Source], scope: ProjectScope,
) -> tuple[str, str]:
    """Render complete rule fields and deduplicated, newline-normalized evidence."""
    candidate_rows, evidence_rows, source_order = _candidate_lines(rules)
    scope_data = {"kind": scope.kind.value, "selectors": list(scope.selectors)}
    body = (
        "---\nname: current-source-rule-candidates\n"
        "description: Advisory behavior candidates from explicitly scoped source.\n"
        "---\n\n# Current source rule candidates\n\n"
        "These are advisory CURRENT_SOURCE_RULE_CANDIDATE observations, not proof, confirmed "
        "history, or normative policy. A requested task may change existing behavior; "
        "applicability and exceptions must be checked against current source.\n\n"
        "Entrypoints are [source_id,symbol]; empty exceptions/ordering are omitted; "
        "evidence_refs preserve candidate mappings.\n\n"
        f"## Explicit scope\n\n{_json(scope_data)}\n\n"
        + "\n".join(candidate_rows)
        + "See [references/evidence.md](references/evidence.md).\n"
    )
    return body, _render_evidence(evidence_rows, source_order, sources)


def _delivery_sizes(body: str, evidence: str) -> tuple[int, int]:
    raw = len(body.encode("utf-8")) + len(evidence.encode("utf-8"))
    escaped = len(json.dumps(
        [body, evidence], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8"))
    return raw, escaped


def _suspected_truncated_summary(rule: Rule) -> bool:
    """Flag only cap-length text ending inside an odd-count backtick span.

    This narrow heuristic can reject a legitimate 400-character unmatched-backtick
    summary; it is a truncation safeguard, not semantic validation.
    """
    for field in ("condition", "action", "exceptions", "ordering"):
        values = rule.get(field, ())
        for value in (values,) if isinstance(values, str) else values:
            if (isinstance(value, str) and len(value) == MAX_RULE_FIELD_CHARACTERS
                    and value.count("`") % 2):
                return True
    return False


def budget_source_rules(
    rules: Sequence[Rule],
    sources: Sequence[Source],
    scope: ProjectScope,
    *,
    max_bytes: int = 8192,
    preferred_bindings: frozenset[tuple[str, str]] = frozenset(),
) -> SourceRuleDelivery:
    """Keep primary rules and broaden entrypoint coverage within both size limits.

    Rules are atomic with all their evidence. Oversized rules are reported by their
    zero-based input index and skipped; fields, quotes, and references are never cut.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    if len(rules) > MAX_RULES:
        raise ValueError("too many source-rule candidates")
    empty_body, empty_evidence = render_source_rules((), sources, scope)
    if max(_delivery_sizes(empty_body, empty_evidence)) > max_bytes:
        raise ValueError("source-rule delivery prefix exceeds byte budget")

    kept: list[Rule] = []
    kept_indexes: list[int] = []
    for index, rule in enumerate(rules):
        if _suspected_truncated_summary(rule):
            continue
        candidate = [*kept, rule]
        body, evidence = render_source_rules(candidate, sources, scope)
        raw, escaped = _delivery_sizes(body, evidence)
        if max(raw, escaped) <= max_bytes:
            kept = candidate
            kept_indexes.append(index)

    paths = {source["source_id"]: source["path"] for source in sources}
    eligible = [i for i, rule in enumerate(rules) if not _suspected_truncated_summary(rule)]
    bindings = {
        i: frozenset((paths[item["source_id"]], item["symbol"])
                     for item in rules[i]["entrypoints"])
        for i in eligible
    }

    def measure(indexes: tuple[int, ...]) -> tuple[int, int]:
        return _delivery_sizes(*render_source_rules([rules[i] for i in indexes], sources, scope))

    baseline = prioritize_rule_baseline(
        eligible, tuple(kept_indexes), bindings, measure,
        max_bytes=max_bytes, preferred=preferred_bindings,
    )
    selected = baseline if len(baseline) == len(eligible) else choose_rule_subset(
        eligible, baseline, bindings, measure, max_bytes=max_bytes,
    )
    dropped = [
        (i, "suspected_truncated_summary" if i not in eligible else "rule_exceeds_byte_budget")
        for i in range(len(rules)) if i not in selected
    ]
    body, evidence = render_source_rules([rules[i] for i in selected], sources, scope)
    return SourceRuleDelivery(body, evidence, selected, tuple(dropped))
