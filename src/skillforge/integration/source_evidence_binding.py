"""Lossless, bounded conversion from source line ranges to exact quotations."""

from __future__ import annotations

MAX_RULE_FIELD_CHARACTERS = 400
MAX_QUOTE_CHARACTERS = 500
MAX_EVIDENCE_ITEMS = 16
MAX_RULES = 8


class EvidenceBindingError(ValueError):
    """A source reference cannot be bound without losing evidence or exceeding caps."""


def bind_evidenced_entrypoints(bound: dict, sources: tuple[dict, ...]) -> dict:
    """Derive callable identities from cited source units, never from model guesses."""
    by_id = {source["source_id"]: source for source in sources}
    rules = []
    for rule in bound["rules"]:
        if "entrypoints" in rule:
            raise EvidenceBindingError("evidence-only protocol cannot supply entrypoints")
        ids = list(dict.fromkeys(item["source_id"] for item in rule["evidence"]))
        if not 1 <= len(ids) <= 12:
            raise EvidenceBindingError("evidenced entrypoint count exceeds bounds")
        entrypoints = []
        for sid in ids:
            symbol = by_id.get(sid, {}).get("symbol")
            if not isinstance(symbol, str) or not 1 <= len(symbol) <= 160:
                raise EvidenceBindingError("cited source has no bounded entrypoint identity")
            entrypoints.append({"source_id": sid, "symbol": symbol})
        rules.append({**rule, "entrypoints": entrypoints})
    return {"rules": rules}


def _range_quote(item: object, sources: dict[str, dict]) -> tuple[str, str]:
    if not isinstance(item, dict) or set(item) != {"source_id", "start_line", "end_line"}:
        raise EvidenceBindingError("line-reference item has invalid shape")
    sid, start, end = item["source_id"], item["start_line"], item["end_line"]
    if (
        not isinstance(sid, str) or sid not in sources
        or type(start) is not int or type(end) is not int
        or start < 1 or end < start
    ):
        raise EvidenceBindingError("line-reference range is invalid")
    lines = sources[sid]["text"].splitlines()
    if end > len(lines):
        raise EvidenceBindingError("line-reference range exceeds supplied source")
    quote = "\n".join(lines[start - 1:end])
    if not quote:
        raise EvidenceBindingError("line-reference quote is empty")
    return sid, quote


def bind_source_lines(raw: object, sources: tuple[dict, ...]) -> dict:
    """Preserve every selected character, or reject the complete response.

    A line range is a transport reference, not an individual quote. Splitting an
    expanded range does not relax the final 500-character/16-item evidence caps.
    Original model JSON is never mutated, including when a later range fails.
    """
    if (
        not isinstance(raw, dict) or set(raw) != {"rules"}
        or not isinstance(raw["rules"], list) or len(raw["rules"]) > MAX_RULES
    ):
        raise EvidenceBindingError("line-reference response has invalid shape")
    by_id = {source["source_id"]: source for source in sources}
    bound = []
    for rule in raw["rules"]:
        if (
            not isinstance(rule, dict) or not isinstance(rule.get("evidence"), list)
            or not 1 <= len(rule["evidence"]) <= MAX_EVIDENCE_ITEMS
        ):
            raise EvidenceBindingError("line-reference evidence has invalid shape")
        evidence = []
        for item in rule["evidence"]:
            sid, quote = _range_quote(item, by_id)
            for offset in range(0, len(quote), MAX_QUOTE_CHARACTERS):
                if len(evidence) >= MAX_EVIDENCE_ITEMS:
                    raise EvidenceBindingError("line-reference quotes exceed evidence item limit")
                evidence.append({"source_id": sid,
                                 "quote": quote[offset:offset + MAX_QUOTE_CHARACTERS]})
        bound.append({**rule, "evidence": evidence})
    return {"rules": bound}
