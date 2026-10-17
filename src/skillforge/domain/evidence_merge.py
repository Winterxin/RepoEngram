"""Deterministic Evidence union with exact identity conflict checks."""

from collections.abc import Iterable

from skillforge.domain.evidence import Evidence


def merge_evidence(results: Iterable) -> tuple[Evidence, ...]:
    merged: dict[str, Evidence] = {}
    for result in results:
        for item in result.evidence:
            prior = merged.get(item.evidence_id)
            if prior is not None and prior != item:
                raise ValueError(f"conflicting semantic Evidence ID: {item.evidence_id}")
            merged[item.evidence_id] = item
    return tuple(merged[key] for key in sorted(merged))
