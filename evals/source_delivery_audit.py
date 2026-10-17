"""Pure consistency audit for current-source rule delivery artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

_CANDIDATE = re.compile(r"^## Candidate ([1-9][0-9]*)\s*$", re.MULTILINE)


class SourceDeliveryAuditError(ValueError):
    """A source-rule response, delivery index, document, or digest disagrees."""


def _canonical(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _unique_indexes(values: object, raw_count: int, label: str) -> list[int]:
    if not isinstance(values, list) or any(
        type(value) is not int or not 0 <= value < raw_count for value in values
    ):
        raise SourceDeliveryAuditError(f"{label} indexes are invalid")
    if len(set(values)) != len(values):
        raise SourceDeliveryAuditError(f"{label} indexes contain duplicates")
    return values


def audit_source_delivery(files: Mapping[str, str]) -> dict[str, object]:
    """Verify raw-rule accounting, rendered candidates, hashes, and document cap.

    The result contains only counts, byte measurements, hashes, and the requested
    delivery-completeness boolean; it never copies rule text into audit metadata.
    """
    try:
        manifest = json.loads(files[".skillforge.json"])
        body = files["SKILL.md"]
        evidence = files["references/evidence.md"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SourceDeliveryAuditError("delivery files or manifest are missing/invalid") from exc
    if not isinstance(body, str) or not isinstance(evidence, str) or not isinstance(manifest, dict):
        raise SourceDeliveryAuditError("delivery document types are invalid")

    response = manifest.get("response")
    raw_rules = response.get("rules") if isinstance(response, dict) else None
    if not isinstance(raw_rules, list):
        raise SourceDeliveryAuditError("raw response rules are missing")
    raw_count = len(raw_rules)
    delivery = manifest.get("delivery")
    if not isinstance(delivery, dict):
        raise SourceDeliveryAuditError("delivery metadata is missing")
    retained = _unique_indexes(delivery.get("retained_rule_indexes"), raw_count, "retained")
    dropped_rows = delivery.get("dropped_rules")
    if not isinstance(dropped_rows, list) or any(
        not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], str)
        for row in dropped_rows
    ):
        raise SourceDeliveryAuditError("dropped-rule metadata is invalid")
    dropped = _unique_indexes([row[0] for row in dropped_rows], raw_count, "dropped")
    if set(retained).intersection(dropped) or set(retained).union(dropped) != set(range(raw_count)):
        raise SourceDeliveryAuditError("retained and dropped indexes do not partition raw rules")

    candidate_numbers = [int(value) for value in _CANDIDATE.findall(_canonical(body))]
    if candidate_numbers != list(range(1, len(retained) + 1)):
        raise SourceDeliveryAuditError("Candidate headings do not match retained delivery count")

    declared_hashes = manifest.get("file_hashes")
    if not isinstance(declared_hashes, dict):
        raise SourceDeliveryAuditError("file hashes are missing")
    hashes = {
        "SKILL.md": hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(),
        "references/evidence.md": hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest(),
    }
    if any(declared_hashes.get(path) != digest for path, digest in hashes.items()):
        raise SourceDeliveryAuditError("canonical-LF document hash mismatch")

    limit = delivery.get("max_document_bytes")
    if type(limit) is not int or limit < 0:
        raise SourceDeliveryAuditError("document byte limit is invalid")
    raw_bytes = len(body.encode("utf-8")) + len(evidence.encode("utf-8"))
    escaped_bytes = len(json.dumps(
        [body, evidence], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8"))
    if max(raw_bytes, escaped_bytes) > limit:
        raise SourceDeliveryAuditError("delivered documents exceed declared byte limit")
    return {
        "counts": {"raw_rules": raw_count, "delivered_rules": len(retained)},
        "all_raw_rules_delivered": len(retained) == raw_count,
        "bytes": {"utf8_documents": raw_bytes, "json_escaped_documents": escaped_bytes,
                  "max_document_bytes": limit},
        "hashes": hashes,
    }
