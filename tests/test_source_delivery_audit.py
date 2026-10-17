"""Audit source-rule delivery accounting without exposing candidate text."""

import hashlib
import json

import pytest

from evals.source_delivery_audit import SourceDeliveryAuditError, audit_source_delivery


def _canonical(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _files():
    body = "# Candidate summary 😀\n\n" + "\n".join(
        f"## Candidate {index}\n" for index in range(1, 8)
    )
    evidence = "# Evidence\n- E1: \"exact quote\"\n"
    marker = {
        "response": {"rules": [{"candidate": index} for index in range(8)]},
        "delivery": {
            "max_document_bytes": 8192,
            "retained_rule_indexes": list(range(7)),
            "dropped_rules": [[7, "rule_exceeds_byte_budget"]],
        },
        "file_hashes": {
            "SKILL.md": hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(),
            "references/evidence.md": hashlib.sha256(
                _canonical(evidence).encode("utf-8")
            ).hexdigest(),
        },
    }
    return {
        "SKILL.md": body.replace("\n", "\r\n"),
        "references/evidence.md": evidence,
        ".skillforge.json": json.dumps(marker),
    }


def test_audit_reports_raw_vs_delivered_counts_and_canonical_hashes_only():
    result = audit_source_delivery(_files())
    assert result["counts"] == {"raw_rules": 8, "delivered_rules": 7}
    assert result["all_raw_rules_delivered"] is False
    assert result["hashes"]["SKILL.md"] == hashlib.sha256(
        _canonical(_files()["SKILL.md"]).encode("utf-8")
    ).hexdigest()
    assert result["bytes"]["max_document_bytes"] == 8192
    assert "Candidate 1" not in json.dumps(result)


@pytest.mark.parametrize("corruption", ["candidate_count", "hash", "budget", "indexes"])
def test_audit_rejects_inconsistent_delivery_metadata(corruption):
    files = _files()
    marker = json.loads(files[".skillforge.json"])
    if corruption == "candidate_count":
        files["SKILL.md"] = files["SKILL.md"].replace("## Candidate 7", "## Draft 7")
    elif corruption == "hash":
        marker["file_hashes"]["SKILL.md"] = "invalid"
    elif corruption == "budget":
        marker["delivery"]["max_document_bytes"] = 1
    else:
        marker["delivery"]["retained_rule_indexes"] = [*range(6), 8]
    files[".skillforge.json"] = json.dumps(marker)
    with pytest.raises(SourceDeliveryAuditError):
        audit_source_delivery(files)


def test_audit_rejects_nonpartitioning_rule_indexes():
    files = _files()
    marker = json.loads(files[".skillforge.json"])
    marker["delivery"]["dropped_rules"] = [[6, "skipped"], [7, "skipped"]]
    files[".skillforge.json"] = json.dumps(marker)
    with pytest.raises(SourceDeliveryAuditError, match="partition"):
        audit_source_delivery(files)
