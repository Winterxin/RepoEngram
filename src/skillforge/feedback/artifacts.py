"""Canonical repository-local persistence for Verify feedback."""

from __future__ import annotations

from pathlib import Path

from skillforge.domain.feedback import FeedbackKind, HumanFeedbackLedger
from skillforge.feedback.service import HumanFeedbackService
from skillforge.repository.identity import repository_identity
from skillforge.verify.artifacts import (
    _canonical_json,
    _mapping,
    _read_repo_json,
    _resolve_repo_file,
    _sequence,
    _target_ref,
    _write_json,
    load_verify_artifact,
)

LEDGER_PATH = Path(".skillforge/productization/feedback/ledger.json")
_SCHEMA = "skillforge-human-feedback-ledger-v1"


def source_path(root: Path, supplied: Path | str) -> str:
    root = root.resolve()
    raw = str(supplied)
    if not Path(raw).is_absolute() and ("\\" in raw or ":" in raw.split("/")[0]
            or any(part in {"", ".", ".."} for part in raw.split("/"))):
        raise ValueError("source Verify artifact path must be canonical repository-relative")
    path = _resolve_repo_file(root, supplied)
    relative = path.relative_to(root).as_posix()
    if not relative or relative.startswith("../"):
        raise ValueError("source Verify artifact path must be repository-relative")
    return relative


def _entry(record, source: str) -> dict[str, object]:
    target = record.target
    return {
        "source_verify_artifact": source,
        "source_snapshot_id": record.source_snapshot_id,
        "plan_id": target.plan_id,
        "patch_snapshot_id": target.patch_snapshot_id,
        "target_ref": (None if target.target_ref is None else {
            "kind": target.target_ref.kind.value,
            "skill_id": target.target_ref.skill_id,
            "ref_id": target.target_ref.ref_id,
        }),
        "check_id": target.check_id,
        "kind": record.kind.value,
        "rationale": record.rationale,
        "supersedes_feedback_id": record.supersedes_feedback_id,
        "feedback_id": record.feedback_id,
    }


def _load_entry(root: Path, entry: object):
    if not isinstance(entry, dict):
        raise ValueError("invalid feedback entry")
    supplied = entry.get("source_verify_artifact")
    if not isinstance(supplied, str):
        raise ValueError("invalid feedback source path")
    if source_path(root, supplied) != supplied:
        raise ValueError("source Verify artifact path is not canonical")
    _, report = load_verify_artifact(root, supplied)
    return restore_feedback_entry(entry, report), supplied


def restore_feedback_entry(entry: object, report) -> object:
    """Rebuild a feedback record using a validated frozen Verify report."""
    value = _mapping(entry, {"source_verify_artifact", "source_snapshot_id", "plan_id",
                             "patch_snapshot_id", "target_ref", "check_id", "kind",
                             "rationale", "supersedes_feedback_id", "feedback_id"},
                     "feedback entry")
    supplied = value["source_verify_artifact"]
    if not isinstance(supplied, str) or not supplied or Path(supplied).is_absolute() or (
        "\\" in supplied or ":" in supplied.split("/")[0]
        or any(part in {"", ".", ".."} for part in supplied.split("/"))
    ):
        raise ValueError("source Verify artifact path must be canonical repository-relative")
    target_ref = None if value["target_ref"] is None else _target_ref(value["target_ref"])
    record = HumanFeedbackService().record_verify(
        report, FeedbackKind(value["kind"]), value["rationale"],
        target_ref=target_ref, check_id=value["check_id"],
        expected_plan_id=value["plan_id"],
        expected_patch_snapshot_id=value["patch_snapshot_id"],
        supersedes_feedback_id=value["supersedes_feedback_id"],
    )
    if record.source_snapshot_id != value["source_snapshot_id"]:
        raise ValueError("feedback source snapshot mismatch")
    if record.feedback_id != value["feedback_id"]:
        raise ValueError("feedback_id mismatch")
    return record


def load_feedback_ledger(root: Path) -> tuple[HumanFeedbackLedger, dict[str, str]]:
    root = root.resolve()
    _, data, value = _read_repo_json(root, LEDGER_PATH, "feedback ledger")
    payload = _mapping(value, {"schema", "repository_id", "records", "ledger_id"},
                       "feedback ledger")
    if payload["schema"] != _SCHEMA or payload["repository_id"] != repository_identity(root):
        raise ValueError("feedback ledger schema or repository identity mismatch")
    loaded = [_load_entry(root, item) for item in _sequence(payload["records"], "records")]
    ledger = HumanFeedbackLedger(payload["repository_id"], tuple(item[0] for item in loaded))
    if ledger.ledger_id != payload["ledger_id"]:
        raise ValueError("ledger_id mismatch")
    sources = {record.feedback_id: source for record, source in loaded}
    canonical = {"schema": _SCHEMA, "repository_id": ledger.repository_id,
                 "records": [_entry(record, sources[record.feedback_id])
                             for record in ledger.records], "ledger_id": ledger.ledger_id}
    if data != _canonical_json(canonical):
        raise ValueError("feedback ledger is not canonical or its contents were tampered")
    return ledger, sources


def save_feedback_ledger(root: Path, ledger: HumanFeedbackLedger,
                         sources: dict[str, str]) -> str:
    root = root.resolve()
    if ledger.repository_id != repository_identity(root) or set(sources) != {
        record.feedback_id for record in ledger.records
    }:
        raise ValueError("feedback ledger repository or source mapping mismatch")
    entries = []
    for record in ledger.records:
        entry = _entry(record, source_path(root, sources[record.feedback_id]))
        restored, _ = _load_entry(root, entry)
        if restored != record:
            raise ValueError("feedback source does not match record")
        entries.append(entry)
    value = {"schema": _SCHEMA, "repository_id": ledger.repository_id,
             "records": entries, "ledger_id": ledger.ledger_id}
    return _write_json(root, LEDGER_PATH, value)
