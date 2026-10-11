"""Canonical, data-only persistence for prepare receipts and Verify reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.verification import (
    GitPatchSnapshot,
    PatchChangeKind,
    PrepareVerifyReceipt,
    VerifyCheckKind,
    VerifyCheckResult,
    VerifyCheckStatus,
    VerifyCommandSpec,
    VerifyFinding,
    VerifyFindingSeverity,
    VerifyObligation,
    VerifyObligationKind,
    VerifyOverallStatus,
    VerifyPatchFile,
    VerifyPlan,
    VerifyReport,
    VerifyTargetRef,
    fingerprint,
)
from skillforge.filesystem_safety import no_symlink
from skillforge.repository.identity import repository_identity

_MAX_JSON_BYTES = 1_048_576
_PREPARE_DIR = Path(".skillforge/productization/prepare")
_VERIFY_DIR = Path(".skillforge/productization/verify")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(data: bytes, label: str, max_bytes: int = _MAX_JSON_BYTES) -> Any:
    if len(data) > max_bytes:
        raise ValueError(f"{label} exceeds its size limit")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc


def _mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.keys() != fields:
        raise ValueError(f"invalid {label} fields")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _resolve_repo_file(root: Path, supplied: Path | str) -> Path:
    value = Path(supplied).expanduser()
    candidate = value if value.is_absolute() else root / value
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path must remain inside the repository") from exc
    no_symlink(candidate, root)
    return candidate


def _read_repo_json(root: Path, supplied: Path | str, label: str,
                    max_bytes: int = _MAX_JSON_BYTES) -> tuple[Path, bytes, Any]:
    path = _resolve_repo_file(root, supplied)
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            raise ValueError(f"{label} must be a regular JSON file within its size limit")
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    return path, data, _parse_json(data, label, max_bytes)


def _confidence(value: Any) -> Confidence | None:
    if value is None:
        return None
    payload = _mapping(value, {"score", "method", "support"}, "confidence")
    support = payload["support"]
    parsed_support = None
    if support is not None:
        support = _mapping(support, {"supporting", "examined"}, "support")
        parsed_support = Support(support["supporting"], support["examined"])
    return Confidence(payload["score"], payload["method"], parsed_support)


def _evidence(value: Any) -> Evidence:
    payload = _mapping(value, {"evidence_id", "repository_id", "kind", "location",
                               "description", "provenance", "confidence"}, "Evidence")
    location = _mapping(payload["location"], {"path", "commit", "line", "column"},
                         "Evidence location")
    return Evidence(
        payload["evidence_id"], payload["repository_id"], EvidenceType(payload["kind"]),
        SourceLocation(location["path"], location["commit"], location["line"],
                       location["column"]), payload["description"], payload["provenance"],
        _confidence(payload["confidence"]),
    )


def _target_ref(value: Any) -> VerifyTargetRef:
    payload = _mapping(value, {"kind", "skill_id", "ref_id"}, "VerifyTargetRef")
    return VerifyTargetRef(VerifyObligationKind(payload["kind"]), payload["skill_id"],
                           payload["ref_id"])


def _obligation(value: Any) -> VerifyObligation:
    payload = _mapping(value, {"target_ref", "description", "condition", "source_status",
                               "evidence_ids", "source_confidence"}, "obligation")
    return VerifyObligation(
        _target_ref(payload["target_ref"]), payload["description"], payload["condition"],
        payload["source_status"], tuple(_sequence(payload["evidence_ids"], "evidence_ids")),
        _confidence(payload["source_confidence"]),
    )


def _command(value: Any, *, strict: bool = True) -> VerifyCommandSpec:
    fields = {"check_id", "kind", "argv", "cwd", "timeout_seconds", "covers"}
    if strict:
        payload = _mapping(value, fields, "check")
    else:
        required = {"check_id", "kind", "argv"}
        if not isinstance(value, dict) or not required <= value.keys() or value.keys() - fields:
            raise ValueError("invalid checks JSON fields")
        payload = value
    argv = _sequence(payload["argv"], "argv")
    covers = tuple(_target_ref(item) for item in _sequence(payload.get("covers", []), "covers"))
    return VerifyCommandSpec(payload["check_id"], VerifyCheckKind(payload["kind"]),
                             tuple(argv), payload.get("cwd", "."),
                             payload.get("timeout_seconds", 300), covers)


def _plan(value: Any) -> VerifyPlan:
    payload = _mapping(value, {"repository_id", "registry_id", "source_snapshot_id",
                               "selected_skills", "obligations", "checks", "evidence",
                               "plan_id"}, "VerifyPlan")
    skills = []
    for item in _sequence(payload["selected_skills"], "selected_skills"):
        pair = _sequence(item, "selected Skill")
        if len(pair) != 2:
            raise ValueError("selected Skill must contain an ID and version")
        skills.append(tuple(pair))
    return VerifyPlan(
        payload["repository_id"], payload["registry_id"], payload["source_snapshot_id"],
        tuple(skills),
        tuple(_obligation(item) for item in _sequence(payload["obligations"], "obligations")),
        tuple(_command(item) for item in _sequence(payload["checks"], "checks")),
        tuple(_evidence(item) for item in _sequence(payload["evidence"], "evidence")),
        payload["plan_id"],
    )


def prepare_receipt_path(receipt: PrepareVerifyReceipt) -> str:
    return (_PREPARE_DIR / f"{receipt.receipt_id}.json").as_posix()


def _write_json(root: Path, relative_path: Path, value: object,
                *, reject_different_existing: bool = False,
                max_bytes: int = _MAX_JSON_BYTES) -> str:
    root = root.resolve()
    path = _resolve_repo_file(root, relative_path)
    data = _canonical_json(value)
    if len(data) > max_bytes:
        raise ValueError("artifact exceeds its size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    no_symlink(path, root)
    if path.exists():
        if not path.is_file():
            raise ValueError("artifact path is not a regular file")
        if reject_different_existing:
            if path.read_bytes() != data:
                raise ValueError("existing prepare receipt conflicts with deterministic receipt")
            return relative_path.as_posix()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".skillforge-write-",
                                         delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        no_symlink(temp_path, root)
        no_symlink(path, root)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return relative_path.as_posix()


def save_prepare_receipt(root: Path, receipt: PrepareVerifyReceipt) -> str:
    return _write_json(root, Path(prepare_receipt_path(receipt)), receipt.to_dict(),
                       reject_different_existing=True)


def load_prepare_receipt(root: Path, supplied: Path | str) -> PrepareVerifyReceipt:
    root = root.resolve()
    _, data, value = _read_repo_json(root, supplied, "prepare receipt")
    payload = _mapping(value, {"schema", "repository_id", "registry_id", "source_snapshot_id",
                               "base_commit", "plan", "ignored_patch_paths", "receipt_id"},
                       "prepare receipt")
    if payload["schema"] != "skillforge-prepare-verify-receipt-v2":
        raise ValueError("unsupported prepare receipt schema")
    plan = _plan(payload["plan"])
    receipt = PrepareVerifyReceipt(
        payload["repository_id"], payload["registry_id"], payload["source_snapshot_id"],
        payload["base_commit"], plan,
        tuple(_sequence(payload["ignored_patch_paths"], "ignored_patch_paths")),
        payload["receipt_id"],
    )
    if receipt.repository_id != repository_identity(root):
        raise ValueError("prepare receipt repository identity mismatch")
    for relative_path in receipt.ignored_patch_paths:
        no_symlink(root.joinpath(*relative_path.split("/")), root)
    if data != _canonical_json(receipt.to_dict()):
        raise ValueError("prepare receipt is not canonical or its contents were tampered")
    return receipt


def load_checks(root: Path, supplied: Path | str) -> tuple[tuple[VerifyCommandSpec, ...], str]:
    root = root.resolve()
    path, _, value = _read_repo_json(root, supplied, "checks")
    payload = _mapping(value, {"checks"}, "checks document")
    checks = tuple(_command(item, strict=False)
                   for item in _sequence(payload["checks"], "checks"))
    return checks, path.relative_to(root).as_posix()


def save_verify_report(root: Path, plan: VerifyPlan, report: VerifyReport) -> str:
    if not isinstance(plan, VerifyPlan) or not isinstance(report, VerifyReport):
        raise ValueError("Verify artifact requires a plan and report")
    if report.plan_id != plan.plan_id or report.repository_id != plan.repository_id:
        raise ValueError("Verify artifact Plan identity mismatch")
    # Re-running the same patch/plan can produce different logs or outcomes.
    # Keep each actual result immutable instead of colliding with an earlier run.
    report_key = fingerprint(("verify-report-artifact-v2", report.plan_id,
                              report.patch.snapshot_id, report.to_dict()))
    filename = f"{report_key}.json"
    return _write_json(root, _VERIFY_DIR / filename, {
        "schema": "skillforge-verify-artifact-v2",
        "plan": plan.to_dict(), "report": report.to_dict(),
    }, reject_different_existing=True)


def load_verify_artifact(root: Path, supplied: Path | str) -> tuple[VerifyPlan, VerifyReport]:
    root = root.resolve()
    _, data, value = _read_repo_json(root, supplied, "Verify artifact")
    plan, report = restore_verify_artifact(root, value)
    if data != _canonical_json({"schema": "skillforge-verify-artifact-v2",
                                "plan": plan.to_dict(), "report": report.to_dict()}):
        raise ValueError("Verify artifact is not canonical or its contents were tampered")
    return plan, report


def restore_verify_artifact(root: Path, value: object) -> tuple[VerifyPlan, VerifyReport]:
    """Rebuild a Verify artifact from canonical in-memory data."""
    root = root.resolve()
    payload = _mapping(value, {"schema", "plan", "report"}, "Verify artifact")
    if payload["schema"] != "skillforge-verify-artifact-v2":
        raise ValueError("unsupported Verify artifact schema")
    plan = _plan(payload["plan"])
    report_data = _mapping(payload["report"], {
        "repository_id", "registry_id", "source_snapshot_id", "plan_id", "patch",
        "checks", "obligations", "findings", "overall_status", "evidence",
    }, "VerifyReport")
    patch_data = _mapping(report_data["patch"], {
        "repository_id", "base_commit", "resolved_base_commit", "head_commit",
        "files", "tracked_diff_sha256", "snapshot_id",
    }, "GitPatchSnapshot")
    patch = GitPatchSnapshot(
        patch_data["repository_id"], patch_data["base_commit"],
        patch_data["resolved_base_commit"], patch_data["head_commit"],
        tuple(VerifyPatchFile(**{
            **_mapping(item, {"path", "kind", "previous_path", "content_sha256"},
                       "VerifyPatchFile"),
            "kind": PatchChangeKind(item["kind"]),
        }) for item in _sequence(patch_data["files"], "patch files")),
        patch_data["tracked_diff_sha256"], patch_data["snapshot_id"],
    )
    checks = []
    for item in _sequence(report_data["checks"], "report checks"):
        check = _mapping(item, {"check_id", "kind", "status", "argv", "cwd", "exit_code",
                                "stdout", "stderr", "covered_refs"}, "VerifyCheckResult")
        checks.append(VerifyCheckResult(
            check["check_id"], VerifyCheckKind(check["kind"]),
            VerifyCheckStatus(check["status"]), tuple(_sequence(check["argv"], "argv")),
            check["cwd"], check["exit_code"], check["stdout"], check["stderr"],
            tuple(_target_ref(ref) for ref in _sequence(check["covered_refs"], "covered_refs")),
        ))
    findings = []
    for item in _sequence(report_data["findings"], "findings"):
        finding = _mapping(item, {"code", "severity", "message", "check_id", "target_ref",
                                  "evidence_ids"}, "VerifyFinding")
        findings.append(VerifyFinding(
            finding["code"], VerifyFindingSeverity(finding["severity"]), finding["message"],
            finding["check_id"],
            _target_ref(finding["target_ref"]) if finding["target_ref"] is not None else None,
            tuple(_sequence(finding["evidence_ids"], "finding evidence_ids")),
        ))
    report = VerifyReport(
        report_data["repository_id"], report_data["registry_id"],
        report_data["source_snapshot_id"], report_data["plan_id"], patch, tuple(checks),
        tuple(_obligation(item) for item in _sequence(report_data["obligations"],
                                                   "report obligations")),
        tuple(findings), VerifyOverallStatus(report_data["overall_status"]),
        tuple(_evidence(item) for item in _sequence(report_data["evidence"],
                                                    "report evidence")), plan,
    )
    if report.repository_id != repository_identity(root):
        raise ValueError("Verify artifact repository identity mismatch")
    if value != {"schema": "skillforge-verify-artifact-v2",
                 "plan": plan.to_dict(), "report": report.to_dict()}:
        raise ValueError("Verify artifact is not canonical or its contents were tampered")
    return plan, report
