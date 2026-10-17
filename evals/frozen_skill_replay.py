"""Prepare one immutable, file-delivered Skill replay from a validated route.

The optional reference is a single package root (the directory containing
``.skillforge.json``, ``SKILL.md`` and ``references/evidence.md``). This helper
does not invoke models or select routes; it exports the supplied route against
the source checkout and writes an artifact-side manifest without task content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from skillforge.domain.agent_integration import AgentSkillPackage, CodeAgentTarget
from skillforge.domain.routing import SkillRouteResult
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.knowledge_trace import TRACE_DIRECTORY, TRACE_SLOTS, KnowledgeTrace


class ReplayPreparationError(ValueError):
    """Preflight failure with a non-sensitive, machine-readable reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _reject(reason: str, output: Path | None) -> None:
    if output is not None and _safe_empty_or_absent(output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "replay-manifest.json").write_text(
            json.dumps({"schema": 1, "status": "REJECTED", "reason": reason},
                       sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    raise ReplayPreparationError(reason)


def _safe_empty_or_absent(path: Path) -> bool:
    try:
        _reject_symlink_components(path)
        return not path.exists() or (path.is_dir() and not any(path.iterdir()))
    except (OSError, ReplayPreparationError):
        return False


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(current)):
            raise ReplayPreparationError("SYMLINK_PATH")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReplayPreparationError("SOURCE_GIT_UNAVAILABLE") from exc


def _reference_files(directory: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    _reject_symlink_components(directory)
    if not directory.is_dir():
        raise ReplayPreparationError("REFERENCE_PACKAGE_MISSING")
    paths = (".skillforge.json", "SKILL.md", "references/evidence.md")
    files: dict[str, bytes] = {}
    for relative in paths:
        path = directory / relative
        _reject_symlink_components(path)
        if not path.is_file():
            raise ReplayPreparationError("REFERENCE_PACKAGE_INCOMPLETE")
        try:
            # Historical Windows exports used CRLF; compare canonical text only.
            files[relative] = path.read_text(encoding="utf-8").encode("utf-8")
        except UnicodeError as exc:
            raise ReplayPreparationError("REFERENCE_TEXT_INVALID") from exc
    try:
        marker = json.loads(files[".skillforge.json"].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayPreparationError("REFERENCE_MARKER_INVALID") from exc
    for relative in ("SKILL.md", "references/evidence.md"):
        expected_hash = marker.get("file_hashes", {}).get(relative)
        if (not isinstance(expected_hash, str)
                or hashlib.sha256(files[relative]).hexdigest() != expected_hash):
            raise ReplayPreparationError("REFERENCE_MARKER_HASH_MISMATCH")
    return marker, files


def _compare_reference(package: AgentSkillPackage, directory: Path) -> None:
    marker, files = _reference_files(directory)
    expected = {
        "source_skill_id": package.source_skill_id,
        "source_skill_version": package.source_skill_version,
        "evidence_ids": list(package.evidence_ids),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ReplayPreparationError("REFERENCE_IDENTITY_MISMATCH")
    current = {item.relative_path: item.content.encode("utf-8") for item in package.files}
    if files["references/evidence.md"] != current["references/evidence.md"]:
        raise ReplayPreparationError("REFERENCE_EVIDENCE_MISMATCH")
    try:
        files["SKILL.md"].decode("utf-8")
        current["SKILL.md"].decode("utf-8")
    except UnicodeError as exc:
        raise ReplayPreparationError("REFERENCE_SKILL_INVALID") from exc


def prepare_frozen_skill_replay(
    route: SkillRouteResult,
    source_root: Path,
    expected_head: str,
    output_bundle: Path,
    *,
    reference_package: Path | None = None,
    trace_reference: str | None = None,
) -> dict[str, Any]:
    """Export the supplied route from a clean, pinned checkout and deliver files.

    ``reference_package`` is either absent or one package root, as documented in
    the module docstring. The package is installed below ``source_root`` while
    the manifest is written only to ``output_bundle``.
    """
    source_root = Path(source_root).absolute()
    output_bundle = Path(output_bundle).absolute()
    try:
        _reject_symlink_components(source_root)
        _reject_symlink_components(output_bundle)
        if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
            raise ReplayPreparationError("EXPECTED_HEAD_INVALID")
        if (not source_root.is_dir()
                or Path(_git(source_root, "rev-parse", "--show-toplevel")) != source_root):
            raise ReplayPreparationError("SOURCE_ROOT_INVALID")
        actual_head = _git(source_root, "rev-parse", "HEAD")
        if actual_head != expected_head:
            raise ReplayPreparationError("SOURCE_HEAD_MISMATCH")
        if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
            raise ReplayPreparationError("SOURCE_TRACKED_CHANGES")
        if output_bundle.exists() and (not output_bundle.is_dir() or any(output_bundle.iterdir())):
            raise ReplayPreparationError("OUTPUT_BUNDLE_NOT_EMPTY")
        if not isinstance(route, SkillRouteResult):
            raise ReplayPreparationError("ROUTE_INVALID")
        if not route.matches:
            raise ReplayPreparationError("NO_ROUTED_KNOWLEDGE")

        # Explicitly bind all source-derived observations to the actor checkout.
        trace = KnowledgeTrace(route.repository_id)
        plan = AgentSkillExporter().plan(
            route, CodeAgentTarget.CODEX, repository_root=source_root, trace=trace,
        )
        if not plan.packages:
            raise ReplayPreparationError("NO_ROUTED_KNOWLEDGE")
        if reference_package is not None:
            if len(plan.packages) != 1:
                raise ReplayPreparationError("REFERENCE_REQUIRES_SINGLE_PACKAGE")
            _compare_reference(plan.packages[0], Path(reference_package).absolute())

        destinations: list[tuple[AgentSkillPackage, Path]] = []
        knowledge_root = source_root / ".runtime" / "knowledge"
        _reject_symlink_components(knowledge_root)
        if knowledge_root.exists() and not knowledge_root.is_dir():
            raise ReplayPreparationError("KNOWLEDGE_ROOT_INVALID")
        for package in plan.packages:
            destination = knowledge_root / package.agent_skill_name
            _reject_symlink_components(destination)
            if destination.exists():
                raise ReplayPreparationError("KNOWLEDGE_DESTINATION_EXISTS")
            destinations.append((package, destination))

        manifest: dict[str, Any] = {
            "schema": 1,
            "status": "PREPARED",
            "source_head": actual_head,
            "source_root": str(source_root),
            "delivery": "file",
            "trace_reference": (
                f"{TRACE_DIRECTORY}/trace-{int(trace.run_id, 16) % TRACE_SLOTS}.json"
            ),
            "caller_trace_reference": trace_reference,
            "reference_comparison": "MATCHED" if reference_package else "NOT_PROVIDED",
            "packages": [],
        }
        for package, destination in destinations:
            files_info = []
            destination.mkdir(parents=True)
            for item in package.files:
                raw = item.content.encode("utf-8")
                if hashlib.sha256(raw).hexdigest() != item.content_sha256:
                    raise ReplayPreparationError("PACKAGE_HASH_MISMATCH")
                target = destination / item.relative_path
                _reject_symlink_components(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                if target.read_bytes() != raw:
                    raise ReplayPreparationError("PACKAGE_WRITE_MISMATCH")
                files_info.append({
                    "path": item.relative_path,
                    "sha256": item.content_sha256,
                    "utf8_bytes": len(raw),
                })
            marker_path = destination / ".skillforge.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for info in files_info:
                if info["path"] != ".skillforge.json":
                    if marker.get("file_hashes", {}).get(info["path"]) != info["sha256"]:
                        raise ReplayPreparationError("PACKAGE_MARKER_MISMATCH")
            manifest["packages"].append({
                "name": package.agent_skill_name,
                "source_skill_id": package.source_skill_id,
                "source_skill_version": package.source_skill_version,
                "evidence_ids": list(package.evidence_ids),
                "installed_at": str(destination.relative_to(source_root)),
                "files": files_info,
            })
        output_bundle.mkdir(parents=True, exist_ok=True)
        (output_bundle / "replay-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        return manifest
    except ReplayPreparationError as exc:
        _reject(exc.reason, output_bundle)
