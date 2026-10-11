"""Conservative repository-local sync for SkillForge-owned Agent Skills."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from skillforge.domain.agent_integration import (
    SCHEMA,
    AgentSkillExportPlan,
    AgentSkillSyncResult,
    CodeAgentTarget,
    digest,
    skill_name,
)
from skillforge.filesystem_safety import is_link as _is_link
from skillforge.filesystem_safety import no_symlink as _no_symlink

_ROOT = {CodeAgentTarget.CODEX: (".codex", "skills"),
         CodeAgentTarget.CLAUDE_CODE: (".claude", "skills")}
_FILES = {".skillforge.json", "SKILL.md", "references/evidence.md"}


def _owned(directory: Path, repository_id: str, target: CodeAgentTarget,
           root: Path) -> dict[str, object]:
    marker_path = directory / ".skillforge.json"
    _no_symlink(marker_path, root)
    if not marker_path.is_file():
        raise ValueError(f"existing Skill has no ownership marker: {directory}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ownership marker: {directory}") from exc
    if (not isinstance(marker, dict) or marker.get("schema") != SCHEMA
            or marker.get("target") != target.value
            or marker.get("repository_id") != repository_id
            or marker.get("agent_skill_name") != directory.name
            or marker.get("agent_skill_name") != skill_name(marker.get("source_skill_id"))
            or any(not isinstance(marker.get(key), str) or not marker[key].strip()
                   for key in ("registry_id", "source_snapshot_id", "source_skill_version"))
            or not isinstance(marker.get("evidence_ids"), list)
            or not isinstance(marker.get("file_hashes"), dict)
            or marker["file_hashes"].keys() != (_FILES - {".skillforge.json"})):
        raise ValueError(f"invalid or foreign ownership marker: {directory}")
    actual = set()
    directories = set()
    for path in directory.rglob("*"):
        _no_symlink(path, root)
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
        elif path.is_dir():
            directories.add(path.relative_to(directory).as_posix())
        else:
            raise ValueError(f"unexpected filesystem entry: {path}")
    if actual != _FILES or directories != {"references"}:
        raise ValueError(f"managed Skill has unexpected or missing files: {directory}")
    for relative, expected in marker["file_hashes"].items():
        if not isinstance(expected, str) or digest(
            (directory / relative).read_text(encoding="utf-8")
        ) != expected:
            raise ValueError(f"managed Skill was modified: {directory}")
    return marker


class RepoLocalSkillSync:
    """Preflight every managed directory before writing or removing any package."""

    def sync(self, repo_root: Path, plan: AgentSkillExportPlan) -> AgentSkillSyncResult:
        if not isinstance(plan, AgentSkillExportPlan):
            raise ValueError("a canonical AgentSkillExportPlan is required")
        repo_root = Path(repo_root).absolute()
        if not repo_root.is_dir() or any(_is_link(path) for path in
                                         (repo_root, *repo_root.parents)):
            raise ValueError("repository root must be a real directory")
        target_root = repo_root.joinpath(*_ROOT[plan.target])
        _no_symlink(target_root, repo_root)
        if target_root.exists() and not target_root.is_dir():
            raise ValueError("target skill root must be a directory")
        desired = {item.agent_skill_name: item for item in plan.packages}
        stale = []
        if target_root.exists():
            for directory in target_root.iterdir():
                _no_symlink(directory, repo_root)
                if directory.name in desired:
                    if not directory.is_dir():
                        raise ValueError(f"Skill collision: {directory}")
                    marker = _owned(directory, plan.repository_id, plan.target, repo_root)
                    if marker["source_skill_id"] != desired[directory.name].source_skill_id:
                        raise ValueError(f"Skill identity collision: {directory}")
                elif directory.name.startswith("skillforge-") and directory.is_dir():
                    marker_path = directory / ".skillforge.json"
                    if marker_path.exists() or _is_link(marker_path):
                        try:
                            _owned(directory, plan.repository_id, plan.target, repo_root)
                        except ValueError:
                            raise
                        stale.append(directory)
                    # No marker means user-owned; preserve it.
        written = []
        unchanged = []
        target_root.mkdir(parents=True, exist_ok=True)
        for package in plan.packages:
            directory = target_root / package.agent_skill_name
            _no_symlink(directory, repo_root)
            directory.mkdir(exist_ok=True)
            for file in package.files:
                path = directory.joinpath(*file.relative_path.split("/"))
                _no_symlink(path, repo_root)
                relative = path.relative_to(repo_root).as_posix()
                if path.exists() and path.read_text(encoding="utf-8") == file.content:
                    unchanged.append(relative)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(file.content, encoding="utf-8", newline="\n")
                written.append(relative)
        for directory in stale:
            shutil.rmtree(directory)
        return AgentSkillSyncResult(plan.target, plan.plan_id, tuple(written),
                                    tuple(unchanged), tuple(sorted(item.name for item in stale)))
