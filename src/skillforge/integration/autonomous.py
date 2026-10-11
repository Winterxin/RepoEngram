"""Codex discovery and repository-bound, data-only production knowledge storage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from skillforge.domain.project_dna import ProjectDNA
from skillforge.domain.skills import SkillCompilationResult
from skillforge.filesystem_safety import no_symlink
from skillforge.productization.evolution_artifacts import _restore, load_baseline
from skillforge.repository.identity import repository_identity
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.verify.artifacts import _read_repo_json, _write_json

SNAPSHOT = Path(".skillforge/autonomous/snapshot.json")
BOOTSTRAP = Path(".agents/skills/skillforge-discovery/SKILL.md")
CONFIG = Path(".codex/config.toml")
SCHEMA = "skillforge-autonomous-v1"
APPROVAL_MODES = ("auto", "prompt", "writes", "approve")
APPROVAL_TOOLS = ("get_project_summary", "search_project_rules", "get_related_workflow",
                  "get_skill", "get_evidence", "bind_verify_check", "verify_patch")
CONTRACT = """---
name: skillforge-discovery
description: Query repository knowledge for repository-specific engineering tasks.
---

SkillForge provides evidence-backed project knowledge through the registered skillforge MCP.
For a repository-specific engineering task, call get_related_workflow with the user's task
and any known exact scope before editing. Do not assume a skill ID. Use get_project_summary
for project context and search_project_rules for additional constraints when needed.
Call get_skill for each relevant returned ID and read its complete executable instructions.
Follow applicability, ownership checks, decision procedure, constraints and exceptions.
Returned executable Project Skills are execution contracts: resolve applicability first,
and resolve all applicable REQUIRED obligations before task completion, including test changes
when required by an action. Matched scope is context, not an edit boundary. Keep rule guidance
conditional and respect contrary evidence. Perform the Skill's final completion gate.
Query get_evidence when supporting observations need inspection.
After modifying code, call verify_patch for the current working tree. Inspect the real
status and findings; repair failures and reverify. Report missing coverage or errors honestly.
If identity is missing or mismatched, report the error instead of using another repository.
"""


def repo_root(path: Path) -> Path:
    """Resolve cwd through Git, never through a task name or a global knowledge store."""
    requested = path.expanduser().absolute()
    no_symlink(requested, requested)
    result = subprocess.run(["git", "-C", str(requested), "rev-parse", "--show-toplevel"],
                            capture_output=True, encoding="utf-8", timeout=30, check=False)
    if result.returncode:
        raise ValueError("repository identity unavailable: a Git working directory is required")
    root = Path(result.stdout.strip()).resolve()
    no_symlink(requested, root)
    return root


def load_snapshot(root: Path):
    no_symlink(root / SNAPSHOT, root)
    if not (root / SNAPSHOT).is_file():
        raise ValueError("repository metadata unavailable: initialize Codex discovery first")
    try:
        _, _, value = _read_repo_json(root, SNAPSHOT, "autonomous snapshot", 16 * 1_048_576)
    except OSError as exc:
        raise ValueError("repository metadata unavailable: initialize Codex discovery") from exc
    if not isinstance(value, dict) or value.keys() != {
        "schema", "repository_id", "base_commit", "dna", "compilation"
    } or value["schema"] != SCHEMA:
        raise ValueError("invalid autonomous snapshot")
    if value["repository_id"] != repository_identity(root):
        raise ValueError("autonomous repository identity mismatch")
    dna = _restore(ProjectDNA, value["dna"])
    compiled = _restore(SkillCompilationResult, value["compilation"])
    if (dna.repository_id != value["repository_id"]
            or compiled.repository_id != dna.repository_id
            or compiled.source_snapshot_id != dna.snapshot_id):
        raise ValueError("autonomous DNA/compilation identity mismatch")
    registry = SkillRegistryBuilder().build(compiled)
    base = value["base_commit"]
    if not isinstance(base, str) or not base or base.startswith("-"):
        raise ValueError("invalid autonomous base commit")
    return dna, registry, base


def _registration(root):
    command = str(Path(sys.executable).resolve())
    args = ["-m", "skillforge.integration.mcp_server", "--repo", str(root)]
    source = str(Path(__file__).resolve().parents[2])
    section = ("\n[mcp_servers.skillforge]\ncommand = " + json.dumps(command)
               + "\nargs = " + json.dumps(args, ensure_ascii=False)
               + "\n[mcp_servers.skillforge.env]\nPYTHONPATH = "
               + json.dumps(source, ensure_ascii=False) + "\n")
    config_path = root / CONFIG
    bootstrap_path = root / BOOTSTRAP
    for path in (config_path, bootstrap_path, root / SNAPSHOT):
        no_symlink(path, root)
    text = config_path.read_text("utf-8") if config_path.exists() else ""
    existing = tomllib.loads(text).get("mcp_servers", {}).get("skillforge")
    expected = tomllib.loads(section)["mcp_servers"]["skillforge"]
    if existing is not None and any(existing.get(key) != value
                                    for key, value in expected.items()):
        raise ValueError("existing SkillForge MCP registration conflicts; preserve user config")
    if bootstrap_path.exists() and bootstrap_path.read_text("utf-8") != CONTRACT:
        raise ValueError("bootstrap collision; preserve user Skill")
    return config_path, bootstrap_path, text, section, existing


def _publish_registration(root, registration):
    config_path, bootstrap_path, text, section, existing = registration
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(CONTRACT, encoding="utf-8", newline="\n")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if existing is None:
        config_path.write_text(text + section, encoding="utf-8", newline="\n")
    return {"repository_id": repository_identity(root), "bootstrap": BOOTSTRAP.as_posix(),
            "config": CONFIG.as_posix(), "target_skill_preinjected": False}


def register(root: Path) -> dict:
    """Connect capability without mining, model calls or manufacturing knowledge."""
    root = repo_root(root)
    result = _publish_registration(root, _registration(root))
    result.update(semantic_calls=0, knowledge_snapshot_present=(root / SNAPSHOT).is_file())
    return result


def install(root: Path, dna: ProjectDNA, compiled: SkillCompilationResult,
            base_commit: str) -> dict:
    """Publish audited knowledge separately from zero-model capability connection."""
    root = repo_root(root)
    if (dna.repository_id != repository_identity(root)
            or compiled.repository_id != dna.repository_id
            or compiled.source_snapshot_id != dna.snapshot_id):
        raise ValueError("installation repository/snapshot mismatch")
    SkillRegistryBuilder().build(compiled)
    if not isinstance(base_commit, str) or not base_commit or base_commit.startswith("-"):
        raise ValueError("invalid base commit")
    base = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options",
                           f"{base_commit}^{{commit}}"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=30).stdout.strip()
    registration = _registration(root)
    _write_json(root, SNAPSHOT, {"schema": SCHEMA, "repository_id": dna.repository_id,
                               "base_commit": base, "dna": dna.to_dict(),
                               "compilation": compiled.to_dict()}, max_bytes=16 * 1_048_576)
    return _publish_registration(root, registration)


def refresh_connected(root: Path, baseline_path: Path | str, *,
                      expected_head: str, expected_snapshot_id: str) -> dict:
    """Publish a successful prepare to existing discovery; never initiate preparation."""
    root = repo_root(root)
    no_symlink(root / SNAPSHOT, root)
    if not (root / SNAPSHOT).exists():
        return {"status": "NOT_CONNECTED", "semantic_calls": 0}
    # Invalid existing metadata must not be silently replaced by an update.
    dna, _, base = load_snapshot(root)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=10).stdout.strip()
    if head != expected_head:
        raise ValueError("repository HEAD changed after prepare; discovery not refreshed")
    if base == expected_head and dna.snapshot_id == expected_snapshot_id:
        return {"status": "UNCHANGED", "semantic_calls": 0}
    baseline = load_baseline(root, baseline_path)
    if (baseline.head_commit != expected_head
            or baseline.source_dna.snapshot_id != expected_snapshot_id):
        raise ValueError("prepare baseline identity mismatch; discovery not refreshed")
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    return {"status": "REFRESHED", "base_commit": baseline.head_commit,
            "snapshot_id": baseline.source_dna.snapshot_id, "semantic_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Codex autonomous discovery locally")
    parser.add_argument("path", type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--tool-approval", choices=APPROVAL_MODES,
                        help="explicit opt-in project-local SkillForge tool approval mode")
    parser.add_argument("--approval-tool", action="append", choices=APPROVAL_TOOLS,
                        help="scope approval to these tools; defaults to the six current tools")
    args = parser.parse_args()
    root = repo_root(args.path)
    baseline = load_baseline(root, args.baseline)
    if args.approval_tool and args.tool_approval is None:
        parser.error("--approval-tool requires --tool-approval")
    result = install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    if args.tool_approval is not None:
        result["approval"] = configure_codex_approval(
            root, args.tool_approval, tuple(args.approval_tool or APPROVAL_TOOLS))
    print(json.dumps(result, ensure_ascii=False))
    return 0


def codex_registration_overrides(root: Path) -> list[str]:
    """Register generic capability for isolated CLI sessions that skip project trust layers."""
    root = repo_root(root)
    no_symlink(root / CONFIG, root)
    registration = tomllib.loads((root / CONFIG).read_text("utf-8"))[
        "mcp_servers"]["skillforge"]
    overrides = []
    for key in ("command", "args"):
        overrides.extend(["-c", f"mcp_servers.skillforge.{key}=" + json.dumps(
            registration[key], ensure_ascii=False)])
    overrides.extend(["-c", "mcp_servers.skillforge.env.PYTHONPATH=" + json.dumps(
        registration["env"]["PYTHONPATH"], ensure_ascii=False)])
    if "default_tools_approval_mode" in registration:
        overrides.extend(["-c", "mcp_servers.skillforge.default_tools_approval_mode="
                          + json.dumps(registration["default_tools_approval_mode"])])
    for name in APPROVAL_TOOLS:
        policy = registration.get("tools", {}).get(name, {})
        if "approval_mode" in policy:
            overrides.extend(["-c", f"mcp_servers.skillforge.tools.{name}.approval_mode="
                              + json.dumps(policy["approval_mode"])])
    return overrides


def configure_codex_approval(root: Path, mode: str,
                             tools: tuple[str, ...] = APPROVAL_TOOLS) -> dict:
    """User-selected project policy only; never assume trust or change global approval."""
    if mode not in APPROVAL_MODES or not tools or set(tools) - set(APPROVAL_TOOLS):
        raise ValueError("supported approval mode and SkillForge tool names are required")
    root = repo_root(root)
    path = root / CONFIG
    no_symlink(path, root)
    before = path.read_text("utf-8")
    document = tomllib.loads(before)
    if "skillforge" not in document.get("mcp_servers", {}):
        raise ValueError("SkillForge must be registered before configuring project approval")
    existing = document["mcp_servers"]["skillforge"].get("tools", {})
    additions = []
    for name in sorted(set(tools)):
        if name in existing:
            if existing[name].get("approval_mode") != mode:
                raise ValueError(f"existing tool policy conflicts; preserve user setting: {name}")
            continue
        additions.append(f'\n[mcp_servers.skillforge.tools.{name}]\napproval_mode = "{mode}"\n')
    after = before + "".join(additions)
    tomllib.loads(after)
    if after != before:
        path.write_text(after, encoding="utf-8", newline="\n")
    return {"scope": "project-local SkillForge tools", "mode": mode,
            "tools": sorted(set(tools)), "global_settings_modified": False}


if __name__ == "__main__":
    raise SystemExit(main())
