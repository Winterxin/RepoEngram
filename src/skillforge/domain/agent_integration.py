"""Canonical, portable Agent Skill export contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import PortableProjectSkill, ProjectSkill, RuleGuidanceSkill

SCHEMA = "skillforge-agent-skill-v1"
_NAME = re.compile(r"skillforge-[a-f0-9]{32}")
_PORTABLE_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class CodeAgentTarget(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE = "claude_code"


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _path(value: str) -> None:
    if (not isinstance(value, str) or not value or "\\" in value
            or value.startswith("/") or ":" in value.split("/")[0]
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("file path must be canonical and relative")


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def skill_name(skill_id: str) -> str:
    _required(skill_id, "skill_id")
    return "skillforge-" + hashlib.sha256(skill_id.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class AgentSkillFile:
    relative_path: str
    content: str
    content_sha256: str

    def __post_init__(self) -> None:
        _path(self.relative_path)
        if not isinstance(self.content, str) or self.content_sha256 != digest(self.content):
            raise ValueError("file hash must match UTF-8 content")


@dataclass(frozen=True, slots=True)
class AgentSkillPackage:
    agent_skill_name: str
    source_skill_id: str
    source_skill_version: str
    target: CodeAgentTarget
    repository_id: str
    registry_id: str
    source_snapshot_id: str
    evidence_ids: tuple[str, ...]
    files: tuple[AgentSkillFile, ...]
    source_skill: PortableProjectSkill

    def __post_init__(self) -> None:
        for key in ("source_skill_id", "source_skill_version", "repository_id",
                    "registry_id", "source_snapshot_id"):
            _required(getattr(self, key), key)
        if (self.agent_skill_name != skill_name(self.source_skill_id)
                or not 1 <= len(self.agent_skill_name) <= 64
                or not _PORTABLE_NAME.fullmatch(self.agent_skill_name)
                or not _NAME.fullmatch(self.agent_skill_name)):
            raise ValueError("Agent Skill name differs from source Skill identity")
        if (not isinstance(self.source_skill, (ProjectSkill, RuleGuidanceSkill))
                or self.source_skill.skill_id != self.source_skill_id
                or self.source_skill.version != self.source_skill_version
                or self.source_skill.repository_id != self.repository_id
                or self.source_skill.source_snapshot_id != self.source_snapshot_id
                or self.source_skill.evidence_ids != self.evidence_ids):
            raise ValueError("package source Skill identity or Evidence mismatch")
        if not isinstance(self.target, CodeAgentTarget):
            raise ValueError("invalid target")
        if (not isinstance(self.evidence_ids, tuple)
                or self.evidence_ids != tuple(sorted(set(self.evidence_ids)))
                or any(not isinstance(key, str) or not key.strip() for key in self.evidence_ids)):
            raise ValueError("package Evidence IDs must be canonical")
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, AgentSkillFile) for item in self.files
        ):
            raise ValueError("package files must be AgentSkillFile tuple")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(set(paths))) or set(paths) != {
            ".skillforge.json", "SKILL.md", "references/evidence.md"
        }:
            raise ValueError("package files must be the canonical portable set")
        marker = json.loads(next(item.content for item in self.files
                                 if item.relative_path == ".skillforge.json"))
        expected = {
            "schema": SCHEMA, "target": self.target.value,
            "repository_id": self.repository_id, "registry_id": self.registry_id,
            "source_snapshot_id": self.source_snapshot_id,
            "source_skill_id": self.source_skill_id,
            "source_skill_version": self.source_skill_version,
            "agent_skill_name": self.agent_skill_name,
            "evidence_ids": list(self.evidence_ids),
            "file_hashes": {item.relative_path: item.content_sha256 for item in self.files
                            if item.relative_path != ".skillforge.json"},
        }
        if isinstance(self.source_skill, RuleGuidanceSkill):
            expected["source_kind"] = "RULE"
            expected["source_rule_id"] = self.source_skill.source_rule_id
        if marker != expected:
            raise ValueError("ownership metadata differs from package content")


def plan_fingerprint(repository_id: str, registry_id: str, source_snapshot_id: str,
                     target: CodeAgentTarget, packages: tuple[AgentSkillPackage, ...]) -> str:
    payload = (SCHEMA, repository_id, registry_id, source_snapshot_id, target.value,
               tuple((package.agent_skill_name, package.source_skill_id,
                      package.source_skill_version, package.evidence_ids,
                      tuple((item.relative_path, item.content_sha256) for item in package.files))
                     for package in packages))
    return digest(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@dataclass(frozen=True, slots=True)
class AgentSkillExportPlan:
    repository_id: str
    registry_id: str
    source_snapshot_id: str
    target: CodeAgentTarget
    packages: tuple[AgentSkillPackage, ...]
    plan_id: str

    def __post_init__(self) -> None:
        for key in ("repository_id", "registry_id", "source_snapshot_id"):
            _required(getattr(self, key), key)
        if not isinstance(self.target, CodeAgentTarget):
            raise ValueError("invalid target")
        if not isinstance(self.packages, tuple) or any(
            not isinstance(item, AgentSkillPackage) for item in self.packages
        ):
            raise ValueError("packages must be AgentSkillPackage tuple")
        names = tuple(item.agent_skill_name for item in self.packages)
        if names != tuple(sorted(set(names))):
            raise ValueError("packages must have unique canonical names and order")
        if any(item.target != self.target or item.repository_id != self.repository_id
               or item.registry_id != self.registry_id
               or item.source_snapshot_id != self.source_snapshot_id for item in self.packages):
            raise ValueError("package and plan identity mismatch")
        if self.plan_id != plan_fingerprint(self.repository_id, self.registry_id,
                                            self.source_snapshot_id, self.target,
                                            self.packages):
            raise ValueError("plan_id differs from canonical content")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class AgentSkillSyncResult:
    target: CodeAgentTarget
    plan_id: str
    written_files: tuple[str, ...]
    unchanged_files: tuple[str, ...]
    removed_packages: tuple[str, ...]
