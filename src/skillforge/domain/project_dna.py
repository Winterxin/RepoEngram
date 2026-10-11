"""Canonical, evidence-closed Project DNA knowledge contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from skillforge.domain.evidence import Confidence, Evidence, Support
from skillforge.domain.serialization import to_json_value

if TYPE_CHECKING:
    from skillforge.domain.semantic_antipattern import SemanticAntiPatternDraft
    from skillforge.domain.semantic_workflow import SemanticWorkflowDraft


class RuleType(StrEnum):
    ARCHITECTURE = "architecture"
    CONVENTION = "convention"
    TESTING = "testing"
    BUSINESS = "business"
    RELEASE = "release"


class ProjectKnowledgeStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
    CONFLICTED = "conflicted"
    MANUAL_OVERRIDE = "manual_override"


class ProjectScopeKind(StrEnum):
    REPOSITORY = "repository"
    DIRECTORY = "directory"
    MODULE = "module"
    FILE = "file"
    FILE_TYPE = "file_type"
    DOMAIN = "domain"


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _canonical_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ) or values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be a sorted unique tuple of nonempty strings")


@dataclass(frozen=True, slots=True)
class ProjectScope:
    """Selectors name existing entities of kind; no scope expansion is performed."""
    kind: ProjectScopeKind
    selectors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProjectScopeKind):
            raise ValueError("scope kind must be ProjectScopeKind")
        _canonical_ids(self.selectors, "scope selectors")
        if (self.kind == ProjectScopeKind.REPOSITORY) != (not self.selectors):
            raise ValueError("repository scope has no selectors; other scopes require selectors")


@dataclass(frozen=True, slots=True)
class ProjectRuleException:
    description: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.description, "exception description")
        _canonical_ids(self.evidence_ids, "exception Evidence IDs")


@dataclass(frozen=True, slots=True)
class ProjectRule:
    rule_id: str
    repository_id: str
    type: RuleType
    statement: str
    scope: ProjectScope
    status: ProjectKnowledgeStatus
    evidence_ids: tuple[str, ...]
    support: Support | None = None
    confidence: Confidence | None = None
    exceptions: tuple[ProjectRuleException, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.rule_id, "rule_id")
        _required_text(self.repository_id, "repository_id")
        _required_text(self.statement, "rule statement")
        if not isinstance(self.type, RuleType) or not isinstance(self.scope, ProjectScope):
            raise ValueError("rule type and scope must use Project DNA contracts")
        if not isinstance(self.status, ProjectKnowledgeStatus):
            raise ValueError("rule status must be ProjectKnowledgeStatus")
        _canonical_ids(self.evidence_ids, "rule Evidence IDs")
        if not self.evidence_ids and self.status != ProjectKnowledgeStatus.MANUAL_OVERRIDE:
            raise ValueError("automatically derived rules require Evidence")
        if not isinstance(self.exceptions, tuple) or any(
            not isinstance(item, ProjectRuleException) for item in self.exceptions
        ):
            raise ValueError("rule exceptions must be evidence-backed objects")
        if self.status != ProjectKnowledgeStatus.MANUAL_OVERRIDE and any(
            not item.evidence_ids for item in self.exceptions
        ):
            raise ValueError("automatically derived rule exceptions require Evidence")
        if any(not set(item.evidence_ids) <= set(self.evidence_ids) for item in self.exceptions):
            raise ValueError("exception Evidence must be included in rule Evidence")


@dataclass(frozen=True, slots=True)
class ProjectWorkflow:
    workflow_id: str
    repository_id: str
    status: ProjectKnowledgeStatus
    draft: SemanticWorkflowDraft

    def __post_init__(self) -> None:
        _required_text(self.workflow_id, "workflow_id")
        if (
            self.workflow_id != self.draft.draft_id
            or self.repository_id != self.draft.repository_id
        ):
            raise ValueError("workflow identity must match its accepted semantic draft")
        if not isinstance(self.status, ProjectKnowledgeStatus):
            raise ValueError("workflow status must be ProjectKnowledgeStatus")
        _canonical_ids(self.draft.evidence_ids, "workflow Evidence IDs")
        nested = {
            *self.draft.name_evidence_ids,
            *self.draft.intent_evidence_ids,
            *self.draft.scope.evidence_ids,
            *(
                key
                for collection in (
                    self.draft.roles,
                    self.draft.ordered_steps,
                    self.draft.required_steps,
                    self.draft.optional_steps,
                    self.draft.conditional_steps,
                    self.draft.exceptions,
                )
                for item in collection
                for key in item.evidence_ids
            ),
        }
        if not self.draft.evidence_ids or not nested <= set(self.draft.evidence_ids):
            raise ValueError("workflow draft Evidence must include all nested references")


@dataclass(frozen=True, slots=True)
class ProjectAntiPattern:
    anti_pattern_id: str
    repository_id: str
    status: ProjectKnowledgeStatus
    draft: SemanticAntiPatternDraft

    def __post_init__(self) -> None:
        _required_text(self.anti_pattern_id, "anti_pattern_id")
        if (
            self.anti_pattern_id != self.draft.draft_id
            or self.repository_id != self.draft.repository_id
        ):
            raise ValueError("Anti-pattern identity must match its accepted semantic draft")
        if not isinstance(self.status, ProjectKnowledgeStatus):
            raise ValueError("Anti-pattern status must be ProjectKnowledgeStatus")
        _canonical_ids(self.draft.evidence_ids, "Anti-pattern Evidence IDs")
        if self.draft.confidence.support is not None:
            raise ValueError("semantic Anti-pattern must not contain Support")
        nested = {
            *self.draft.name_evidence_ids,
            *self.draft.description_evidence_ids,
            *self.draft.avoid_behavior_evidence_ids,
            *self.draft.negative_evidence_ids,
            *self.draft.decision_evidence_ids,
            *self.draft.scope.evidence_ids,
            *(key for item in self.draft.exceptions for key in item.evidence_ids),
        }
        if not self.draft.evidence_ids or nested != set(self.draft.evidence_ids):
            raise ValueError("Anti-pattern draft Evidence must match nested references")


@dataclass(frozen=True, slots=True)
class ProjectDNA:
    repository_id: str
    rules: tuple[ProjectRule, ...]
    workflows: tuple[ProjectWorkflow, ...]
    anti_patterns: tuple[ProjectAntiPattern, ...]
    evidence: tuple[Evidence, ...]
    schema_version: str = "1"
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        if self.schema_version != "1":
            raise ValueError("unsupported Project DNA schema version")
        for name, items, key in (
            ("rules", self.rules, "rule_id"),
            ("workflows", self.workflows, "workflow_id"),
            ("anti_patterns", self.anti_patterns, "anti_pattern_id"),
            ("evidence", self.evidence, "evidence_id"),
        ):
            ids = tuple(getattr(item, key) for item in items)
            _canonical_ids(ids, f"Project DNA {name} IDs")
            if any(item.repository_id != self.repository_id for item in items):
                raise ValueError(f"Project DNA {name} repository_id mismatch")
        referenced = {
            evidence_id for item in self.rules for evidence_id in item.evidence_ids
        } | {
            evidence_id for item in self.workflows for evidence_id in item.draft.evidence_ids
        } | {
            evidence_id for item in self.anti_patterns for evidence_id in item.draft.evidence_ids
        }
        if referenced != {item.evidence_id for item in self.evidence}:
            raise ValueError("Project DNA Evidence must exactly match knowledge references")
        canonical = json.dumps(
            self._content_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        object.__setattr__(self, "snapshot_id", hashlib.sha256(canonical.encode()).hexdigest())

    def _content_dict(self) -> dict[str, object]:
        return {
            "repository_id": self.repository_id,
            "schema_version": self.schema_version,
            "rules": to_json_value(self.rules),
            "workflows": to_json_value(self.workflows),
            "anti_patterns": to_json_value(self.anti_patterns),
            "evidence": to_json_value(self.evidence),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "snapshot_id": self.snapshot_id}
