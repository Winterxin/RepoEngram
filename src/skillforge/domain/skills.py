"""Contracts for workflow Skills and independently sourced rule guidance."""

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.evidence import Confidence, Evidence, Support
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRuleException,
    ProjectScope,
    RuleType,
)
from skillforge.domain.semantic_structural_rule import SEMANTIC_METHOD
from skillforge.domain.serialization import to_json_value


class SkillStepKind(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"


class SkillConstraintEnforcement(StrEnum):
    REQUIRED = "required"
    ADVISORY = "advisory"


class SkillSkipReason(StrEnum):
    DEPRECATED = "DEPRECATED"
    CONFLICTED = "CONFLICTED"
    UNKNOWN_SCOPE = "UNKNOWN_SCOPE"


@dataclass(frozen=True, slots=True)
class SkillTrigger:
    task_description: str
    scope: ProjectScope
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillParameter:
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class SkillStep:
    step_id: str
    instruction: str
    kind: SkillStepKind
    condition: str | None
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillException:
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillCompletionCondition:
    step_id: str
    condition: str | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillConstraint:
    rule_id: str
    rule_type: RuleType
    statement: str
    scope: ProjectScope
    source_status: ProjectKnowledgeStatus
    enforcement: SkillConstraintEnforcement
    exceptions: tuple[ProjectRuleException, ...]
    evidence_ids: tuple[str, ...]
    source_support: Support | None = None
    source_confidence: Confidence | None = None


@dataclass(frozen=True, slots=True)
class SkillRisk:
    anti_pattern_id: str
    name: str
    description: str
    avoid_behavior: str
    scope: ProjectScope
    source_status: ProjectKnowledgeStatus
    confidence: Confidence
    exceptions: tuple[SkillException, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillProcedure:
    """Executable compilation; empty step Evidence denotes generic reasoning only."""

    steps: tuple[SkillStep, ...]
    constraints: tuple[SkillConstraint, ...]
    verify: tuple[SkillStep, ...]
    excluded_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for collection in (self.steps, self.verify):
            if not isinstance(collection, tuple) or not collection:
                raise ValueError("procedure requires executable steps and verification")
            ids = [item.step_id for item in collection]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate procedure step ID")
            for item in collection:
                _required_text(item.instruction, "procedure instruction")
                _canonical_ids(item.evidence_ids, "procedure Evidence")
                if not isinstance(item.kind, SkillStepKind) or (
                    (item.kind == SkillStepKind.CONDITIONAL) != bool(item.condition)
                ):
                    raise ValueError("procedure condition must match step kind")
        _canonical_ids(self.excluded_rule_ids, "excluded rule IDs")
        ids = [item.rule_id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate procedure constraint")
        for item in self.constraints:
            if not item.evidence_ids or item.source_status in {
                ProjectKnowledgeStatus.CONFLICTED, ProjectKnowledgeStatus.DEPRECATED,
            }:
                raise ValueError("procedure constraints require usable Evidence-backed knowledge")
            expected = (SkillConstraintEnforcement.REQUIRED if item.source_status in {
                ProjectKnowledgeStatus.CONFIRMED, ProjectKnowledgeStatus.MANUAL_OVERRIDE,
            } else SkillConstraintEnforcement.ADVISORY)
            if item.enforcement != expected:
                raise ValueError("procedure constraint promotes source authority")

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted({key for item in (*self.steps, *self.verify, *self.constraints)
                             for key in item.evidence_ids}))


@dataclass(frozen=True, slots=True)
class ProjectSkill:
    skill_id: str
    repository_id: str
    source_snapshot_id: str
    version: str
    description: str
    trigger: SkillTrigger
    parameters: tuple[SkillParameter, ...]
    steps: tuple[SkillStep, ...]
    workflow_id: str
    source_status: ProjectKnowledgeStatus
    source_support: Support
    source_confidence: Confidence
    source_workflow_evidence_ids: tuple[str, ...]
    constraints: tuple[SkillConstraint, ...]
    risks: tuple[SkillRisk, ...]
    exceptions: tuple[SkillException, ...]
    completion_conditions: tuple[SkillCompletionCondition, ...]
    verify_rule_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    procedure: SkillProcedure | None = None

    def __post_init__(self) -> None:
        if self.procedure is not None and not set(self.procedure.evidence_ids) <= set(
            self.evidence_ids
        ):
            raise ValueError("procedure Evidence outside Skill closure")
        for name in ("skill_id", "repository_id", "source_snapshot_id", "version", "workflow_id"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.source_status, ProjectKnowledgeStatus) or self.source_status not in {
            ProjectKnowledgeStatus.CANDIDATE,
            ProjectKnowledgeStatus.CONFIRMED,
            ProjectKnowledgeStatus.MANUAL_OVERRIDE,
        }:
            raise ValueError("Skill source status must be compilable")
        if not isinstance(self.source_support, Support) or not isinstance(
            self.source_confidence, Confidence
        ):
            raise ValueError("Skill source Support and Confidence are required")
        if not isinstance(self.parameters, tuple) or any(
            not isinstance(item, SkillParameter) for item in self.parameters
        ):
            raise ValueError("Skill parameters must be SkillParameter tuple")
        _canonical_ids(self.source_workflow_evidence_ids, "source workflow Evidence IDs")
        if not self.source_workflow_evidence_ids:
            raise ValueError("Skill must retain source workflow Evidence")
        _canonical_ids(self.evidence_ids, "Skill Evidence IDs")
        _canonical_ids(self.verify_rule_ids, "verify rule IDs")
        for name, values, kind in (
            ("steps", self.steps, SkillStep),
            ("constraints", self.constraints, SkillConstraint),
            ("risks", self.risks, SkillRisk),
            ("exceptions", self.exceptions, SkillException),
            ("completion conditions", self.completion_conditions, SkillCompletionCondition),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, kind) for item in values):
                raise ValueError(f"Skill {name} must be a tuple of {kind.__name__}")
        for name, values, attr in (
            ("step", self.steps, "step_id"),
            ("constraint", self.constraints, "rule_id"),
            ("risk", self.risks, "anti_pattern_id"),
            ("completion condition", self.completion_conditions, "step_id"),
        ):
            identifiers = [getattr(item, attr) for item in values]
            if any(not isinstance(key, str) or not key.strip() for key in identifiers):
                raise ValueError(f"Skill {name} IDs must be nonempty")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate Skill {name} ID")
        constraints = {item.rule_id: item for item in self.constraints}
        for constraint in self.constraints:
            if (
                not isinstance(constraint.rule_type, RuleType)
                or not isinstance(constraint.scope, ProjectScope)
                or not isinstance(constraint.source_status, ProjectKnowledgeStatus)
            ):
                raise ValueError("Skill constraint source contract is invalid")
            expected = (
                SkillConstraintEnforcement.REQUIRED
                if constraint.source_status
                in {
                    ProjectKnowledgeStatus.CONFIRMED,
                    ProjectKnowledgeStatus.MANUAL_OVERRIDE,
                }
                else SkillConstraintEnforcement.ADVISORY
            )
            if constraint.source_status == ProjectKnowledgeStatus.DEPRECATED or (
                not isinstance(constraint.enforcement, SkillConstraintEnforcement)
                or constraint.enforcement != expected
            ):
                raise ValueError("Skill constraint enforcement contradicts source status")
            if (
                constraint.source_support is not None
                and not isinstance(constraint.source_support, Support)
            ) or (
                constraint.source_confidence is not None
                and not isinstance(constraint.source_confidence, Confidence)
            ):
                raise ValueError("Skill constraint source metrics have invalid types")
            if any(
                not set(exception.evidence_ids) <= set(constraint.evidence_ids)
                for exception in constraint.exceptions
            ):
                raise ValueError("constraint exception Evidence must belong to its rule")
        for key in self.verify_rule_ids:
            item = constraints.get(key)
            if (
                item is None
                or item.enforcement != SkillConstraintEnforcement.REQUIRED
                or (item.rule_type not in {RuleType.TESTING, RuleType.ARCHITECTURE})
            ):
                raise ValueError(
                    "verify rule IDs must reference required testing/architecture constraints"
                )
        steps = {item.step_id: item for item in self.steps}
        completion = {item.step_id: item for item in self.completion_conditions}
        for step in self.steps:
            if not isinstance(step.kind, SkillStepKind):
                raise ValueError("invalid Skill step kind")
            if step.kind in {SkillStepKind.REQUIRED, SkillStepKind.OPTIONAL}:
                if step.condition is not None:
                    raise ValueError("required/optional Skill step cannot have a condition")
            elif not isinstance(step.condition, str) or not step.condition.strip():
                raise ValueError("conditional Skill step requires a condition")
            item = completion.get(step.step_id)
            if step.kind == SkillStepKind.OPTIONAL:
                if item is not None:
                    raise ValueError("optional Skill step cannot have completion condition")
            elif item is None or item.condition != step.condition:
                raise ValueError(
                    "required/conditional Skill step needs matching completion condition"
                )
        if set(completion) - set(steps):
            raise ValueError("completion condition references unknown Skill step")
        for risk in self.risks:
            if (
                not isinstance(risk.scope, ProjectScope)
                or not isinstance(risk.source_status, ProjectKnowledgeStatus)
                or risk.source_status == ProjectKnowledgeStatus.DEPRECATED
            ):
                raise ValueError("Skill risk source contract is invalid")
            if not isinstance(risk.confidence, Confidence) or risk.confidence.support is not None:
                raise ValueError("Skill risk Confidence must not contain Support")
            if any(
                not set(exception.evidence_ids) <= set(risk.evidence_ids)
                for exception in risk.exceptions
            ):
                raise ValueError("risk exception Evidence must belong to its Anti-pattern")
        if not isinstance(self.trigger, SkillTrigger) or not isinstance(
            self.trigger.scope, ProjectScope
        ):
            raise ValueError("Skill trigger must be SkillTrigger")
        referenced = {
            *self.trigger.evidence_ids,
            *self.source_workflow_evidence_ids,
            *(
                key
                for collection in (
                    self.steps,
                    self.constraints,
                    self.risks,
                    self.exceptions,
                    self.completion_conditions,
                )
                for item in collection
                for key in item.evidence_ids
            ),
            *(
                key
                for constraint in self.constraints
                for exception in constraint.exceptions
                for key in exception.evidence_ids
            ),
            *(
                key
                for risk in self.risks
                for exception in risk.exceptions
                for key in exception.evidence_ids
            ),
        }
        if self.evidence_ids != tuple(sorted(referenced)):
            raise ValueError("Skill Evidence must exactly match included references")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class RuleGuidanceSkill:
    """Semantic-gated conditional method; preserves the candidate's authority."""

    skill_id: str
    repository_id: str
    source_snapshot_id: str
    version: str
    description: str
    trigger: SkillTrigger
    constraint: SkillConstraint
    source_rule_id: str
    source_rule_type: RuleType
    source_status: ProjectKnowledgeStatus
    source_scope: ProjectScope
    source_support: Support | None
    source_confidence: Confidence | None
    semantic_candidate_id: str
    semantic_method: str
    semantic_reason: str
    semantic_evidence_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    procedure: SkillProcedure | None = None

    def __post_init__(self) -> None:
        for name in ("skill_id", "repository_id", "source_snapshot_id", "version",
                     "description", "source_rule_id", "semantic_candidate_id",
                     "semantic_method", "semantic_reason"):
            _required_text(getattr(self, name), name)
        if (not isinstance(self.trigger, SkillTrigger)
                or not isinstance(self.constraint, SkillConstraint)
                or not isinstance(self.source_scope, ProjectScope)
                or self.source_rule_type not in {RuleType.ARCHITECTURE, RuleType.CONVENTION}
                or self.source_status != ProjectKnowledgeStatus.CANDIDATE
                or self.source_confidence is not None
                or self.constraint.enforcement != SkillConstraintEnforcement.ADVISORY
                or self.constraint.rule_id != self.source_rule_id
                or self.constraint.rule_type != self.source_rule_type
                or self.constraint.source_status != self.source_status
                or self.constraint.scope != self.source_scope
                or self.constraint.source_support != self.source_support
                or self.constraint.source_confidence != self.source_confidence
                or self.constraint.statement != self.description
                or self.trigger.task_description != self.description
                or self.trigger.scope != self.source_scope
                or self.trigger.evidence_ids != self.constraint.evidence_ids
                or self.semantic_candidate_id != self.source_rule_id):
            raise ValueError("Rule Guidance must preserve its advisory canonical source")
        expected_method = (SEMANTIC_METHOD if self.source_rule_type == RuleType.ARCHITECTURE
                           else "llm_semantic_history_tendency_v1")
        if self.semantic_method != expected_method or not isinstance(
            self.source_support, Support
        ):
            raise ValueError("Rule Guidance requires structural semantic audit and Support")
        _canonical_ids(self.semantic_evidence_ids, "semantic Evidence IDs")
        _canonical_ids(self.evidence_ids, "Guidance Evidence IDs")
        if (not self.semantic_evidence_ids
                or not set(self.semantic_evidence_ids) <= set(self.constraint.evidence_ids)
                or self.evidence_ids != tuple(sorted(set(self.constraint.evidence_ids) | (
                    set(self.procedure.evidence_ids) if self.procedure is not None else set()
                )))
                or any(not set(item.evidence_ids) <= set(self.evidence_ids)
                       for item in self.constraint.exceptions)):
            raise ValueError("Rule Guidance Evidence must close its source and audit references")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


PortableProjectSkill = ProjectSkill | RuleGuidanceSkill


@dataclass(frozen=True, slots=True)
class SkippedWorkflow:
    workflow_id: str
    reason: SkillSkipReason


@dataclass(frozen=True, slots=True)
class SkillCompilationResult:
    repository_id: str
    source_snapshot_id: str
    skills: tuple[PortableProjectSkill, ...]
    skipped_workflows: tuple[SkippedWorkflow, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        _required_text(self.source_snapshot_id, "source_snapshot_id")
        if not isinstance(self.skills, tuple) or any(
            not isinstance(item, (ProjectSkill, RuleGuidanceSkill)) for item in self.skills
        ):
            raise ValueError("compilation skills must be portable Skill tuple")
        if not isinstance(self.skipped_workflows, tuple) or any(
            not isinstance(item, SkippedWorkflow) for item in self.skipped_workflows
        ):
            raise ValueError("skipped workflows must be SkippedWorkflow tuple")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, Evidence) for item in self.evidence
        ):
            raise ValueError("compilation Evidence must be Evidence tuple")
        skill_ids = [item.skill_id for item in self.skills]
        compiled_ids = [item.workflow_id for item in self.skills if isinstance(item, ProjectSkill)]
        skipped_ids = [item.workflow_id for item in self.skipped_workflows]
        if len(skill_ids) != len(set(skill_ids)) or len(compiled_ids) != len(set(compiled_ids)):
            raise ValueError("compiled Skill/workflow IDs must be unique")
        if len(skipped_ids) != len(set(skipped_ids)) or set(compiled_ids) & set(skipped_ids):
            raise ValueError("skipped workflow IDs must be unique and separate")
        if any(not isinstance(key, str) or not key.strip() for key in skipped_ids):
            raise ValueError("skipped workflow IDs must be nonempty")
        if any(not isinstance(item.reason, SkillSkipReason) for item in self.skipped_workflows):
            raise ValueError("skipped workflow reason is invalid")
        if any(
            item.repository_id != self.repository_id
            or item.source_snapshot_id != self.source_snapshot_id
            for item in self.skills
        ):
            raise ValueError("Skill repository/snapshot identity mismatch")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        _canonical_ids(evidence_ids, "compilation Evidence IDs")
        if any(item.repository_id != self.repository_id for item in self.evidence):
            raise ValueError("compilation Evidence repository_id mismatch")
        referenced = {key for skill in self.skills for key in skill.evidence_ids}
        if evidence_ids != tuple(sorted(referenced)):
            raise ValueError("compilation Evidence must match Skill references")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _canonical_ids(values: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(values, tuple)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or values != tuple(sorted(set(values)))
    ):
        raise ValueError(f"{name} must be sorted, unique, nonempty strings")
