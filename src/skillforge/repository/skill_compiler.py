"""Pure compilation of Project DNA into workflow Skills and rule guidance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    ProjectWorkflow,
    RuleType,
)
from skillforge.domain.semantic_antipattern import SemanticAntiPatternScopeKind
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralResponse,
    SemanticStructuralRuleResult,
)
from skillforge.domain.semantic_tendency import TENDENCY_METHOD, SemanticTendencyResult
from skillforge.domain.semantic_workflow import SemanticWorkflowScopeKind
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import (
    ProjectSkill,
    RuleGuidanceSkill,
    SkillCompilationResult,
    SkillCompletionCondition,
    SkillConstraint,
    SkillConstraintEnforcement,
    SkillException,
    SkillProcedure,
    SkillRisk,
    SkillSkipReason,
    SkillStep,
    SkillStepKind,
    SkillTrigger,
    SkippedWorkflow,
)
from skillforge.domain.structural_rules import DELEGATION_METHOD, StructuralRuleMiningResult
from skillforge.repository.delegation_guidance import delegation_decision_steps
from skillforge.repository.semantic_structural_rules import (
    build_semantic_structural_request,
    validate_semantic_structural_response,
)
from skillforge.repository.semantic_tendencies import validate_semantic_tendency_response
from skillforge.repository.testing_guidance import companion_steps, testing_decision_steps

SCHEMA = "skill-compiler-v1"
GUIDANCE_SCHEMA = "rule-guidance-compiler-v1"
EXECUTABLE_SCHEMA = "executable-skill-compiler-v1"


def _procedure(
    dna: ProjectDNA, scope: ProjectScope, steps: tuple[SkillStep, ...],
    primary: SkillConstraint | None = None,
    *, ownership_guard: bool = False,
) -> SkillProcedure:
    # Exact scopes or repository-wide knowledge compose without widening applicability.
    relevant = tuple(rule for rule in dna.rules if rule.scope == scope
                     or rule.scope.kind == ProjectScopeKind.REPOSITORY)
    excluded = tuple(sorted(rule.rule_id for rule in relevant
                            if rule.status == ProjectKnowledgeStatus.CONFLICTED))
    constraints = tuple(item for item in _constraints(relevant, scope)
                        if item.evidence_ids and item.source_status not in {
                            ProjectKnowledgeStatus.CONFLICTED, ProjectKnowledgeStatus.DEPRECATED,
                        } and (primary is None or item.rule_id != primary.rule_id))
    verify = [SkillStep(
        "verify-behavior", "Run the repository's existing checks relevant to the changed behavior; "
        "record commands, results, and checks that could not be run. Use those same checks "
        "to assess the concrete conditions, outcomes, and exceptions in applicable steps; "
        "do not infer unstated behavior or treat missing coverage as proof.",
        SkillStepKind.REQUIRED,
        None, "Generic independent verification; no project command is inferred.", (),
    )]
    for rule in constraints:
        if rule.rule_type == RuleType.TESTING:
            verify.append(SkillStep(
                f"verify-{rule.rule_id}", f"Check the applicable verification knowledge: "
                f"{rule.statement} Respect its documented exceptions and source authority.",
                SkillStepKind.CONDITIONAL, f"The {rule.rule_id} scope and exceptions apply",
                "Project DNA testing knowledge", rule.evidence_ids,
            ))
    if ownership_guard:
        verify.append(SkillStep(
            "verify-ownership", "If a new sibling module was created, recheck the existing "
            "owner candidates, their functions, callers, and tests: does any existing module "
            "already own the same core responsibility? Confirm that the new responsibility "
            "is genuinely distinct, rather than splitting files because implementation "
            "complexity increased. Recheck that historical tendency remains advisory and "
            "conditional, its scope still matches, and no contrary project evidence, stronger "
            "Project Rule, Workflow, or exception applies. Record the candidate exclusions, "
            "distinct responsibility, and supporting historical Evidence; if this justification "
            "cannot be completed, preserve existing ownership and revise the placement. "
            "Functional tests passing alone cannot prove the ownership decision correct.",
            SkillStepKind.CONDITIONAL, "A new sibling module was created",
            "Generic independent ownership self-check.", (),
        ))
    verify.append(SkillStep(
        "verify-placement", "Review the final diff against the chosen ownership, applicable "
        "constraints, and each executed step. Record deviations and the evidence for them; "
        "passing functional tests alone does not establish structural compliance.",
        SkillStepKind.REQUIRED, None, "Generic independent completion review.", (),
    ))
    return SkillProcedure(steps, constraints, tuple(verify), excluded)


def _decision_steps(primary: SkillConstraint, *, child_modules: bool) -> tuple[SkillStep, ...]:
    if not child_modules:
        return (
            SkillStep("inspect-relationships", "Inspect the current units and dependency "
                      "relationships covered by the source scope; compare them with the cited "
                      "structural pattern before editing.", SkillStepKind.REQUIRED, None,
                      "Generic current-state inspection.", ()),
            SkillStep("apply-local-pattern", "Prefer the cited architectural relationship "
                      f"when it fits the requested change: {primary.statement} Check source "
                      "exceptions and contrary evidence; record the chosen relationship and "
                      "any reason for a different choice.", SkillStepKind.CONDITIONAL,
                      "The changed relationships fall within the observed scope, the pattern "
                      "still applies, and no stronger rule or exception conflicts",
                      f"Conditional preference from {primary.rule_id}", primary.evidence_ids),
        )
    pattern = "prefer a distinct direct child Python module within the applicable scope"
    return (
        SkillStep("inspect-ownership", "Inspect current units, responsibilities, callers, and "
                  "functions and nearby tests in the applicable repository scope. Identify "
                  "existing owner candidates for the requested core responsibility before "
                  "considering historical placement.", SkillStepKind.REQUIRED, None,
                  "Generic current-state inspection.", ()),
        SkillStep("compare-ownership", "Compare the requested behavior with existing unit "
                  "responsibilities. If an existing unit owns the same core responsibility, "
                  "extend that unit by default, even when the request adds computation steps, "
                  "parameters, boundary conditions, or implementation complexity. A new internal "
                  "algorithm, extra computation, a helper, or more logic does not by itself "
                  "establish a distinct capability. Only after establishing no clear existing "
                  "owner, determine whether the responsibility is genuinely distinct. If "
                  "ownership remains uncertain, preserve existing ownership rather than "
                  "defaulting to historical sibling placement; record the decision before editing.",
                  SkillStepKind.REQUIRED, None,
                  "Generic ownership decision; proximity alone does not establish ownership.", ()),
        SkillStep("apply-local-pattern", f"Compare the current structure with the cited "
                  f"observation. Use historical structural tendency only as a tie-breaker / "
                  f"conditional guidance, never a default new-module rule. If all conditions "
                  f"hold, {pattern}. Before choosing NEW_SIBLING_MODULE, record which existing "
                  "owner candidates were checked, why none owns this core responsibility, why "
                  "the responsibility is genuinely distinct, and which historical Evidence "
                  "supports sibling placement. If this justification cannot be completed, "
                  "preserve existing ownership. Choose names from the "
                  "actual responsibility and current conventions. If contrary evidence or "
                  "an exception applies, record it and choose the supported alternative.",
                  SkillStepKind.CONDITIONAL,
                  "There is no clear existing owner, the responsibility is genuinely distinct, "
                  "the observed "
                  "pattern still fits the current scope, "
                  "and no stronger Project Rule, Workflow, exception, or contrary project "
                  "evidence conflicts; all candidate exclusions and placement Evidence "
                  "can be justified",
                  f"Conditional preference from {primary.rule_id}: {primary.statement}",
                  primary.evidence_ids),
        SkillStep("apply-constraints", "Implement the ownership decision while checking the "
                  "Evidence-backed project constraints below. Treat candidate guidance as "
                  "conditional and preserve documented exceptions.", SkillStepKind.REQUIRED,
                  None, "Generic implementation discipline.", ()),
    )


def _ids(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({value for group in groups for value in group}))


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope(workflow: ProjectWorkflow) -> ProjectScope:
    source = workflow.draft.scope
    if not isinstance(source.kind, SemanticWorkflowScopeKind):
        raise ValueError("invalid workflow scope kind")
    if not isinstance(source.selectors, tuple):
        raise ValueError("invalid workflow scope selectors")
    if source.kind == SemanticWorkflowScopeKind.UNKNOWN:
        if source.selectors:
            raise ValueError("UNKNOWN workflow scope cannot have selectors")
        raise ValueError("UNKNOWN workflow scope cannot be compiled")
    return ProjectScope(ProjectScopeKind(source.kind.value), source.selectors)


def _validate_workflow(workflow: ProjectWorkflow, evidence_ids: set[str]) -> None:
    draft = workflow.draft
    steps = draft.ordered_steps
    step_ids = [item.step_id for item in steps]
    if any(not isinstance(key, str) or not key.strip() for key in step_ids):
        raise ValueError("workflow step IDs must be nonempty")
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("duplicate ordered workflow step ID")
    classified = (*draft.required_steps, *draft.optional_steps, *draft.conditional_steps)
    classification_ids = [item.step_id for item in classified]
    if len(classification_ids) != len(set(classification_ids)):
        raise ValueError("duplicate workflow step classification")
    if set(classification_ids) != set(step_ids):
        raise ValueError("workflow classifications must cover exactly the ordered steps")
    if any(item.condition is not None for item in (*draft.required_steps, *draft.optional_steps)):
        raise ValueError("required and optional steps cannot have conditions")
    if any(
        not isinstance(item.condition, str) or not item.condition.strip()
        for item in draft.conditional_steps
    ):
        raise ValueError("conditional steps require a nonempty condition")
    _scope(workflow)
    nested = _ids(
        draft.name_evidence_ids,
        draft.intent_evidence_ids,
        draft.scope.evidence_ids,
        *(item.evidence_ids for item in (*draft.roles, *steps, *classified, *draft.exceptions)),
    )
    if not set(nested) <= set(draft.evidence_ids):
        raise ValueError("workflow nested Evidence is outside its aggregate")
    if not set(draft.evidence_ids) <= evidence_ids:
        raise ValueError("workflow references missing Project DNA Evidence")


def _applicable(rule_scope: ProjectScope, skill_scope: ProjectScope) -> bool:
    if rule_scope.kind == ProjectScopeKind.REPOSITORY:
        return True
    return rule_scope.kind == skill_scope.kind and bool(
        set(rule_scope.selectors) & set(skill_scope.selectors)
    )


def _risk_scope(pattern: ProjectAntiPattern) -> ProjectScope | None:
    source = pattern.draft.scope
    if source.kind == SemanticAntiPatternScopeKind.UNKNOWN:
        return None
    if source.kind not in {
        SemanticAntiPatternScopeKind.REPOSITORY,
        SemanticAntiPatternScopeKind.FILE,
    }:
        raise ValueError("invalid Anti-pattern scope kind")
    return ProjectScope(ProjectScopeKind(source.kind.value.lower()), source.selectors)


def _constraints(
    rules: tuple[ProjectRule, ...], scope: ProjectScope
) -> tuple[SkillConstraint, ...]:
    result = []
    for rule in sorted(rules, key=lambda item: item.rule_id):
        if rule.status == ProjectKnowledgeStatus.DEPRECATED or not _applicable(rule.scope, scope):
            continue
        enforcement = (
            SkillConstraintEnforcement.REQUIRED
            if rule.status
            in {ProjectKnowledgeStatus.CONFIRMED, ProjectKnowledgeStatus.MANUAL_OVERRIDE}
            else SkillConstraintEnforcement.ADVISORY
        )
        result.append(
            SkillConstraint(
                rule.rule_id,
                rule.type,
                rule.statement,
                rule.scope,
                rule.status,
                enforcement,
                rule.exceptions,
                rule.evidence_ids,
                rule.support,
                rule.confidence,
            )
        )
    return tuple(result)


def _risks(patterns: tuple[ProjectAntiPattern, ...], scope: ProjectScope) -> tuple[SkillRisk, ...]:
    result = []
    for pattern in sorted(patterns, key=lambda item: item.anti_pattern_id):
        if pattern.status == ProjectKnowledgeStatus.DEPRECATED:
            continue
        risk_scope = _risk_scope(pattern)
        if risk_scope is None or not (
            risk_scope.kind == ProjectScopeKind.REPOSITORY
            or (
                risk_scope.kind == scope.kind == ProjectScopeKind.FILE
                and set(risk_scope.selectors) & set(scope.selectors)
            )
        ):
            continue
        draft = pattern.draft
        if draft.confidence.support is not None:
            raise ValueError("Anti-pattern Confidence must not contain Support")
        result.append(
            SkillRisk(
                pattern.anti_pattern_id,
                draft.name,
                draft.description,
                draft.avoid_behavior,
                risk_scope,
                pattern.status,
                draft.confidence,
                tuple(
                    SkillException(item.description, item.evidence_ids) for item in draft.exceptions
                ),
                draft.evidence_ids,
            )
        )
    return tuple(result)


def _compile_one(
    dna: ProjectDNA, workflow: ProjectWorkflow, evidence_ids: set[str]
) -> ProjectSkill:
    _validate_workflow(workflow, evidence_ids)
    draft = workflow.draft
    scope = _scope(workflow)
    trigger = SkillTrigger(
        draft.intent, scope, _ids(draft.intent_evidence_ids, draft.scope.evidence_ids)
    )
    classifications = {
        item.step_id: (kind, item)
        for kind, collection in (
            (SkillStepKind.REQUIRED, draft.required_steps),
            (SkillStepKind.OPTIONAL, draft.optional_steps),
            (SkillStepKind.CONDITIONAL, draft.conditional_steps),
        )
        for item in collection
    }
    steps = tuple(
        SkillStep(
            step.step_id,
            step.action,
            classifications[step.step_id][0],
            classifications[step.step_id][1].condition,
            classifications[step.step_id][1].reason,
            _ids(step.evidence_ids, classifications[step.step_id][1].evidence_ids),
        )
        for step in draft.ordered_steps
    )
    if not steps:
        raise ValueError("workflow has no executable steps")
    completion = tuple(
        SkillCompletionCondition(step.step_id, step.condition, step.evidence_ids)
        for step in steps
        if step.kind != SkillStepKind.OPTIONAL
    )
    exceptions = tuple(
        SkillException(item.description, item.evidence_ids) for item in draft.exceptions
    )
    constraints = _constraints(dna.rules, scope)
    risks = _risks(dna.anti_patterns, scope)
    verify_ids = tuple(
        item.rule_id
        for item in constraints
        if item.enforcement == SkillConstraintEnforcement.REQUIRED
        and item.rule_type in {RuleType.TESTING, RuleType.ARCHITECTURE}
    )
    closure = _ids(
        trigger.evidence_ids,
        draft.evidence_ids,
        *(
            item.evidence_ids
            for collection in (steps, exceptions, completion, constraints, risks)
            for item in collection
        ),
        *(exc.evidence_ids for constraint in constraints for exc in constraint.exceptions),
        *(exception.evidence_ids for risk in risks for exception in risk.exceptions),
    )
    if not set(closure) <= evidence_ids:
        raise ValueError("Skill references missing Project DNA Evidence")
    identity = _fingerprint((SCHEMA, dna.repository_id, workflow.workflow_id))
    semantic = (
        SCHEMA,
        dna.repository_id,
        workflow.workflow_id,
        draft.name,
        workflow.status,
        draft.support,
        draft.confidence,
        trigger,
        steps,
        constraints,
        risks,
        exceptions,
        completion,
        verify_ids,
        closure,
    )
    skill = ProjectSkill(
        f"skill-{identity[:24]}",
        dna.repository_id,
        dna.snapshot_id,
        _fingerprint(semantic),
        draft.name,
        trigger,
        (),
        steps,
        workflow.workflow_id,
        workflow.status,
        draft.support,
        draft.confidence,
        draft.evidence_ids,
        constraints,
        risks,
        exceptions,
        completion,
        verify_ids,
        closure,
    )
    procedure = _procedure(dna, scope, steps)
    return replace(skill, procedure=procedure,
                   version=_fingerprint((EXECUTABLE_SCHEMA, skill.version, procedure)))


def _compile_guidance(
    dna: ProjectDNA, semantic: SemanticStructuralRuleResult,
) -> tuple[RuleGuidanceSkill, ...]:
    if semantic.repository_id != dna.repository_id:
        raise ValueError("semantic acceptance repository mismatch")
    rules = {rule.rule_id: rule for rule in dna.rules}
    evidence = {item.evidence_id: item for item in dna.evidence}
    semantic_evidence = {item.evidence_id: item for item in semantic.evidence}
    result = []
    for accepted in semantic.accepted:
        rule = accepted.candidate.rule
        current = rules.get(rule.rule_id)
        if (current is not None
                and current.status == ProjectKnowledgeStatus.DEPRECATED
                and replace(current, status=ProjectKnowledgeStatus.CANDIDATE) == rule):
            continue
        if current != rule:
            raise ValueError("semantic acceptance does not match Project DNA rule")
        if any(semantic_evidence.get(key) != evidence.get(key)
               for key in rule.evidence_ids):
            raise ValueError("semantic acceptance Evidence differs from Project DNA")
        # Rebuild the Phase 2 request and rerun its deterministic evidence gate.
        mined = StructuralRuleMiningResult(
            dna.repository_id, (accepted.candidate,),
            tuple(semantic_evidence[key] for key in rule.evidence_ids),
        )
        request = build_semantic_structural_request(mined, accepted.candidate)
        response = SemanticStructuralResponse(
            dna.repository_id, accepted.candidate_id, accepted.decision,
            accepted.reason, accepted.decision_evidence_ids,
            accepted.candidate.source_package, accepted.candidate.target_package,
            accepted.scope_modules, accepted.scope_evidence_ids, (),
        )
        if validate_semantic_structural_response(request, response, accepted.candidate) != accepted:
            raise ValueError("semantic acceptance fails deterministic gate")
        constraint = SkillConstraint(
            rule.rule_id, rule.type, rule.statement, rule.scope, rule.status,
            SkillConstraintEnforcement.ADVISORY, rule.exceptions,
            rule.evidence_ids, rule.support, rule.confidence,
        )
        identity = _fingerprint((GUIDANCE_SCHEMA, dna.repository_id, rule.rule_id))
        version = _fingerprint((GUIDANCE_SCHEMA, rule, accepted.candidate_id,
                                accepted.method, accepted.reason,
                                accepted.decision_evidence_ids,
                                accepted.scope_evidence_ids))
        steps = _decision_steps(constraint, child_modules=False)
        if accepted.candidate.method == DELEGATION_METHOD:
            steps = delegation_decision_steps(accepted.candidate)
        elif accepted.candidate.test_linkages:
            steps = testing_decision_steps(accepted.candidate)
        # Compose accepted executable knowledge at the same scope and family.
        # This preserves each source rule and the router's existing ranking.
        steps += companion_steps(accepted.candidate, semantic, rules)
        procedure = _procedure(dna, rule.scope, steps, constraint)
        result.append(RuleGuidanceSkill(
            f"skill-{identity[:24]}", dna.repository_id, dna.snapshot_id,
            _fingerprint((EXECUTABLE_SCHEMA, version, procedure)),
            rule.statement, SkillTrigger(rule.statement, rule.scope, rule.evidence_ids),
            constraint, rule.rule_id, rule.type, rule.status, rule.scope,
            rule.support, rule.confidence, accepted.candidate_id,
            accepted.method, accepted.reason, accepted.decision_evidence_ids,
            _ids(rule.evidence_ids, procedure.evidence_ids), procedure,
        ))
    return tuple(result)


class SkillCompiler:
    """No repository reads, model calls, parameter guesses, or Verify execution."""

    def compile(
        self, dna: ProjectDNA,
        semantic_structural: SemanticStructuralRuleResult | None = None,
        semantic_tendencies: SemanticTendencyResult | None = None,
    ) -> SkillCompilationResult:
        if not isinstance(dna, ProjectDNA):
            raise TypeError("SkillCompiler requires ProjectDNA")
        evidence = {}
        for item in dna.evidence:
            if item.evidence_id in evidence:
                if evidence[item.evidence_id] != item:
                    raise ValueError(f"conflicting Evidence ID: {item.evidence_id}")
                raise ValueError(f"duplicate Evidence ID: {item.evidence_id}")
            evidence[item.evidence_id] = item
        if any(item.repository_id != dna.repository_id for item in dna.evidence):
            raise ValueError("Evidence repository_id mismatch")
        for name, items, key in (
            ("workflow", dna.workflows, "workflow_id"),
            ("rule", dna.rules, "rule_id"),
            ("Anti-pattern", dna.anti_patterns, "anti_pattern_id"),
        ):
            ids = [getattr(item, key) for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} ID")
            if any(item.repository_id != dna.repository_id for item in items):
                raise ValueError(f"{name} repository_id mismatch")
            if any(
                not set(item.evidence_ids if name == "rule" else item.draft.evidence_ids)
                <= evidence.keys()
                for item in items
            ):
                raise ValueError(f"{name} references missing Evidence")
        skills = []
        skipped = []
        for workflow in sorted(dna.workflows, key=lambda item: item.workflow_id):
            if workflow.status in {
                ProjectKnowledgeStatus.DEPRECATED,
                ProjectKnowledgeStatus.CONFLICTED,
            }:
                skipped.append(
                    SkippedWorkflow(workflow.workflow_id, SkillSkipReason(workflow.status.name))
                )
            elif workflow.draft.scope.kind == SemanticWorkflowScopeKind.UNKNOWN:
                if workflow.draft.scope.selectors:
                    raise ValueError("UNKNOWN workflow scope cannot have selectors")
                skipped.append(SkippedWorkflow(workflow.workflow_id, SkillSkipReason.UNKNOWN_SCOPE))
            else:
                skills.append(_compile_one(dna, workflow, set(evidence)))
        if semantic_structural is not None:
            if not isinstance(semantic_structural, SemanticStructuralRuleResult):
                raise TypeError("semantic_structural requires audited semantic result")
            skills.extend(_compile_guidance(dna, semantic_structural))
        if semantic_tendencies is not None:
            if semantic_tendencies.repository_id != dna.repository_id:
                raise ValueError("tendency compilation repository mismatch")
            rules = {item.rule_id: item for item in dna.rules}
            audited = {item.evidence_id: item for item in semantic_tendencies.evidence}
            for accepted in semantic_tendencies.accepted:
                rule = accepted.rule
                validated = validate_semantic_tendency_response(accepted.request, accepted.response)
                if validated != accepted:
                    raise ValueError("tendency acceptance fails deterministic gate")
                if rules.get(rule.rule_id) != rule or any(
                    audited.get(key) != evidence.get(key) for key in rule.evidence_ids
                ):
                    raise ValueError("tendency acceptance differs from DNA")
                constraint = SkillConstraint(
                    rule.rule_id, rule.type, rule.statement, rule.scope, rule.status,
                    SkillConstraintEnforcement.ADVISORY, rule.exceptions,
                    rule.evidence_ids, rule.support, rule.confidence,
                )
                identity = _fingerprint((GUIDANCE_SCHEMA, dna.repository_id, rule.rule_id))
                procedure = _procedure(dna, rule.scope,
                                       _decision_steps(constraint, child_modules=True), constraint,
                                       ownership_guard=True)
                skills.append(RuleGuidanceSkill(
                    f"skill-{identity[:24]}", dna.repository_id, dna.snapshot_id,
                    _fingerprint((EXECUTABLE_SCHEMA, accepted, procedure)), rule.statement,
                    SkillTrigger(rule.statement, rule.scope, rule.evidence_ids), constraint,
                    rule.rule_id, rule.type, rule.status, rule.scope, rule.support, rule.confidence,
                    accepted.response.candidate_id, TENDENCY_METHOD, accepted.response.reason,
                    accepted.response.evidence_ids,
                    _ids(rule.evidence_ids, procedure.evidence_ids), procedure,
                ))
        used_ids = _ids(*(item.evidence_ids for item in skills))
        return SkillCompilationResult(
            dna.repository_id,
            dna.snapshot_id,
            tuple(skills),
            tuple(skipped),
            tuple(evidence[key] for key in used_ids),
        )
