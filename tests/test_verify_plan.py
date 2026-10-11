"""Selected Skill obligations and exact caller-declared coverage."""

from dataclasses import replace

import pytest
from test_skill_registry import evidence, skill

from skillforge.domain.evidence import Confidence
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.skills import (
    SkillCompilationResult,
    SkillCompletionCondition,
    SkillConstraint,
    SkillConstraintEnforcement,
    SkillRisk,
    SkillStep,
    SkillStepKind,
)
from skillforge.domain.verification import (
    VerifyCheckKind,
    VerifyCommandSpec,
    VerifyObligationKind,
    VerifyTargetRef,
)
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify import VerifyPlanBuilder


def complex_route(repo_id="repo"):
    base = skill("selected", repo=repo_id)
    scope = ProjectScope(ProjectScopeKind.REPOSITORY)
    steps = (
        SkillStep("required", "Run migration", SkillStepKind.REQUIRED,
                  None, "reason", ("step-ev",)),
        SkillStep("conditional", "Check adapter", SkillStepKind.CONDITIONAL,
                  "when adapter exists", "reason", ("step-ev",)),
        SkillStep("optional", "Update docs", SkillStepKind.OPTIONAL, None, "reason", ("step-ev",)),
    )
    constraint = SkillConstraint(
        "rule", RuleType.TESTING, "Tests pass", scope, ProjectKnowledgeStatus.CONFIRMED,
        SkillConstraintEnforcement.REQUIRED, (), ("rule-ev",),
    )
    risk = SkillRisk("risk", "Risk", "Review old failure", "Avoid it", scope,
                     ProjectKnowledgeStatus.CANDIDATE, Confidence(0.4, "fixture"),
                     (), ("risk-ev",))
    selected = replace(base, steps=steps, constraints=(constraint,), risks=(risk,),
                       completion_conditions=(
                           SkillCompletionCondition("required", None, ("step-ev",)),
                           SkillCompletionCondition("conditional", "when adapter exists",
                                                    ("step-ev",)),
                       ), verify_rule_ids=("rule",),
                       evidence_ids=("risk-ev", "rule-ev", "selected", "step-ev"))
    compiled = SkillCompilationResult(repo_id, "snap", (selected,), (), tuple(
        evidence(key, repo_id) for key in selected.evidence_ids
    ))
    registry = SkillRegistryBuilder().build(compiled)
    return SkillRouter().route(registry, SkillRoutingRequest(repo_id, "create route"))


def test_obligation_derivation_and_evidence_closure():
    route = complex_route()
    plan = VerifyPlanBuilder().build(route)
    refs = {(item.target_ref.kind, item.target_ref.ref_id) for item in plan.obligations}
    assert refs == {(VerifyObligationKind.RULE, "rule"),
                    (VerifyObligationKind.WORKFLOW_STEP, "required"),
                    (VerifyObligationKind.WORKFLOW_STEP, "conditional"),
                    (VerifyObligationKind.RISK, "risk")}
    assert tuple(item.evidence_id for item in plan.evidence) == (
        "risk-ev", "rule-ev", "step-ev"
    )
    assert VerifyPlanBuilder().build(route).plan_id == plan.plan_id
    assert all(item.source_status for item in plan.obligations)
    with pytest.raises(ValueError, match="Evidence closure"):
        replace(plan, evidence=route.evidence)


def test_explicit_coverage_only_and_bad_refs():
    route = complex_route()
    plan = VerifyPlanBuilder().build(route)
    ref = plan.obligations[0].target_ref
    check = VerifyCommandSpec("test", VerifyCheckKind.TEST, ("python", "-V"), covers=(ref,))
    assert VerifyPlanBuilder().build(route, (check,)).checks[0].covers == (ref,)
    for bad in (
        VerifyTargetRef(ref.kind, "unknown", ref.ref_id),
        VerifyTargetRef(VerifyObligationKind.RULE, "selected", "unknown"),
        VerifyTargetRef(VerifyObligationKind.WORKFLOW_STEP, "selected", "unknown"),
        VerifyTargetRef(VerifyObligationKind.RISK, "selected", "unknown"),
    ):
        with pytest.raises(ValueError, match="unknown obligation"):
            VerifyPlanBuilder().build(route, (replace(check, covers=(bad,)),))
    with pytest.raises(ValueError):
        replace(check, covers=(ref, ref))
    with pytest.raises(ValueError):
        VerifyPlanBuilder().build(route, (check, check))


def test_with_checks_preserves_the_frozen_seed_without_a_route():
    seed = VerifyPlanBuilder().build(complex_route())
    ref = seed.obligations[0].target_ref
    check = VerifyCommandSpec("test", VerifyCheckKind.TEST, ("python", "-V"), covers=(ref,))
    plan = VerifyPlanBuilder().with_checks(seed, (check,))
    assert plan.repository_id == seed.repository_id
    assert plan.registry_id == seed.registry_id
    assert plan.source_snapshot_id == seed.source_snapshot_id
    assert plan.selected_skills == seed.selected_skills
    assert plan.obligations == seed.obligations
    assert plan.evidence == seed.evidence
    assert plan.checks == (check,)
    assert plan.plan_id != seed.plan_id
    unknown = VerifyTargetRef(ref.kind, "unknown", ref.ref_id)
    with pytest.raises(ValueError, match="unknown obligation"):
        VerifyPlanBuilder().with_checks(seed, (replace(check, covers=(unknown,)),))
    with pytest.raises(ValueError, match="must not contain checks"):
        VerifyPlanBuilder().with_checks(plan, ())


@pytest.mark.parametrize("cwd", ("../outside", "/tmp", "C:/outside", "a//b", "a/./b", "a\\b"))
def test_command_contract_rejects_unsafe_cwd(cwd):
    with pytest.raises(ValueError):
        VerifyCommandSpec("x", VerifyCheckKind.CUSTOM, ("python",), cwd=cwd)


def test_command_contract_rejects_bad_argv_and_timeout():
    for argv in ("python -V", (), ("",), ("python", 4)):
        with pytest.raises(ValueError):
            VerifyCommandSpec("x", VerifyCheckKind.CUSTOM, argv)
    for timeout in (0, 3601, True):
        with pytest.raises(ValueError):
            VerifyCommandSpec("x", VerifyCheckKind.CUSTOM, ("python",),
                              timeout_seconds=timeout)
    for argv in (("sh", "-c", "echo unsafe"),
                 ("bash", "-lc", "echo unsafe"),
                 ("cmd.exe", "/c", "echo unsafe"),
                 ("pwsh", "-Command", "echo unsafe")):
        with pytest.raises(ValueError, match="shell command wrappers"):
            VerifyCommandSpec("x", VerifyCheckKind.CUSTOM, argv)
