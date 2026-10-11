"""Repository-local feedback contracts and deterministic Project DNA governance."""

import json
import subprocess
import sys
from dataclasses import fields, replace

import pytest
from test_skill_compiler import Status
from test_skill_compiler import dna as compiler_dna
from test_skill_compiler import rule as compiler_rule
from test_skill_compiler import workflow as compiler_workflow
from test_verify_plan import complex_route

from skillforge.domain.evidence import Confidence, Support
from skillforge.domain.feedback import (
    FeedbackApplicationResult,
    FeedbackKind,
    FeedbackTargetKind,
    HumanFeedback,
    HumanFeedbackLedger,
    KnowledgeFeedbackTarget,
    RouteFeedbackTarget,
    SkillFeedbackTarget,
    VerifyFeedbackTarget,
)
from skillforge.domain.project_dna import (
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRuleException,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.verification import (
    VerifyCheckKind,
    VerifyCommandSpec,
    VerifyObligationKind,
    VerifyTargetRef,
)
from skillforge.feedback import HumanFeedbackService
from skillforge.repository.identity import repository_identity
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify import IndependentVerifier, VerifyPlanBuilder


def rule_dna(status=Status.CANDIDATE, *, exceptions=()):
    scope = ProjectScope(ProjectScopeKind.FILE, ("src/a.py",))
    source_rule = compiler_rule("r", status, scope, RuleType.TESTING)
    source_rule = replace(
        source_rule,
        support=Support(3, 4),
        confidence=Confidence(0.75, "fixture", Support(3, 4)),
        exceptions=(*source_rule.exceptions, *exceptions),
    )
    return compiler_dna(rules=(source_rule,))


def _repository_scoped_dna(source, repository_id):
    rules = tuple(replace(item, repository_id=repository_id) for item in source.rules)
    workflows = tuple(replace(
        item, repository_id=repository_id,
        draft=replace(item.draft, repository_id=repository_id),
    ) for item in source.workflows)
    anti_patterns = tuple(replace(
        item, repository_id=repository_id,
        draft=replace(item.draft, repository_id=repository_id),
    ) for item in source.anti_patterns)
    evidence = tuple(replace(item, repository_id=repository_id) for item in source.evidence)
    return ProjectDNA(repository_id, rules, workflows, anti_patterns, evidence)


def make_verify_report(tmp_path, source, *, cover_first_obligation=False):
    root = tmp_path / "feedback-verify-repo"
    root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(
        ("git", "-c", "user.name=Feedback Test", "-c", "user.email=feedback@example.invalid",
         "commit", "--allow-empty", "-qm", "base"),
        cwd=root,
        check=True,
    )
    repo_id = repository_identity(root)
    source = _repository_scoped_dna(source, repo_id)
    compiled = SkillCompiler().compile(source)
    registry = SkillRegistryBuilder().build(compiled)
    route = SkillRouter().route(
        registry,
        SkillRoutingRequest(repo_id, "Do name", current_file="src/a.py"),
    )
    plan = VerifyPlanBuilder().build(
        route,
        (VerifyCommandSpec("probe", VerifyCheckKind.CUSTOM,
                           (sys.executable, "-c", "pass")),),
    )
    if cover_first_obligation:
        plan = VerifyPlanBuilder().build(
            route,
            (VerifyCommandSpec(
                "probe", VerifyCheckKind.CUSTOM, (sys.executable, "-c", "pass"),
                covers=(plan.obligations[0].target_ref,),
            ),),
        )
    return source, IndependentVerifier().verify(root, "HEAD", plan)


def test_feedback_id_ledger_id_and_serialization_are_canonical():
    service = HumanFeedbackService()
    dna = compiler_dna()
    first = service.record_knowledge(
        dna, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    same = service.record_knowledge(
        dna, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    second = service.record_knowledge(
        dna, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.INACCURATE, "partial issue"
    )
    assert first.feedback_id == same.feedback_id
    left = HumanFeedbackLedger("repo", (first, second))
    right = HumanFeedbackLedger("repo", (second, first))
    assert left.ledger_id == right.ledger_id
    assert left.records == right.records
    assert json.dumps(left.to_dict(), sort_keys=True, separators=(",", ":")) == json.dumps(
        right.to_dict(), sort_keys=True, separators=(",", ":")
    )


def test_feedback_and_ledger_reject_cross_repository_and_duplicate_ids():
    service = HumanFeedbackService()
    dna = compiler_dna()
    feedback = service.record_knowledge(
        dna, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    with pytest.raises(ValueError, match="repository_id mismatch"):
        HumanFeedbackLedger("another-repo", (feedback,))
    with pytest.raises(ValueError, match="duplicate feedback_id"):
        HumanFeedbackLedger("repo", (feedback, feedback))


def test_knowledge_feedback_requires_an_existing_id_and_exact_snapshot():
    service = HumanFeedbackService()
    dna = compiler_dna()
    with pytest.raises(ValueError, match="unknown PROJECT_RULE"):
        service.record_knowledge(
            dna, FeedbackTargetKind.PROJECT_RULE, "unknown", FeedbackKind.ACCURATE, "reviewed"
        )

    stale = service.record_knowledge(
        dna, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    changed = replace(dna, workflows=(replace(
        dna.workflows[0], status=ProjectKnowledgeStatus.CONFIRMED
    ),))
    with pytest.raises(ValueError, match="stale feedback"):
        service.apply(changed, HumanFeedbackLedger("repo", (stale,)))


def test_skill_feedback_binds_the_actual_compiled_skill_identity():
    service = HumanFeedbackService()
    source = compiler_dna()
    skill = SkillCompiler().compile(source).skills[0]
    feedback = service.record_skill(skill, FeedbackKind.MISSING_STEP, "add deploy check")
    assert feedback.target.skill_id == skill.skill_id
    assert feedback.target.version == skill.version
    assert feedback.target.source_snapshot_id == source.snapshot_id
    with pytest.raises(ValueError, match="Skill version does not match"):
        service.record_skill(
            skill, FeedbackKind.MISSING_STEP, "stale identity", expected_version="wrong-version"
        )
    with pytest.raises(ValueError, match="requires ProjectSkill"):
        service.record_skill(feedback.target, FeedbackKind.WRONG_STEP, "wrong target")


def test_skill_target_cannot_be_constructed_without_its_real_skill():
    skill = SkillCompiler().compile(compiler_dna()).skills[0]
    with pytest.raises(TypeError):
        SkillFeedbackTarget(skill.repository_id, skill.source_snapshot_id, "fake", "fake")
    with pytest.raises(ValueError, match="must match its source ProjectSkill"):
        SkillFeedbackTarget(
            skill.repository_id, skill.source_snapshot_id, skill.skill_id, "fake-version", skill
        )


def test_route_feedback_binds_registry_request_and_selected_skill_versions():
    service = HumanFeedbackService()
    route = complex_route()
    feedback = service.record_route(route, FeedbackKind.WRONG_ROUTE, "selected the wrong Skill")
    target = feedback.target
    assert target.registry_id == route.registry_id
    assert target.routing_request == route.request
    assert tuple((item.skill_id, item.version) for item in target.selected_skills) == tuple(
        sorted((item.skill.skill_id, item.skill.version) for item in route.matches)
    )
    with pytest.raises(ValueError, match="WRONG_ROUTE is not valid"):
        service.record_skill(route.matches[0].skill, FeedbackKind.WRONG_ROUTE, "wrong layer")


def test_route_target_cannot_be_constructed_without_its_real_route():
    route = complex_route()
    selected = tuple(sorted(
        (item.skill.skill_id, item.skill.version) for item in route.matches
    ))
    identities = HumanFeedbackService().record_route(
        route, FeedbackKind.WRONG_ROUTE, "reviewed"
    ).target.selected_skills
    assert tuple((item.skill_id, item.version) for item in identities) == selected
    with pytest.raises(TypeError):
        RouteFeedbackTarget(
            route.repository_id, route.source_snapshot_id, "fake-registry", route.request,
            identities,
        )
    with pytest.raises(ValueError, match="must match its source SkillRouteResult"):
        RouteFeedbackTarget(
            route.repository_id, route.source_snapshot_id, "fake-registry", route.request,
            identities, route,
        )


def test_verify_feedback_binds_actual_report_and_never_rewrites_it(tmp_path):
    service = HumanFeedbackService()
    source, report = make_verify_report(tmp_path, compiler_dna())
    before = report.to_dict()
    feedback = service.record_verify(
        report, FeedbackKind.FALSE_POSITIVE, "human review disagrees", check_id="probe"
    )
    assert feedback.target.plan_id == report.plan_id
    assert feedback.target.patch_snapshot_id == report.patch.snapshot_id
    assert feedback.target.check_id == "probe"
    with pytest.raises(ValueError, match="unknown Verify plan_id"):
        service.record_verify(
            report, FeedbackKind.FALSE_POSITIVE, "unknown plan", check_id="probe",
            expected_plan_id="missing-plan",
        )
    with pytest.raises(ValueError, match="unknown Verify check_id"):
        service.record_verify(
            report, FeedbackKind.FALSE_POSITIVE, "unknown run", check_id="missing"
        )
    unknown_ref = VerifyTargetRef(VerifyObligationKind.RULE, "ghost-skill", "ghost-rule")
    with pytest.raises(ValueError, match="unknown Verify obligation"):
        service.record_verify(
            report, FeedbackKind.FALSE_NEGATIVE, "unknown target", target_ref=unknown_ref
        )
    with pytest.raises(ValueError, match="requires a real VerifyReport"):
        service.record_verify(object(), FeedbackKind.FALSE_POSITIVE, "unbound plan")

    result = service.apply(source, HumanFeedbackLedger(source.repository_id, (feedback,)))
    assert report.to_dict() == before
    assert result.resulting_project_dna is source
    assert result.result_snapshot_id == source.snapshot_id
    assert result.diagnostic_feedback_ids == (feedback.feedback_id,)


def test_verify_target_rejects_fake_report_identity_and_uncovered_check(tmp_path):
    service = HumanFeedbackService()
    _, report = make_verify_report(tmp_path, compiler_dna())
    target_ref = report.obligations[0].target_ref
    assert target_ref not in report.checks[0].covered_refs
    with pytest.raises(TypeError):
        VerifyFeedbackTarget(
            report.repository_id, report.source_snapshot_id, "fake-plan",
            report.patch.snapshot_id, check_id="probe",
        )
    with pytest.raises(ValueError, match="must match its source VerifyReport"):
        VerifyFeedbackTarget(
            report.repository_id, report.source_snapshot_id, "fake-plan",
            report.patch.snapshot_id, report, check_id="probe",
        )
    with pytest.raises(ValueError, match="must match its source VerifyReport"):
        VerifyFeedbackTarget(
            report.repository_id, report.source_snapshot_id, report.plan_id,
            "fake-patch", report, check_id="probe",
        )
    service.record_verify(report, FeedbackKind.FALSE_NEGATIVE, "obligation", target_ref=target_ref)
    service.record_verify(report, FeedbackKind.FALSE_POSITIVE, "check", check_id="probe")
    with pytest.raises(ValueError, match="does not cover"):
        service.record_verify(
            report, FeedbackKind.FALSE_POSITIVE, "uncovered", target_ref=target_ref,
            check_id="probe",
        )
    with pytest.raises(ValueError, match="does not cover"):
        VerifyFeedbackTarget(
            report.repository_id, report.source_snapshot_id, report.plan_id,
            report.patch.snapshot_id, report, target_ref=target_ref, check_id="probe",
        )


def test_verify_target_accepts_check_that_covers_the_obligation(tmp_path):
    _, report = make_verify_report(
        tmp_path, compiler_dna(), cover_first_obligation=True
    )
    target_ref = report.obligations[0].target_ref
    feedback = HumanFeedbackService().record_verify(
        report, FeedbackKind.FALSE_POSITIVE, "covered", target_ref=target_ref,
        check_id="probe",
    )
    assert feedback.target.target_ref == target_ref
    assert feedback.target.check_id == "probe"


def test_diagnostic_signals_never_mutate_dna_and_exception_signal_is_not_governance():
    service = HumanFeedbackService()
    source = rule_dna()
    records = (
        service.record_knowledge(
            source, FeedbackTargetKind.PROJECT_RULE, "r", FeedbackKind.INACCURATE, "needs scope"
        ),
        service.record_knowledge(
            source, FeedbackTargetKind.PROJECT_RULE, "r", FeedbackKind.EXCEPTION,
            "some modules differ",
        ),
    )
    result = service.apply(source, HumanFeedbackLedger("repo", records))
    assert result.resulting_project_dna is source
    assert result.result_snapshot_id == source.snapshot_id
    assert result.applied_feedback_ids == ()
    assert result.diagnostic_feedback_ids == tuple(sorted(item.feedback_id for item in records))
    assert result.resulting_project_dna.rules[0] == source.rules[0]


def test_application_result_requires_source_and_active_ledger_closure():
    service = HumanFeedbackService()
    source = compiler_dna()
    governance = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "confirm"
    )
    diagnostic = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    ledger = HumanFeedbackLedger(source.repository_id, (governance, diagnostic))
    result = service.apply(source, ledger)
    assert result.applied_feedback_ids == (governance.feedback_id,)
    assert result.diagnostic_feedback_ids == (diagnostic.feedback_id,)
    assert "source_dna" not in result.to_dict()
    assert "ledger" not in result.to_dict()

    def construct(source_snapshot_id, applied_ids, diagnostic_ids, *, source_dna=source,
                  source_ledger=ledger):
        return FeedbackApplicationResult(
            source.repository_id, source_snapshot_id, result.result_snapshot_id,
            applied_ids, diagnostic_ids, result.resulting_project_dna,
            source_dna, source_ledger,
        )

    with pytest.raises(TypeError):
        FeedbackApplicationResult(
            source.repository_id, source.snapshot_id, result.result_snapshot_id,
            result.applied_feedback_ids, result.diagnostic_feedback_ids,
            result.resulting_project_dna,
        )
    with pytest.raises(ValueError, match="source Project DNA/Ledger identity mismatch"):
        construct("fake-snapshot", result.applied_feedback_ids, result.diagnostic_feedback_ids)
    with pytest.raises(ValueError, match="close over active Ledger records"):
        construct(source.snapshot_id, ("fake-applied-id",), result.diagnostic_feedback_ids)
    with pytest.raises(ValueError, match="close over active Ledger records"):
        construct(source.snapshot_id, result.applied_feedback_ids, ("fake-diagnostic-id",))
    with pytest.raises(ValueError, match="close over active Ledger records"):
        construct(source.snapshot_id, result.applied_feedback_ids, ())
    with pytest.raises(ValueError, match="close over active Ledger records"):
        construct(source.snapshot_id, (), result.diagnostic_feedback_ids)
    with pytest.raises(ValueError, match="source Project DNA/Ledger identity mismatch"):
        construct(
            source.snapshot_id, result.applied_feedback_ids,
            result.diagnostic_feedback_ids, source_ledger=HumanFeedbackLedger("other", ()),
        )


def test_governance_conflict_requires_supersession():
    service = HumanFeedbackService()
    source = compiler_dna()
    confirmed = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "confirm"
    )
    deprecated = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.DEPRECATE, "deprecate"
    )
    ledger = HumanFeedbackLedger("repo", (confirmed, deprecated))
    with pytest.raises(ValueError, match="conflicting active governance decisions"):
        service.apply(source, ledger)


def test_supersession_is_auditable_and_only_active_feedback_is_applied():
    service = HumanFeedbackService()
    source = rule_dna()
    original = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_RULE, "r", FeedbackKind.CONFIRM, "first decision"
    )
    superseding = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_RULE,
        "r",
        FeedbackKind.MANUAL_OVERRIDE,
        "explicit override",
        supersedes_feedback_id=original.feedback_id,
    )
    ledger = HumanFeedbackLedger("repo", (superseding, original))
    assert len(ledger.records) == 2
    assert ledger.active_records == (superseding,)
    result = service.apply(source, ledger)
    assert result.applied_feedback_ids == (superseding.feedback_id,)
    assert result.resulting_project_dna.rules[0].status == ProjectKnowledgeStatus.MANUAL_OVERRIDE


def test_invalid_supersession_missing_target_and_cycles_are_rejected():
    service = HumanFeedbackService()
    source = compiler_dna()
    record = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "first"
    )
    missing = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_WORKFLOW,
        "wf",
        FeedbackKind.INACCURATE,
        "second",
        supersedes_feedback_id="does-not-exist",
    )
    with pytest.raises(ValueError, match="must exist"):
        HumanFeedbackLedger("repo", (record, missing))

    other_target = KnowledgeFeedbackTarget(
        FeedbackTargetKind.PROJECT_WORKFLOW, "repo", source.snapshot_id, "another-workflow"
    )
    other = HumanFeedback(
        "repo", source.snapshot_id, other_target, FeedbackKind.ACCURATE, "other target"
    )
    wrong_target = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_WORKFLOW,
        "wf",
        FeedbackKind.INACCURATE,
        "points to other target",
        supersedes_feedback_id=other.feedback_id,
    )
    with pytest.raises(ValueError, match="same logical artifact"):
        HumanFeedbackLedger("repo", (other, wrong_target))

    left = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "left"
    )
    right = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.INACCURATE, "right"
    )

    def forged(record, feedback_id, supersedes_feedback_id):
        clone = object.__new__(HumanFeedback)
        for item in fields(HumanFeedback):
            value = getattr(record, item.name)
            if item.name == "feedback_id":
                value = feedback_id
            elif item.name == "supersedes_feedback_id":
                value = supersedes_feedback_id
            object.__setattr__(clone, item.name, value)
        return clone

    cycle_left = forged(left, "cycle-left", "cycle-right")
    cycle_right = forged(right, "cycle-right", "cycle-left")
    with pytest.raises(ValueError, match="supersession cycle"):
        HumanFeedbackLedger("repo", (cycle_left, cycle_right))


def test_manual_override_and_rule_exception_preserve_all_source_metrics():
    service = HumanFeedbackService()
    source = rule_dna()
    original = source.rules[0]
    feedback = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_RULE,
        original.rule_id,
        FeedbackKind.ADD_RULE_EXCEPTION,
        "human exception was approved",
        exception_description="Legacy import omits this check",
    )
    updated = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    rule = updated.resulting_project_dna.rules[0]
    assert rule.status == ProjectKnowledgeStatus.MANUAL_OVERRIDE
    assert rule.exceptions[-1] == ProjectRuleException("Legacy import omits this check", ())
    assert rule.statement == original.statement
    assert rule.scope == original.scope
    assert rule.type == original.type
    assert rule.evidence_ids == original.evidence_ids
    assert rule.support == original.support
    assert rule.confidence == original.confidence
    assert updated.resulting_project_dna.evidence == source.evidence


def test_duplicate_manual_exception_is_idempotent_and_non_rule_target_is_rejected():
    service = HumanFeedbackService()
    existing = ProjectRuleException("same human exception", ())
    source = rule_dna(Status.MANUAL_OVERRIDE, exceptions=(existing,))
    feedback = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_RULE,
        "r",
        FeedbackKind.ADD_RULE_EXCEPTION,
        "repeat the existing exception",
        exception_description=existing.description,
    )
    result = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    assert result.resulting_project_dna.rules[0].exceptions.count(existing) == 1

    workflow_source = compiler_dna()
    with pytest.raises(ValueError, match="MANUAL_OVERRIDE is not valid"):
        service.record_knowledge(
            workflow_source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf",
            FeedbackKind.MANUAL_OVERRIDE, "not a Rule",
        )
    with pytest.raises(ValueError, match="ADD_RULE_EXCEPTION is not valid"):
        service.record_knowledge(
            workflow_source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf",
            FeedbackKind.ADD_RULE_EXCEPTION, "not a Rule",
            exception_description="invalid",
        )


def test_confirm_rejects_deprecated_and_manual_override_statuses():
    service = HumanFeedbackService()
    for status in (ProjectKnowledgeStatus.DEPRECATED, ProjectKnowledgeStatus.MANUAL_OVERRIDE):
        source = compiler_dna(workflows=(compiler_workflow(status),))
        feedback = service.record_knowledge(
            source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "confirm"
        )
        with pytest.raises(ValueError, match="cannot CONFIRM"):
            service.apply(source, HumanFeedbackLedger("repo", (feedback,)))


def test_compiler_sees_confirmed_workflow_and_skips_deprecated_workflow():
    service = HumanFeedbackService()
    source = compiler_dna()
    feedback = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "reviewed"
    )
    result = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    compiled = SkillCompiler().compile(result.resulting_project_dna)
    assert compiled.skills[0].source_status == ProjectKnowledgeStatus.CONFIRMED

    deprecate = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.DEPRECATE, "obsolete"
    )
    deprecated = service.apply(source, HumanFeedbackLedger("repo", (deprecate,)))
    skipped = SkillCompiler().compile(deprecated.resulting_project_dna)
    assert skipped.skills == ()
    assert skipped.skipped_workflows[0].workflow_id == "wf"


@pytest.mark.parametrize("status", (
    ProjectKnowledgeStatus.CONFLICTED,
    ProjectKnowledgeStatus.CONFIRMED,
))
def test_confirm_can_resolve_conflict_and_is_idempotent_for_confirmed(status):
    service = HumanFeedbackService()
    source = compiler_dna(workflows=(compiler_workflow(status),))
    feedback = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "confirmed"
    )
    result = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    assert result.resulting_project_dna.workflows[0].status == ProjectKnowledgeStatus.CONFIRMED
    if status == ProjectKnowledgeStatus.CONFIRMED:
        assert result.result_snapshot_id == source.snapshot_id


def test_mark_conflicted_changes_only_governance_status():
    service = HumanFeedbackService()
    source = compiler_dna()
    original = source.workflows[0]
    feedback = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_WORKFLOW,
        "wf",
        FeedbackKind.MARK_CONFLICTED,
        "two maintainers disagree",
    )
    result = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    updated = result.resulting_project_dna.workflows[0]
    assert updated.status == ProjectKnowledgeStatus.CONFLICTED
    assert updated.draft == original.draft
    assert result.resulting_project_dna.evidence == source.evidence


def test_compiler_maps_manual_rule_override_to_required_without_changing_metrics():
    service = HumanFeedbackService()
    source = rule_dna()
    original = source.rules[0]
    feedback = service.record_knowledge(
        source,
        FeedbackTargetKind.PROJECT_RULE,
        "r",
        FeedbackKind.MANUAL_OVERRIDE,
        "required by maintainers",
    )
    result = service.apply(source, HumanFeedbackLedger("repo", (feedback,)))
    updated_rule = result.resulting_project_dna.rules[0]
    assert updated_rule.support == original.support
    assert updated_rule.confidence == original.confidence
    assert updated_rule.evidence_ids == original.evidence_ids
    compiled = SkillCompiler().compile(result.resulting_project_dna)
    constraint = next(item for item in compiled.skills[0].constraints if item.rule_id == "r")
    assert constraint.enforcement.value == "required"
    assert constraint.source_support == original.support
    assert constraint.source_confidence == original.confidence


def test_governance_actions_apply_deterministically_independent_of_record_order():
    service = HumanFeedbackService()
    source = rule_dna()
    confirm = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.CONFIRM, "confirm"
    )
    override = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_RULE, "r", FeedbackKind.MANUAL_OVERRIDE, "override"
    )
    left = service.apply(source, HumanFeedbackLedger("repo", (confirm, override)))
    right = service.apply(source, HumanFeedbackLedger("repo", (override, confirm)))
    assert left.to_dict() == right.to_dict()


def test_feedback_diagnostic_kinds_are_partitioned_by_artifact(tmp_path):
    service = HumanFeedbackService()
    source, report = make_verify_report(tmp_path, compiler_dna())
    compilation = SkillCompiler().compile(source)
    skill = compilation.skills[0]
    registry = SkillRegistryBuilder().build(compilation)
    route = SkillRouter().route(
        registry,
        SkillRoutingRequest(source.repository_id, "Do name", current_file="src/a.py"),
    )
    records = (
        service.record_knowledge(
            source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.PREFERRED_NEW_PATTERN,
            "prefer a smaller workflow",
        ),
        service.record_skill(skill, FeedbackKind.TOO_COARSE, "separate deployment step"),
        service.record_skill(skill, FeedbackKind.MISSING_STEP, "add a migration check"),
        service.record_skill(skill, FeedbackKind.WRONG_STEP, "replace the validation step"),
        service.record_skill(skill, FeedbackKind.TOO_FINE, "combine adjacent checks"),
        service.record_route(route, FeedbackKind.MISSING_SKILL, "no migration Skill"),
        service.record_route(route, FeedbackKind.WRONG_ROUTE, "use the migration Skill"),
        service.record_verify(
            report, FeedbackKind.PERMANENT_WHITELIST, "human policy exception", check_id="probe"
        ),
        service.record_verify(
            report, FeedbackKind.FALSE_POSITIVE, "the check is a false positive", check_id="probe"
        ),
        service.record_verify(
            report, FeedbackKind.FALSE_NEGATIVE, "the check missed a regression", check_id="probe"
        ),
        service.record_verify(
            report, FeedbackKind.MODULE_EXCEPTION, "one module has an exception", check_id="probe"
        ),
    )
    result = service.apply(source, HumanFeedbackLedger(source.repository_id, records))
    assert result.applied_feedback_ids == ()
    assert result.resulting_project_dna is source
    assert result.diagnostic_feedback_ids == tuple(sorted(item.feedback_id for item in records))


def test_ledger_append_keeps_history_and_rejects_wrong_repository():
    service = HumanFeedbackService()
    source = compiler_dna()
    feedback = service.record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, "wf", FeedbackKind.ACCURATE, "reviewed"
    )
    empty = HumanFeedbackLedger("repo", ())
    appended = service.append(empty, feedback)
    assert empty.records == ()
    assert appended.records == (feedback,)
    with pytest.raises(ValueError, match="repository_id mismatch"):
        service.append(HumanFeedbackLedger("other", ()), feedback)


def test_knowledge_target_kind_and_feedback_kind_must_match():
    target = KnowledgeFeedbackTarget(
        FeedbackTargetKind.PROJECT_RULE, "repo", "snapshot", "rule"
    )
    with pytest.raises(ValueError, match="not valid"):
        HumanFeedback(
            "repo", "snapshot", target, FeedbackKind.MISSING_STEP, "wrong signal group"
        )
