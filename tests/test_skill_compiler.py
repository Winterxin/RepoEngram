"""Integration and adversarial checks for the pure Project DNA compiler."""

from dataclasses import replace

import pytest

from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectRule,
    ProjectRuleException,
    ProjectScope,
    ProjectScopeKind,
    ProjectWorkflow,
    RuleType,
)
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus as Status,
)
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDraft,
    SemanticAntiPatternException,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.domain.semantic_workflow import (
    SemanticStepClassification,
    SemanticWorkflowDraft,
    SemanticWorkflowException,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.skills import (
    SkillConstraintEnforcement,
    SkillSkipReason,
    SkillStepKind,
    SkippedWorkflow,
)
from skillforge.repository.skill_compiler import SkillCompiler


def ev(key):
    return Evidence(key, "repo", EvidenceType.GIT, SourceLocation("src/a.py"), key, "fixture")


def workflow(status=Status.CANDIDATE, scope_kind=SemanticWorkflowScopeKind.FILE):
    refs = ("w-action", "w-class", "w-decision", "w-exception", "w-intent", "w-scope")
    draft = SemanticWorkflowDraft(
        "wf",
        "repo",
        "candidate",
        "Do {name}",
        ("w-intent",),
        "Do {name}",
        ("w-intent",),
        (),
        (
            SemanticWorkflowStep("b", "Change src/a.py", ("w-action",)),
            SemanticWorkflowStep("a", "Review", ("w-action",)),
            SemanticWorkflowStep("c", "When needed", ("w-action",)),
        ),
        (SemanticStepClassification("b", "must", ("w-class",)),),
        (SemanticStepClassification("a", "may", ("w-class",)),),
        (SemanticStepClassification("c", "when", ("w-class",), "flag enabled"),),
        SemanticWorkflowScope(
            scope_kind,
            ("src/a.py",) if scope_kind == SemanticWorkflowScopeKind.FILE else (),
            "scope",
            "observed",
            ("w-scope",),
        ),
        (SemanticWorkflowException("Legacy path", ("w-exception",)),),
        refs,
        Support(2, 4),
        Confidence(0.8, "semantic", Support(2, 4)),
    )
    return ProjectWorkflow("wf", "repo", status, draft)


def rule(key, status, scope, kind=RuleType.TESTING, refs=None):
    refs = refs if refs is not None else (() if status == Status.MANUAL_OVERRIDE else (key,))
    exceptions = (ProjectRuleException("exception", refs),) if refs else ()
    return ProjectRule(key, "repo", kind, key, scope, status, refs, exceptions=exceptions)


def anti(key, status, scope_kind=SemanticAntiPatternScopeKind.FILE, selector="src/a.py"):
    refs = (key,)
    draft = SemanticAntiPatternDraft(
        key,
        "repo",
        key,
        key,
        refs,
        "history",
        refs,
        "avoid repeat",
        refs,
        "negative evidence",
        refs,
        SemanticAntiPatternScope(
            scope_kind,
            (selector,) if scope_kind == SemanticAntiPatternScopeKind.FILE else (),
            "scope",
            "observed",
            refs,
        ),
        (SemanticAntiPatternException("except legacy", refs),),
        "accepted",
        refs,
        refs,
        Confidence(0.6, "anti"),
    )
    return ProjectAntiPattern(key, "repo", status, draft)


def dna(workflows=None, rules=(), anti_patterns=()):
    workflows = (workflow(),) if workflows is None else workflows
    refs = (
        {key for item in workflows for key in item.draft.evidence_ids}
        | {key for item in rules for key in item.evidence_ids}
        | {key for item in anti_patterns for key in item.draft.evidence_ids}
    )
    return ProjectDNA(
        "repo",
        tuple(sorted(rules, key=lambda item: item.rule_id)),
        tuple(sorted(workflows, key=lambda item: item.workflow_id)),
        tuple(sorted(anti_patterns, key=lambda item: item.anti_pattern_id)),
        tuple(ev(key) for key in sorted(refs)),
    )


def compile_one(source):
    return SkillCompiler().compile(source).skills[0]


def test_basic_steps_completion_and_evidence_closure():
    source = dna()
    result = SkillCompiler().compile(source)
    skill = result.skills[0]
    assert [step.step_id for step in skill.steps] == ["b", "a", "c"]
    assert [step.kind for step in skill.steps] == [
        SkillStepKind.REQUIRED,
        SkillStepKind.OPTIONAL,
        SkillStepKind.CONDITIONAL,
    ]
    assert [step.condition for step in skill.steps] == [None, None, "flag enabled"]
    assert [item.step_id for item in skill.completion_conditions] == ["b", "c"]
    assert skill.steps[0].evidence_ids == ("w-action", "w-class")
    assert skill.trigger.task_description == "Do {name}"
    assert skill.parameters == ()
    assert skill.source_support == Support(2, 4)
    assert skill.source_confidence == source.workflows[0].draft.confidence
    assert skill.source_workflow_evidence_ids == source.workflows[0].draft.evidence_ids
    assert skill.evidence_ids == source.workflows[0].draft.evidence_ids
    assert tuple(item.evidence_id for item in result.evidence) == skill.evidence_ids
    assert result.to_dict()["skills"][0]["parameters"] == []


@pytest.mark.parametrize("status", [Status.CANDIDATE, Status.CONFIRMED, Status.MANUAL_OVERRIDE])
def test_compile_eligible_status_without_promotion(status):
    assert compile_one(dna(workflows=(workflow(status),))).source_status == status


@pytest.mark.parametrize(
    "status,reason",
    [
        (Status.DEPRECATED, SkillSkipReason.DEPRECATED),
        (Status.CONFLICTED, SkillSkipReason.CONFLICTED),
    ],
)
def test_skip_status_and_exclude_its_evidence(status, reason):
    result = SkillCompiler().compile(dna(workflows=(workflow(status),)))
    assert result.skills == result.evidence == ()
    assert result.skipped_workflows[0].reason == reason


def test_unknown_scope_is_skipped_without_widening():
    result = SkillCompiler().compile(
        dna(workflows=(workflow(scope_kind=SemanticWorkflowScopeKind.UNKNOWN),))
    )
    assert result.skills == result.evidence == ()
    assert result.skipped_workflows[0].reason == SkillSkipReason.UNKNOWN_SCOPE


def test_no_workflow_yields_empty_result():
    result = SkillCompiler().compile(dna(workflows=()))
    assert result.skills == result.skipped_workflows == result.evidence == ()


def test_rule_enforcement_scope_verify_and_leakage():
    repo = ProjectScope(ProjectScopeKind.REPOSITORY)
    file = ProjectScope(ProjectScopeKind.FILE, ("src/a.py",))
    other = ProjectScope(ProjectScopeKind.FILE, ("src/b.py",))
    directory = ProjectScope(ProjectScopeKind.DIRECTORY, ("src/",))
    rules = (
        rule("a-confirmed", Status.CONFIRMED, repo),
        rule("b-manual", Status.MANUAL_OVERRIDE, file),
        rule("c-candidate", Status.CANDIDATE, file),
        rule("d-conflicted", Status.CONFLICTED, file),
        rule("e-deprecated", Status.DEPRECATED, file),
        rule("f-other", Status.CONFIRMED, other),
        rule("g-directory", Status.CONFIRMED, directory),
        rule("h-release", Status.CONFIRMED, repo, RuleType.RELEASE),
    )
    result = SkillCompiler().compile(dna(rules=rules))
    skill = result.skills[0]
    assert [item.rule_id for item in skill.constraints] == [
        "a-confirmed",
        "b-manual",
        "c-candidate",
        "d-conflicted",
        "h-release",
    ]
    assert [item.enforcement for item in skill.constraints] == [
        SkillConstraintEnforcement.REQUIRED,
        SkillConstraintEnforcement.REQUIRED,
        SkillConstraintEnforcement.ADVISORY,
        SkillConstraintEnforcement.ADVISORY,
        SkillConstraintEnforcement.REQUIRED,
    ]
    assert skill.verify_rule_ids == ("a-confirmed", "b-manual")
    assert "b-manual" not in skill.evidence_ids
    assert {"e-deprecated", "f-other", "g-directory"}.isdisjoint(skill.evidence_ids)
    assert set(skill.evidence_ids) == {item.evidence_id for item in result.evidence}
    assert skill.constraints[0].exceptions[0].evidence_ids == ("a-confirmed",)


def test_risk_applicability_and_status_preservation():
    patterns = (
        anti("a-repo", Status.CANDIDATE, SemanticAntiPatternScopeKind.REPOSITORY),
        anti("b-file", Status.CONFLICTED),
        anti("c-other", Status.CONFIRMED, selector="src/b.py"),
        anti("d-unknown", Status.CONFIRMED, SemanticAntiPatternScopeKind.UNKNOWN),
        anti("e-deprecated", Status.DEPRECATED),
    )
    result = SkillCompiler().compile(dna(anti_patterns=patterns))
    skill = result.skills[0]
    assert [item.anti_pattern_id for item in skill.risks] == ["a-repo", "b-file"]
    assert skill.risks[0].source_status == Status.CANDIDATE
    assert skill.risks[1].source_status == Status.CONFLICTED
    assert skill.risks[0].confidence.support is None
    assert skill.risks[0].exceptions[0].evidence_ids == ("a-repo",)
    assert {"c-other", "d-unknown", "e-deprecated"}.isdisjoint(skill.evidence_ids)
    assert set(skill.evidence_ids) == {item.evidence_id for item in result.evidence}


def test_identity_version_and_unrelated_knowledge():
    source = dna()
    original = compile_one(source)
    changed_step = replace(source.workflows[0].draft.ordered_steps[0], action="Different action")
    changed_draft = replace(
        source.workflows[0].draft,
        ordered_steps=(changed_step, *source.workflows[0].draft.ordered_steps[1:]),
    )
    changed = compile_one(dna(workflows=(replace(source.workflows[0], draft=changed_draft),)))
    assert changed.skill_id == original.skill_id
    assert changed.version != original.version
    unrelated = compile_one(
        dna(
            rules=(
                rule(
                    "unrelated",
                    Status.CONFIRMED,
                    ProjectScope(ProjectScopeKind.FILE, ("src/other.py",)),
                ),
            )
        )
    )
    assert unrelated.skill_id == original.skill_id
    assert unrelated.version == original.version
    assert unrelated.source_snapshot_id != original.source_snapshot_id
    applicable = compile_one(
        dna(
            rules=(rule("applicable", Status.CONFIRMED, ProjectScope(ProjectScopeKind.REPOSITORY)),)
        )
    )
    assert applicable.version != original.version


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_step",
        "unknown_classification",
        "missing_classification",
        "duplicate_classification",
        "invalid_required_condition",
        "invalid_conditional_condition",
        "missing_nested_evidence",
        "invalid_scope",
    ],
)
def test_malformed_workflow_hard_fails(mutation):
    base = workflow()
    draft = base.draft
    if mutation == "duplicate_step":
        draft = replace(draft, ordered_steps=(draft.ordered_steps[0], *draft.ordered_steps))
    elif mutation == "unknown_classification":
        draft = replace(
            draft, required_steps=(replace(draft.required_steps[0], step_id="missing"),)
        )
    elif mutation == "missing_classification":
        draft = replace(draft, optional_steps=())
    elif mutation == "duplicate_classification":
        draft = replace(draft, optional_steps=(replace(draft.optional_steps[0], step_id="b"),))
    elif mutation == "invalid_required_condition":
        draft = replace(draft, required_steps=(replace(draft.required_steps[0], condition="if"),))
    elif mutation == "invalid_conditional_condition":
        draft = replace(
            draft, conditional_steps=(replace(draft.conditional_steps[0], condition=None),)
        )
    elif mutation == "missing_nested_evidence":
        draft = replace(
            draft,
            ordered_steps=(
                replace(draft.ordered_steps[0], evidence_ids=("alien",)),
                *draft.ordered_steps[1:],
            ),
        )
    else:
        draft = replace(draft, scope=replace(draft.scope, selectors=()))
    source = dna()
    object.__setattr__(source.workflows[0], "draft", draft)
    with pytest.raises(ValueError):
        SkillCompiler().compile(source)


def test_conflicting_or_missing_evidence_hard_fails():
    source = dna()
    object.__setattr__(
        source, "evidence", (*source.evidence, replace(source.evidence[0], description="other"))
    )
    with pytest.raises(ValueError, match="conflicting"):
        SkillCompiler().compile(source)
    source = dna()
    object.__setattr__(source, "evidence", source.evidence[1:])
    with pytest.raises(ValueError, match="missing"):
        SkillCompiler().compile(source)
    source = dna(workflows=(workflow(Status.DEPRECATED),))
    object.__setattr__(source, "evidence", source.evidence[1:])
    with pytest.raises(ValueError, match="missing"):
        SkillCompiler().compile(source)


def test_collection_permutation_is_deterministic():
    source = dna(
        rules=(
            rule("a", Status.CONFIRMED, ProjectScope(ProjectScopeKind.REPOSITORY)),
            rule("b", Status.CANDIDATE, ProjectScope(ProjectScopeKind.REPOSITORY)),
        ),
        anti_patterns=(anti("a-risk", Status.CANDIDATE), anti("b-risk", Status.CONFIRMED)),
    )
    expected = SkillCompiler().compile(source).to_dict()
    object.__setattr__(source, "rules", tuple(reversed(source.rules)))
    object.__setattr__(source, "anti_patterns", tuple(reversed(source.anti_patterns)))
    object.__setattr__(source, "evidence", tuple(reversed(source.evidence)))
    assert SkillCompiler().compile(source).to_dict() == expected


def test_multiple_workflows_exact_result_closure_and_order():
    first = workflow()
    second = ProjectWorkflow(
        "aa",
        "repo",
        Status.CONFIRMED,
        replace(first.draft, draft_id="aa", candidate_id="aa-candidate"),
    )
    skipped = ProjectWorkflow(
        "zz",
        "repo",
        Status.DEPRECATED,
        replace(
            first.draft,
            draft_id="zz",
            candidate_id="zz-candidate",
            evidence_ids=("z-skipped",),
            name_evidence_ids=("z-skipped",),
            intent_evidence_ids=("z-skipped",),
            scope=replace(first.draft.scope, evidence_ids=("z-skipped",)),
            ordered_steps=(),
            required_steps=(),
            optional_steps=(),
            conditional_steps=(),
            exceptions=(),
        ),
    )
    source = dna(workflows=(skipped, first, second))
    result = SkillCompiler().compile(source)
    assert [item.workflow_id for item in result.skills] == ["aa", "wf"]
    assert [item.workflow_id for item in result.skipped_workflows] == ["zz"]
    assert "z-skipped" not in {item.evidence_id for item in result.evidence}
    assert {item.evidence_id for item in result.evidence} == {
        key for skill in result.skills for key in skill.evidence_ids
    }


@pytest.mark.parametrize(
    "kind,selector,expected",
    [
        (SemanticWorkflowScopeKind.REPOSITORY, (), ProjectScopeKind.REPOSITORY),
        (SemanticWorkflowScopeKind.DIRECTORY, ("src/",), ProjectScopeKind.DIRECTORY),
        (SemanticWorkflowScopeKind.MODULE, ("pkg.a",), ProjectScopeKind.MODULE),
        (SemanticWorkflowScopeKind.FILE, ("src/a.py",), ProjectScopeKind.FILE),
    ],
)
def test_scope_conversion_is_exact(kind, selector, expected):
    base = workflow()
    source = dna()
    object.__setattr__(
        source.workflows[0],
        "draft",
        replace(base.draft, scope=replace(base.draft.scope, kind=kind, selectors=selector)),
    )
    assert compile_one(source).trigger.scope == ProjectScope(expected, selector)


def test_applicable_rule_provenance_metrics_and_version():
    repo = ProjectScope(ProjectScopeKind.REPOSITORY)
    base_rule = rule("metric-rule", Status.CANDIDATE, repo)
    low = replace(base_rule, support=Support(1, 4), confidence=Confidence(0.1, "observed"))
    high_support = replace(low, support=Support(2, 4))
    high_confidence = replace(low, confidence=Confidence(0.9, "observed"))
    base = compile_one(dna(rules=(low,)))
    constraint = base.constraints[0]
    assert constraint.source_support == low.support
    assert constraint.source_confidence == low.confidence
    assert constraint.enforcement == SkillConstraintEnforcement.ADVISORY
    assert compile_one(dna(rules=(high_support,))).version != base.version
    confidence_skill = compile_one(dna(rules=(high_confidence,)))
    assert confidence_skill.version != base.version
    assert confidence_skill.constraints[0].enforcement == constraint.enforcement
    confirmed = replace(low, status=Status.CONFIRMED)
    assert compile_one(dna(rules=(confirmed,))).constraints[0].enforcement == (
        SkillConstraintEnforcement.REQUIRED
    )
    unrelated_scope = ProjectScope(ProjectScopeKind.FILE, ("src/unrelated.py",))
    unrelated_low = replace(low, scope=unrelated_scope)
    unrelated_high = replace(high_confidence, scope=unrelated_scope)
    unrelated_support = replace(high_support, scope=unrelated_scope)
    assert (
        compile_one(dna(rules=(unrelated_low,))).version
        == compile_one(dna(rules=(unrelated_high,))).version
    )
    assert (
        compile_one(dna(rules=(unrelated_low,))).version
        == compile_one(dna(rules=(unrelated_support,))).version
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_evidence",
        "missing_nested_evidence",
        "duplicate_evidence",
        "invalid_verify_id",
        "optional_completion",
        "missing_required_completion",
        "conditional_completion_mismatch",
        "duplicate_step",
        "candidate_required_constraint",
        "invalid_source_status",
        "risk_with_support",
        "missing_source_workflow_evidence",
        "constraint_exception_outside_rule",
        "risk_exception_outside_pattern",
    ],
)
def test_direct_project_skill_rejects_invalid_contract(mutation):
    base = compile_one(
        dna(
            rules=(rule("verify", Status.CONFIRMED, ProjectScope(ProjectScopeKind.REPOSITORY)),),
            anti_patterns=(anti("risk", Status.CANDIDATE),),
        )
    )
    if mutation == "extra_evidence":
        changes = {"evidence_ids": (*base.evidence_ids, "z-extra")}
    elif mutation == "missing_nested_evidence":
        changes = {"evidence_ids": tuple(key for key in base.evidence_ids if key != "w-action")}
    elif mutation == "duplicate_evidence":
        changes = {"evidence_ids": (*base.evidence_ids, base.evidence_ids[-1])}
    elif mutation == "invalid_verify_id":
        changes = {"verify_rule_ids": ("unknown",)}
    elif mutation == "optional_completion":
        changes = {
            "completion_conditions": (
                *base.completion_conditions,
                replace(base.completion_conditions[0], step_id="a"),
            )
        }
    elif mutation == "missing_required_completion":
        changes = {"completion_conditions": base.completion_conditions[1:]}
    elif mutation == "conditional_completion_mismatch":
        changes = {
            "completion_conditions": (
                base.completion_conditions[0],
                replace(base.completion_conditions[1], condition="different"),
            )
        }
    elif mutation == "duplicate_step":
        changes = {"steps": (*base.steps, base.steps[0])}
    elif mutation == "candidate_required_constraint":
        changes = {"constraints": (replace(base.constraints[0], source_status=Status.CANDIDATE),)}
    elif mutation == "invalid_source_status":
        changes = {"source_status": "candidate"}
    elif mutation == "missing_source_workflow_evidence":
        changes = {"source_workflow_evidence_ids": ()}
    elif mutation == "constraint_exception_outside_rule":
        changes = {
            "constraints": (
                replace(
                    base.constraints[0],
                    exceptions=(
                        replace(base.constraints[0].exceptions[0], evidence_ids=("z-extra",)),
                    ),
                ),
            ),
            "evidence_ids": (*base.evidence_ids, "z-extra"),
        }
    elif mutation == "risk_exception_outside_pattern":
        changes = {
            "risks": (
                replace(
                    base.risks[0],
                    exceptions=(replace(base.risks[0].exceptions[0], evidence_ids=("z-extra",)),),
                ),
            ),
            "evidence_ids": (*base.evidence_ids, "z-extra"),
        }
    else:
        changes = {
            "risks": (replace(base.risks[0], confidence=Confidence(0.5, "anti", Support(1, 2))),)
        }
    with pytest.raises(ValueError):
        replace(base, **changes)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_evidence",
        "missing_evidence",
        "wrong_repository_evidence",
        "duplicate_skill_id",
        "compiled_and_skipped",
        "snapshot_mismatch",
    ],
)
def test_direct_compilation_result_rejects_invalid_contract(mutation):
    base = SkillCompiler().compile(dna())
    if mutation == "extra_evidence":
        changes = {"evidence": (*base.evidence, ev("z-extra"))}
    elif mutation == "missing_evidence":
        changes = {"evidence": base.evidence[1:]}
    elif mutation == "wrong_repository_evidence":
        changes = {
            "evidence": (replace(base.evidence[0], repository_id="other"), *base.evidence[1:])
        }
    elif mutation == "duplicate_skill_id":
        changes = {"skills": (base.skills[0], replace(base.skills[0], workflow_id="other"))}
    elif mutation == "compiled_and_skipped":
        changes = {
            "skipped_workflows": (
                SkippedWorkflow(base.skills[0].workflow_id, SkillSkipReason.DEPRECATED),
            )
        }
    else:
        changes = {"skills": (replace(base.skills[0], source_snapshot_id="other"),)}
    with pytest.raises(ValueError):
        replace(base, **changes)
