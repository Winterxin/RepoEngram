"""Project DNA promotion and Evidence closure integration contracts."""

import json
from dataclasses import replace

import pytest

from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectRuleException,
    ProjectScope,
    ProjectScopeKind,
    ProjectWorkflow,
    RuleType,
)
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDraft,
    SemanticAntiPatternRejection,
    SemanticAntiPatternResult,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticWorkflowDraft,
    SemanticWorkflowRejection,
    SemanticWorkflowResult,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
)
from skillforge.repository.project_dna import ProjectDNABuilder


def evidence(key, description=None):
    return Evidence(
        key, "repo", EvidenceType.GIT, SourceLocation("src/a.py"),
        description or key, "fixture",
    )


def rule(key="rule", refs=("rule-ev",), status=ProjectKnowledgeStatus.CANDIDATE):
    return ProjectRule(
        key, "repo", RuleType.ARCHITECTURE, "Use a service layer",
        ProjectScope(ProjectScopeKind.DIRECTORY, ("src/",)), status, refs,
    )


def workflow(key="workflow", ref="wf-ev"):
    return SemanticWorkflowDraft(
        draft_id=key,
        repository_id="repo",
        candidate_id=f"candidate-{key}",
        name="Add route",
        name_evidence_ids=(ref,),
        intent="Add route",
        intent_evidence_ids=(ref,),
        roles=(),
        ordered_steps=(),
        required_steps=(),
        optional_steps=(),
        conditional_steps=(),
        scope=SemanticWorkflowScope(
            SemanticWorkflowScopeKind.FILE, ("src/a.py",), "file", "observed", (ref,)
        ),
        exceptions=(),
        evidence_ids=(ref,),
        support=Support(2, 4),
        confidence=Confidence(0.8, "llm_semantic_workflow_v1", Support(2, 4)),
    )


def anti_pattern(key="anti", ref="ap-ev"):
    return SemanticAntiPatternDraft(
        draft_id=key,
        repository_id="repo",
        candidate_id=f"candidate-{key}",
        name="Avoid repeated edit",
        name_evidence_ids=(ref,),
        description="Repeated edit",
        description_evidence_ids=(ref,),
        avoid_behavior="Avoid repeated edit",
        avoid_behavior_evidence_ids=(ref,),
        negative_evidence_summary="Repeated edit",
        negative_evidence_ids=(ref,),
        scope=SemanticAntiPatternScope(
            SemanticAntiPatternScopeKind.FILE, ("src/a.py",), "file", "observed", (ref,)
        ),
        exceptions=(),
        decision_reason="Evidence supports the pattern",
        decision_evidence_ids=(ref,),
        evidence_ids=(ref,),
        confidence=Confidence(0.7, "llm_semantic_antipattern_v1"),
    )


def workflow_result(drafts=(), rejection=False):
    rejected = (
        SemanticWorkflowRejection(
            "repo", "rejected-workflow", SemanticDecisionKind.INSUFFICIENT_EVIDENCE,
            "insufficient", ("wf-reject",),
            Confidence(0.2, "llm_semantic_workflow_v1"), Support(1, 4),
        ),
    ) if rejection else ()
    keys = {ref for draft in drafts for ref in draft.evidence_ids}
    if rejection:
        keys.add("wf-reject")
    return SemanticWorkflowResult(
        "repo", tuple(drafts), tuple(evidence(key) for key in sorted(keys)), rejected
    )


def anti_result(drafts=(), rejection=False):
    rejected = (
        SemanticAntiPatternRejection(
            "repo", "rejected-anti", SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
            "insufficient", ("ap-reject",), Confidence(0.2, "llm_semantic_antipattern_v1"),
        ),
    ) if rejection else ()
    keys = {ref for draft in drafts for ref in draft.evidence_ids}
    if rejection:
        keys.add("ap-reject")
    return SemanticAntiPatternResult(
        "repo", tuple(drafts), rejected, tuple(evidence(key) for key in sorted(keys))
    )


def build(**kwargs):
    return ProjectDNABuilder().build("repo", **kwargs)


def test_basic_aggregate_preserves_distinct_objects_metrics_and_exact_evidence():
    dna = build(
        rules=(rule(),),
        rule_evidence=(evidence("rule-ev"), evidence("unused")),
        workflows=workflow_result((workflow(),)),
        anti_patterns=anti_result((anti_pattern(),)),
    )
    assert tuple(item.evidence_id for item in dna.evidence) == (
        "ap-ev", "rule-ev", "wf-ev"
    )
    assert isinstance(dna.workflows[0], ProjectWorkflow)
    assert isinstance(dna.anti_patterns[0], ProjectAntiPattern)
    assert dna.workflows[0].status == dna.anti_patterns[0].status == (
        ProjectKnowledgeStatus.CANDIDATE
    )
    changed_workflow_status = replace(
        dna,
        workflows=(replace(dna.workflows[0], status=ProjectKnowledgeStatus.CONFIRMED),),
    )
    assert changed_workflow_status.snapshot_id != dna.snapshot_id
    assert dna.workflows[0].draft.support == Support(2, 4)
    assert dna.anti_patterns[0].draft.confidence.support is None
    assert dna.rules[0].support is None
    assert dna.rules[0].confidence is None
    assert "WORKFLOW" not in RuleType.__members__
    assert "ANTI_PATTERN" not in RuleType.__members__
    assert json.loads(json.dumps(dna.to_dict())) == dna.to_dict()


def test_rejections_and_their_evidence_never_enter_dna():
    dna = build(
        workflows=workflow_result((workflow(),), rejection=True),
        anti_patterns=anti_result(rejection=True),
    )
    assert len(dna.workflows) == 1
    assert dna.anti_patterns == ()
    assert tuple(item.evidence_id for item in dna.evidence) == ("wf-ev",)
    assert "rejections" not in dna.to_dict()
    empty = build(anti_patterns=anti_result(rejection=True))
    assert empty.anti_patterns == empty.evidence == ()


def test_manual_rule_has_no_fabricated_provenance():
    manual = rule(refs=(), status=ProjectKnowledgeStatus.MANUAL_OVERRIDE)
    dna = build(rules=(manual,))
    assert dna.rules == (manual,)
    assert dna.evidence == ()
    assert dna.rules[0].status == ProjectKnowledgeStatus.MANUAL_OVERRIDE
    with pytest.raises(ValueError, match="require Evidence"):
        rule(refs=())


def test_empty_builder_does_not_promote_repository_facts_to_rules():
    dna = build()
    assert dna.rules == dna.workflows == dna.anti_patterns == dna.evidence == ()


@pytest.mark.parametrize("scope", [
    ProjectScopeKind.DIRECTORY, ProjectScopeKind.MODULE, ProjectScopeKind.FILE,
    ProjectScopeKind.FILE_TYPE, ProjectScopeKind.DOMAIN,
])
def test_nonrepository_scope_requires_selectors(scope):
    with pytest.raises(ValueError, match="require selectors"):
        ProjectScope(scope)


def test_scope_and_exception_contract():
    with pytest.raises(ValueError):
        ProjectScope(ProjectScopeKind.REPOSITORY, ("src/",))
    with pytest.raises(ValueError):
        ProjectScope(ProjectScopeKind.FILE, ("b.py", "a.py"))
    with pytest.raises(ValueError):
        ProjectScope(ProjectScopeKind.FILE, ("a.py", "a.py"))
    with pytest.raises(ValueError, match="exception Evidence"):
        replace(rule(), exceptions=(ProjectRuleException("except tests", ("unknown",)),))


@pytest.mark.parametrize("status", [
    ProjectKnowledgeStatus.CANDIDATE,
    ProjectKnowledgeStatus.CONFIRMED,
    ProjectKnowledgeStatus.DEPRECATED,
    ProjectKnowledgeStatus.CONFLICTED,
])
def test_automatic_rule_exception_requires_its_own_evidence(status):
    with pytest.raises(ValueError, match="rule exceptions require Evidence"):
        replace(
            rule(status=status),
            exceptions=(ProjectRuleException("except background jobs"),),
        )


def test_automatic_rule_evidence_backed_exception_enters_exact_closure():
    automatic = replace(
        rule(refs=("exception-ev", "rule-ev")),
        exceptions=(ProjectRuleException("except background jobs", ("exception-ev",)),),
    )
    dna = build(
        rules=(automatic,),
        rule_evidence=(
            evidence("rule-ev"),
            evidence("exception-ev"),
            evidence("unrelated"),
        ),
    )
    assert dna.rules == (automatic,)
    assert tuple(item.evidence_id for item in dna.evidence) == ("exception-ev", "rule-ev")


def test_manual_rule_and_exception_may_both_omit_repository_evidence():
    manual = replace(
        rule(refs=(), status=ProjectKnowledgeStatus.MANUAL_OVERRIDE),
        exceptions=(ProjectRuleException("except background jobs"),),
    )
    dna = build(rules=(manual,))
    assert dna.rules == (manual,)
    assert dna.evidence == ()


def test_manual_rule_with_evidence_still_enforces_exception_subset_and_closure():
    manual = replace(
        rule(status=ProjectKnowledgeStatus.MANUAL_OVERRIDE),
        exceptions=(ProjectRuleException("except background jobs", ("rule-ev",)),),
    )
    dna = build(rules=(manual,), rule_evidence=(evidence("rule-ev"),))
    assert tuple(item.evidence_id for item in dna.evidence) == ("rule-ev",)
    with pytest.raises(ValueError, match="exception Evidence"):
        replace(
            manual,
            exceptions=(ProjectRuleException("except background jobs", ("outside",)),),
        )
    with pytest.raises(ValueError, match="repository_id mismatch"):
        build(
            rules=(manual,),
            rule_evidence=(replace(evidence("rule-ev"), repository_id="other"),),
        )


def test_unknown_rule_evidence_and_repository_mismatch_rejected():
    with pytest.raises(ValueError, match="unknown rule Evidence"):
        build(rules=(rule(),))
    with pytest.raises(ValueError, match="repository_id mismatch"):
        build(rules=(rule(),), rule_evidence=(replace(evidence("rule-ev"), repository_id="other"),))
    with pytest.raises(ValueError, match="rule repository_id mismatch"):
        build(rules=(replace(rule(), repository_id="other"),))


def test_conflicting_and_identical_evidence_across_sources():
    shared = workflow(ref="shared")
    same = anti_pattern(ref="shared")
    dna = build(
        rules=(rule(refs=("shared",)),),
        rule_evidence=(evidence("shared"), evidence("shared")),
        workflows=workflow_result((shared,)),
        anti_patterns=anti_result((same,)),
    )
    assert tuple(item.evidence_id for item in dna.evidence) == ("shared",)
    with pytest.raises(ValueError, match="conflicting Evidence ID"):
        build(rules=(rule(),), rule_evidence=(
            evidence("rule-ev"), evidence("rule-ev", "conflict")
        ))
    with pytest.raises(ValueError, match="conflicting Evidence ID"):
        build(
            rules=(rule(refs=("shared",)),),
            rule_evidence=(evidence("shared", "conflict"),),
            workflows=workflow_result((shared,)),
        )


def test_deterministic_order_and_snapshot_changes_with_content():
    rules = (rule("z", ("a",)), rule("a", ("b",)))
    wf = (workflow("z-wf", "c"), workflow("a-wf", "d"))
    ap = (anti_pattern("z-ap", "e"), anti_pattern("a-ap", "f"))
    first = build(
        rules=rules,
        rule_evidence=(evidence("b"), evidence("a")),
        workflows=workflow_result(wf),
        anti_patterns=anti_result(ap),
    )
    second = build(
        rules=rules[::-1],
        rule_evidence=(evidence("a"), evidence("b")),
        workflows=workflow_result(wf[::-1]),
        anti_patterns=anti_result(ap[::-1]),
    )
    assert first.to_dict() == second.to_dict()
    assert first.snapshot_id == second.snapshot_id
    changed_status = build(
        rules=(replace(rules[0], status=ProjectKnowledgeStatus.CONFIRMED), rules[1]),
        rule_evidence=(evidence("a"), evidence("b")),
        workflows=workflow_result(wf),
        anti_patterns=anti_result(ap),
    )
    assert first.snapshot_id != changed_status.snapshot_id
    changed_evidence = build(
        rules=rules,
        rule_evidence=(evidence("a", "changed"), evidence("b")),
        workflows=workflow_result(wf),
        anti_patterns=anti_result(ap),
    )
    assert first.snapshot_id != changed_evidence.snapshot_id
    changed_reference = build(
        rules=(replace(rules[0], evidence_ids=("g",)), rules[1]),
        rule_evidence=(evidence("g"), evidence("b")),
        workflows=workflow_result(wf),
        anti_patterns=anti_result(ap),
    )
    assert first.snapshot_id != changed_reference.snapshot_id


def test_duplicate_objects_and_invalid_ids_hard_fail():
    with pytest.raises(ValueError, match="duplicate canonical rule_id"):
        build(rules=(rule(), rule()))
    with pytest.raises(ValueError, match="draft IDs must be unique"):
        workflow_result((workflow(), workflow()))
    with pytest.raises(ValueError, match="draft IDs must be unique"):
        anti_result((anti_pattern(), anti_pattern()))
    with pytest.raises(ValueError, match="rule_id is required"):
        rule(key="")
    with pytest.raises(ValueError, match="Evidence ID is required"):
        build(rule_evidence=(evidence(""),))
    with pytest.raises(ValueError, match="exactly match"):
        ProjectDNA("repo", (rule(),), (), (), ())


def test_semantic_repositories_and_unknown_references_fail():
    with pytest.raises(ValueError, match="semantic result repository_id mismatch"):
        build(workflows=replace(workflow_result(), repository_id="other"))
    with pytest.raises(ValueError, match="semantic result repository_id mismatch"):
        build(anti_patterns=replace(anti_result(), repository_id="other"))
    with pytest.raises(ValueError, match="Evidence must match draft references"):
        replace(
            workflow_result((workflow(),)),
            workflows=(replace(workflow(), evidence_ids=("missing",)),),
        )


def test_forged_nested_workflow_reference_cannot_bypass_evidence_closure():
    accepted = workflow()
    forged = replace(
        accepted,
        scope=replace(accepted.scope, evidence_ids=("hidden",)),
    )
    result = replace(workflow_result((accepted,)), workflows=(forged,))
    with pytest.raises(ValueError, match="nested references"):
        build(workflows=result)


def test_direct_aggregate_rejects_extra_evidence():
    with pytest.raises(ValueError, match="exactly match"):
        ProjectDNA("repo", (), (), (), (evidence("unrelated"),))
    with pytest.raises(ValueError, match="nonempty strings"):
        ProjectDNA("repo", (), (), (), (evidence(""),))
