"""Generic generated-method contracts for responsibility-based applicability."""

import pytest
from test_executable_skill_compiler import tendency as base_tendency
from test_skill_compiler import dna

from skillforge.domain.skills import SkillCompilationResult, SkillStepKind
from skillforge.integration.agent_skills import _render_guidance
from skillforge.productization.evolution_artifacts import _restore
from skillforge.repository.skill_compiler import SkillCompiler


@pytest.fixture
def tendency(tmp_path):
    return base_tendency.__wrapped__(tmp_path)


def compile_method(fixture):
    source, semantic = fixture
    compiled = SkillCompiler().compile(source, semantic_tendencies=semantic)
    return compiled, compiled.skills[0]


def test_clear_owner_with_extra_computation_keeps_responsibility(tendency):
    # An existing validation unit still owns validation when another calculation is added.
    _, skill = compile_method(tendency)
    inspect, compare, branch, _ = skill.procedure.steps
    assert "functions" in inspect.instruction and "callers" in inspect.instruction
    assert "tests" in inspect.instruction and "owner candidates" in inspect.instruction
    assert "same core responsibility" in compare.instruction
    assert "extend that unit by default" in compare.instruction
    for phrase in ("computation steps", "parameters", "boundary conditions"):
        assert phrase in compare.instruction
    assert compare.kind == SkillStepKind.REQUIRED
    assert branch.kind == SkillStepKind.CONDITIONAL


def test_distinct_responsibility_can_reach_history_branch(tendency):
    # A new scheduling responsibility may use history after ruling out current owners.
    _, skill = compile_method(tendency)
    branch = skill.procedure.steps[2]
    for phrase in ("no clear existing owner", "genuinely distinct", "current scope",
                   "Project Rule", "Workflow", "exception", "contrary project evidence"):
        assert phrase in branch.condition
    assert "prefer a distinct direct child Python module" in branch.instruction
    assert "tie-breaker / conditional guidance" in branch.instruction
    assert "never a default new-module rule" in branch.instruction


def test_complexity_does_not_establish_a_distinct_capability(tendency):
    _, skill = compile_method(tendency)
    compare = skill.procedure.steps[1].instruction
    for phrase in ("internal algorithm", "extra computation", "a helper", "more logic"):
        assert phrase in compare
    assert "does not by itself establish a distinct capability" in compare
    assert "Only after establishing no clear existing owner" in compare


def test_uncertain_ownership_and_missing_justification_preserve_ownership(tendency):
    _, skill = compile_method(tendency)
    compare, branch = skill.procedure.steps[1:3]
    assert "ownership remains uncertain, preserve existing ownership" in compare.instruction
    for phrase in ("owner candidates were checked", "why none owns", "genuinely distinct",
                   "which historical Evidence", "justification cannot be completed",
                   "preserve existing ownership"):
        assert phrase in branch.instruction


def test_sibling_verify_rechecks_ownership_independently(tendency):
    _, skill = compile_method(tendency)
    verify = next(s for s in skill.procedure.verify if s.step_id == "verify-ownership")
    assert verify.kind == SkillStepKind.CONDITIONAL
    assert verify.condition == "A new sibling module was created"
    for phrase in ("same core responsibility", "genuinely distinct", "complexity increased",
                   "advisory and conditional", "contrary project evidence",
                   "revise the placement", "tests passing alone cannot prove"):
        assert phrase in verify.instruction
    text = _render_guidance(skill, "fixture")
    assert verify.instruction in text.split("# Verify\n")[1]


def test_gate_preserves_provenance_authority_and_serialization(tendency):
    source, semantic = tendency
    compiled, skill = compile_method(tendency)
    rule = source.rules[0]
    assert skill.source_status == rule.status
    assert skill.source_support == rule.support
    assert skill.source_confidence == rule.confidence
    assert skill.semantic_candidate_id == semantic.accepted[0].response.candidate_id
    assert skill.semantic_evidence_ids == semantic.accepted[0].response.evidence_ids
    assert skill.constraint.enforcement == "advisory"
    assert skill.procedure.steps[2].evidence_ids == rule.evidence_ids
    assert compiled.evidence == source.evidence
    assert _restore(SkillCompilationResult, compiled.to_dict()) == compiled


def test_workflow_without_tendency_does_not_acquire_sibling_guard():
    skill, = SkillCompiler().compile(dna()).skills
    assert all(s.step_id != "verify-ownership" for s in skill.procedure.verify)
