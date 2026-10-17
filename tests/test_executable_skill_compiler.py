"""Executable compiler contracts using unrelated synthetic repository knowledge."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_semantic_tendencies import request_for, response_for
from test_skill_compiler import dna, rule, workflow
from test_structural_roles import _repo

from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation
from skillforge.domain.project_dna import (
    ProjectDNA,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus as Status,
)
from skillforge.domain.semantic_tendency import SemanticTendencyResult
from skillforge.domain.skills import SkillCompilationResult, SkillStepKind
from skillforge.integration.agent_skills import _render_guidance, _render_skill
from skillforge.integration.skill_instructions import render_instruction_audit
from skillforge.productization.evolution_artifacts import _restore
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_tendencies import validate_semantic_tendency_response
from skillforge.repository.skill_compiler import SkillCompiler


@pytest.fixture
def tendency(tmp_path):
    inputs = _repo(tmp_path, [(f"nebula/plugins/unit_{i}.py",) for i in range(3)],
                   initial=("nebula/__init__.py", "nebula/plugins/__init__.py"))
    request = request_for(inputs)
    accepted = validate_semantic_tendency_response(request, response_for(request))
    sources = {item.evidence_id: item for item in (*inputs[2].evidence, *inputs[3].evidence)}
    result = SemanticTendencyResult(accepted.rule.repository_id, (accepted,), (),
                                   tuple(sources[key] for key in accepted.rule.evidence_ids))
    source = ProjectDNABuilder().build(result.repository_id, semantic_tendencies=result)
    return source, result


def test_workflow_is_ordered_executable_method():
    skill, = SkillCompiler().compile(dna()).skills
    assert skill.procedure.steps == skill.steps
    assert [step.step_id for step in skill.procedure.steps] == ["b", "a", "c"]
    assert skill.procedure.steps[2].condition == "flag enabled"
    text = _render_skill(skill, "fixture")
    for heading in ("When this applies", "Decision procedure", "Exceptions and risks",
                    "References"):
        assert f"## {heading}\n" in text
    assert text.index("[REQUIRED] Change src/a.py") < text.index("[OPTIONAL] Review")


def test_tendency_has_conditional_preference_not_unconditional_module_rule(tendency):
    source, result = tendency
    skill, = SkillCompiler().compile(source, semantic_tendencies=result).skills
    decision = skill.procedure.steps[2]
    assert decision.kind == SkillStepKind.CONDITIONAL
    assert "no clear existing owner" in decision.condition
    assert "prefer a distinct direct child Python module" in decision.instruction
    assert "extend that unit" in skill.procedure.steps[1].instruction
    assert decision.evidence_ids == source.rules[0].evidence_ids
    assert skill.constraint.enforcement == "advisory"
    text = _render_guidance(skill, "fixture")
    assert "not an execution procedure" not in text
    assert "never unconditional MUST" in text
    assert "unit_0.py" not in text and "unit_1.py" not in text


def test_rules_and_verify_knowledge_compose_in_workflow():
    scope = ProjectScope(ProjectScopeKind.REPOSITORY)
    rules = (replace(rule("naming", Status.CONFIRMED, scope, RuleType.CONVENTION),
                     statement="Preserve the documented public naming convention"),
             replace(rule("checks", Status.CONFIRMED, scope),
                     statement="Run the existing contract checks"))
    skill, = SkillCompiler().compile(dna(rules=rules)).skills
    assert {item.rule_id for item in skill.procedure.constraints} == {"naming", "checks"}
    assert skill.procedure.constraints[1].enforcement == "required"
    verify = next(item for item in skill.procedure.verify if item.step_id == "verify-checks")
    assert "Run the existing contract checks" in verify.instruction
    assert verify.evidence_ids == ("checks",)
    text = _render_skill(skill, "fixture")
    assert "Run the existing contract checks" in text.split("## Verify\n")[1]


def test_tendency_composes_same_scope_rule_and_verify(tendency):
    source, semantic = tendency
    primary, = source.rules
    # A separate synthetic source proves closure expands when knowledge is composed.
    ref = Evidence("z-check", source.repository_id, EvidenceType.GIT,
                   SourceLocation("nebula/plugins/__init__.py"), "contract fixture", "fixture")
    checks = ProjectRule("check-contract", source.repository_id, RuleType.TESTING,
                         "Check exported public behavior", primary.scope, Status.CANDIDATE,
                         (ref.evidence_id,))
    source = replace(source,
                     rules=tuple(sorted((*source.rules, checks), key=lambda r: r.rule_id)),
                     evidence=tuple(sorted((*source.evidence, ref), key=lambda e: e.evidence_id)))
    compilation = SkillCompiler().compile(source, semantic_tendencies=semantic)
    skill, = compilation.skills
    assert skill.procedure.constraints[0].rule_id == checks.rule_id
    assert skill.procedure.constraints[0].enforcement == "advisory"
    assert any(item.step_id == "verify-check-contract" for item in skill.procedure.verify)
    assert skill.evidence_ids == (*primary.evidence_ids, ref.evidence_id)
    assert compilation.evidence == source.evidence
    assert _restore(SkillCompilationResult, compilation.to_dict()) == compilation


@pytest.mark.parametrize("scope", [
    ProjectScope(ProjectScopeKind.FILE, ("src/other.py",)),
    ProjectScope(ProjectScopeKind.DIRECTORY, ("src",)),
    ProjectScope(ProjectScopeKind.FILE, ("src/a.py", "src/other.py")),
])
def test_unrelated_or_partially_overlapping_scope_is_not_silently_composed(scope):
    skill, = SkillCompiler().compile(dna(rules=(rule("other", Status.CONFIRMED, scope),))).skills
    assert skill.procedure.constraints == ()
    assert all("other" not in step.evidence_ids for step in skill.procedure.verify)


def test_conflicted_knowledge_is_explicitly_excluded():
    source = dna(rules=(rule("uncertain", Status.CONFLICTED,
                            ProjectScope(ProjectScopeKind.REPOSITORY)),))
    skill, = SkillCompiler().compile(source).skills
    assert skill.procedure.constraints == ()
    assert skill.procedure.excluded_rule_ids == ("uncertain",)
    audit = render_instruction_audit(skill)
    assert '"excluded_rule_ids": [\n    "uncertain"' in audit
    assert "uncertain" not in _render_skill(skill, "fixture")


def test_low_confidence_and_candidate_boundaries_are_preserved():
    original = workflow()
    low = replace(original, draft=replace(original.draft, confidence=Confidence(0.2, "fixture")))
    skill, = SkillCompiler().compile(dna(workflows=(low,))).skills
    assert skill.source_status == Status.CANDIDATE
    assert skill.source_confidence.score == 0.2
    text = _render_skill(skill, "fixture")
    assert '"score": 0.2' in render_instruction_audit(skill)
    assert '"score": 0.2' not in text
    assert "does not become a confirmed project policy" in text


def test_provenance_and_procedure_survive_canonical_round_trip(tendency):
    source, semantic = tendency
    compiled = SkillCompiler().compile(source, semantic_tendencies=semantic)
    restored = _restore(SkillCompilationResult, compiled.to_dict())
    assert restored == compiled
    assert restored.evidence == source.evidence
    assert restored.skills[0].semantic_evidence_ids == semantic.accepted[0].response.evidence_ids


def test_procedure_cannot_escape_evidence_closure(tendency):
    source, semantic = tendency
    skill, = SkillCompiler().compile(source, semantic_tendencies=semantic).skills
    procedure = replace(skill.procedure, steps=(replace(skill.procedure.steps[0],
                                                       evidence_ids=("invented",)),))
    with pytest.raises(ValueError, match="Evidence"):
        replace(skill, procedure=procedure)


def test_no_unbacked_manual_rule_is_presented_as_evidence_backed_constraint():
    source = dna(rules=(rule("manual", Status.MANUAL_OVERRIDE,
                            ProjectScope(ProjectScopeKind.REPOSITORY)),))
    skill, = SkillCompiler().compile(source).skills
    assert skill.procedure.constraints == ()


def test_no_benchmark_names_in_compiler_logic():
    import skillforge.repository.skill_compiler as compiler
    text = Path(compiler.__file__).read_text(encoding="utf-8").lower()
    for name in ("supportdesk", "operation", "batch", "reply", "tickets", "comments"):
        assert name not in text


def test_dna_still_has_exact_evidence_closure(tendency):
    source, _ = tendency
    assert _restore(ProjectDNA, source.to_dict()) == source


def test_empty_workflow_cannot_be_exported_as_executable_method():
    original = workflow()
    empty = replace(original, draft=replace(original.draft, ordered_steps=(), required_steps=(),
                                           optional_steps=(), conditional_steps=()))
    with pytest.raises(ValueError, match="no executable steps"):
        SkillCompiler().compile(dna(workflows=(empty,)))
