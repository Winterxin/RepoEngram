from dataclasses import replace

import pytest
from test_static_delegation import source

from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralResponse,
)
from skillforge.integration.agent_skills import _render_skill
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_rules import SemanticStructuralRuleEvaluator
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.structural_rules import StructuralRuleMiner


@pytest.mark.parametrize("options", [
    {"operations": 1}, {"inconsistent": True}, {"private": True}, {"helper": True},
])
def test_invalid_topology_has_no_structural_rule(tmp_path, options):
    candidates = StructuralRuleMiner().mine(source(tmp_path, **options)).candidates
    # The original strict chain is still rejected. The independently valid
    # Coordinator -> Vault pair can now be observed, even if Entry is incomplete.
    assert all(candidate.delegation_pattern == "REPEATED_OWNER_DELEGATION"
               and candidate.source_package == "nebula.middle.Coordinator"
               for candidate in candidates)
    if options.get("operations") == 1 or options.get("private"):
        assert candidates == ()


@pytest.mark.parametrize("reject", [False, True])
def test_structural_semantic_dna_guidance(tmp_path, reject):
    facts = source(tmp_path)
    mined = StructuralRuleMiner().mine(facts)
    candidate, = mined.candidates
    assert candidate.rule.support.supporting == 2
    assert len(candidate.delegation_topology) == 3
    assert len(candidate.operation_symbols) == 2
    assert all((tmp_path / e.location.path).is_file() for e in mined.evidence)
    assert sum(e.description.startswith("Python call ") for e in mined.evidence) == 4

    class Model:
        def complete(self, request):
            assert request.observation_kind == "static_delegation"
            refs = tuple(e.evidence_id for e in request.evidence)
            return SemanticStructuralResponse(
                request.repository_id, request.candidate_id,
                SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE if reject
                else SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
                "Current resolved calls support a conditional architecture convention",
                refs, request.source_package, request.target_package,
                () if reject else request.source_modules, () if reject else refs, (),
            )

    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, Model())
    dna = ProjectDNABuilder().build(facts.repository_id, semantic_structural_rules=semantic)
    compiled = SkillCompiler().compile(dna, semantic)
    if reject:
        assert len(semantic.rejections) == 1
        assert not dna.rules and not compiled.skills
    else:
        skill, = compiled.skills
        assert skill.source_rule_id == dna.rules[0].rule_id
        assert len(skill.procedure.steps) == 7
        rendered = _render_skill(skill, "fixture")
        for token in ("same observed family", "contrary", "owning", "end-to-end",
                      "Evidence", "nebula.entry.Entry.change_0", "[CONDITIONAL]"):
            assert token in rendered
        assert all(step.evidence_ids == candidate.rule.evidence_ids
                   for step in skill.procedure.steps)


def test_candidate_rejects_missing_independent_support(tmp_path):
    candidate, = StructuralRuleMiner().mine(source(tmp_path)).candidates
    with pytest.raises(ValueError, match="complete repeated topology"):
        replace(candidate, operation_symbols=candidate.operation_symbols[:1])
