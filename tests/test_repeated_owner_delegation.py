from dataclasses import replace
from textwrap import dedent

import pytest

from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralResponse,
)
from skillforge.domain.structural_rules import REPEATED_OWNER_DELEGATION
from skillforge.integration.agent_skills import _render_skill
from skillforge.repository.delegation_guidance import delegation_decision_steps
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.static_delegation import (
    delegation_candidates,
    repeated_owner_delegation_candidates,
    structural_delegation_candidates,
)
from skillforge.repository.structural_rules import StructuralRuleMiner


def facts_for(tmp_path, caller, *, target=None):
    files = {
        "pkg/__init__.py": "",
        "pkg/target.py": target or """
            class Alpha:
                def one(self): return 1
                def two(self): return 'two'
            class Beta:
                def one(self): return 1
                def two(self): return 'two'
            def plain(): return 1
        """,
        "pkg/helpers.py": "def log(): pass\ndef count(): pass\n",
        "pkg/entry.py": caller,
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text), encoding="utf-8")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    assert not facts.issues
    return facts


PUBLIC = """
from pkg.target import Alpha
def first(client: Alpha) -> int:
    return client.one()
async def second(client: Alpha) -> str:
    return client.two()
"""


def test_top_level_different_return_types_have_real_support_and_scope(tmp_path):
    facts = facts_for(tmp_path, PUBLIC + "\ndef unused(): return 1\n")
    candidate, = structural_delegation_candidates(facts)
    assert candidate.delegation_pattern == REPEATED_OWNER_DELEGATION
    assert candidate.source_package == "pkg.entry"
    assert candidate.target_package == "pkg.target.Alpha"
    assert candidate.operation_symbols == ("pkg.entry.first", "pkg.entry.second")
    assert (candidate.rule.support.supporting, candidate.rule.support.examined) == (2, 3)
    assert candidate.rule.scope.selectors == ("pkg.entry",)
    assert candidate.operation_family == ()
    assert "pkg.entry.unused" in candidate.rule.statement
    assert delegation_candidates(facts) == ()
    refs = {item.evidence_id: item for item in facts.evidence}
    for name in candidate.operation_symbols:
        assert any(refs[key].description.startswith(f"Python call {name} -> ")
                   for key in candidate.rule.evidence_ids)
    again = PythonStaticAnalyzer().analyze(tmp_path)
    assert candidate == structural_delegation_candidates(again)[0]


@pytest.mark.parametrize("caller", [
    "from pkg.target import Alpha\ndef one(c: Alpha): return c.one()",
    "from pkg.target import Alpha, Beta\ndef one(c: Alpha): return c.one()\n"
    "def two(c: Beta): return c.two()",
    "from pkg.helpers import log, count\ndef one(): log(); count()\n"
    "def two(): log(); count()",
    "import external\ndef one(): return external.run()\ndef two(): return external.run()",
    "from pkg.target import Alpha\ndef _one(c: Alpha): return c.one()\n"
    "def two(c: Alpha): return c.two()",
    "from pkg.target import Alpha, Beta\ndef one(a: Alpha, b: Beta):\n"
    " return a.one(), b.one()\ndef two(a: Alpha, b: Beta):\n return a.two(), b.two()",
    "from pkg.target import Alpha\ndef one(c: Alpha):\n c.one()\n return c.two()",
    "from pkg.target import Alpha\ndef one(c: Alpha): c.one()\n"
    "def two(c: Alpha): c.two()",
    "from pkg.target import Alpha\ndef one(c: Alpha): return c._one()\n"
    "def two(c: Alpha): return c._two()",
])
def test_false_positive_guards(tmp_path, caller):
    assert repeated_owner_delegation_candidates(facts_for(tmp_path, caller)) == ()


def test_helpers_do_not_break_returned_delegation_or_supply_support(tmp_path):
    caller = PUBLIC.replace("from pkg.target import Alpha", "from pkg.target import Alpha\n"
                            "from pkg.helpers import log, count")
    caller = caller.replace("    return", "    log()\n    count()\n    return")
    caller += "\ndef helper_only(): log(); count()\n"
    candidate, = repeated_owner_delegation_candidates(facts_for(tmp_path, caller))
    assert candidate.rule.support.supporting == 2
    assert candidate.rule.support.examined == 3
    assert all("helper_only" not in name for name in candidate.operation_symbols)


def test_repeated_calls_count_one_operation(tmp_path):
    caller = PUBLIC.replace("    return client.one()", "    client.one()\n    client.one()\n"
                            "    return client.one()")
    candidate, = repeated_owner_delegation_candidates(facts_for(tmp_path, caller))
    assert candidate.rule.support.supporting == 2


def test_unique_side_effect_can_corroborate_but_ambiguous_cannot(tmp_path):
    caller = PUBLIC + """
def third(client: Alpha):
    client.one()
def fourth(client: Alpha):
    from pkg.helpers import log
    client.one()
    log()
"""
    candidate, = repeated_owner_delegation_candidates(facts_for(tmp_path, caller))
    assert candidate.operation_symbols == ("pkg.entry.first", "pkg.entry.second", "pkg.entry.third")
    assert (candidate.rule.support.supporting, candidate.rule.support.examined) == (3, 4)


def test_competing_repeated_owners_fail_closed(tmp_path):
    caller = PUBLIC + """
from pkg.target import Beta
def third(client: Beta): return client.one()
def fourth(client: Beta): return client.two()
"""
    assert repeated_owner_delegation_candidates(facts_for(tmp_path, caller)) == ()


def test_singleton_other_owner_is_explicitly_outside_observation(tmp_path):
    caller = PUBLIC + "\nfrom pkg.target import Beta\ndef other(c: Beta): return c.one()\n"
    candidate, = repeated_owner_delegation_candidates(facts_for(tmp_path, caller))
    assert candidate.rule.support.examined == 3
    assert "Not covered: pkg.entry.other" in candidate.rule.statement


def test_public_methods_and_module_target_owner(tmp_path):
    caller = """
from pkg.target import plain
class Entry:
    def first(self): return plain()
    async def second(self): return plain()
"""
    candidate, = repeated_owner_delegation_candidates(facts_for(tmp_path, caller))
    assert candidate.source_package == "pkg.entry.Entry"
    assert candidate.target_package == "pkg.target"


def test_missing_call_evidence_fails_closed(tmp_path):
    facts = facts_for(tmp_path, PUBLIC)
    with pytest.raises(ValueError, match="PythonCallEdge Evidence"):
        repeated_owner_delegation_candidates(replace(
            facts, calls=tuple(replace(edge, evidence_ids=()) for edge in facts.calls)))


def test_real_private_targets_are_excluded(tmp_path):
    target = "class Alpha:\n def _one(self): pass\n def _two(self): pass\n"
    caller = PUBLIC.replace("client.one()", "client._one()").replace(
        "client.two()", "client._two()")
    facts = facts_for(tmp_path, caller, target=target)
    assert len(facts.calls) == 2
    assert repeated_owner_delegation_candidates(facts) == ()


def test_private_sources_are_not_examined_and_input_order_is_irrelevant(tmp_path):
    facts = facts_for(tmp_path, PUBLIC + "\ndef _hidden(c: Alpha): return c.one()\n")
    candidate, = repeated_owner_delegation_candidates(facts)
    assert candidate.rule.support.examined == 2
    shuffled = replace(facts, calls=tuple(reversed(facts.calls)),
                       modules=tuple(reversed(facts.modules)),
                       evidence=tuple(reversed(facts.evidence)))
    assert repeated_owner_delegation_candidates(shuffled) == (candidate,)


def test_examined_population_cannot_be_tampered(tmp_path):
    mined = StructuralRuleMiner().mine(facts_for(tmp_path, PUBLIC + "\ndef unused(): pass\n"))
    candidate, = mined.candidates
    with pytest.raises(ValueError, match="complete repeated topology"):
        replace(candidate, examined_operation_symbols=candidate.operation_symbols)
    request = build_semantic_structural_request(mined, candidate)
    with pytest.raises(ValueError, match="examined module mismatch"):
        replace(request, examined_operation_symbols=())
    with pytest.raises(ValueError, match="invalid repeated owner"):
        replace(request, examined_operation_symbols=tuple(reversed(
            request.examined_operation_symbols)))


def test_legacy_chain_and_prompt_keep_precedence(tmp_path):
    from test_static_delegation import source

    mined = StructuralRuleMiner().mine(source(tmp_path))
    candidate, = mined.candidates
    assert candidate.delegation_pattern == ""
    assert len(candidate.delegation_topology) == 3
    request = build_semantic_structural_request(mined, candidate)
    assert "examined_operation_symbols" not in request.to_dict()
    with pytest.raises(ValueError, match="examined module mismatch"):
        replace(request, support=replace(request.support, examined=3))


def test_prepare_roundtrip_and_warm_cache_need_no_second_call(tmp_path):
    from test_incremental_evolution_v2 import TendencyModel
    from test_prepare import AntiPatternModel, WorkflowModel, _git
    from test_repo_snapshot_semantic_cache import RecordingModel

    from skillforge.domain.agent_integration import CodeAgentTarget
    from skillforge.productization.evolution_artifacts import load_baseline
    from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer

    facts_for(tmp_path, PUBLIC + "\ndef unused(): pass\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Offline Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "add", "pkg")
    _git(tmp_path, "commit", "-qm", "Initial public operations")
    cache = tmp_path.parent / (tmp_path.name + "-cache")
    request = PrepareRequest(tmp_path, "Extend analogous operations", CodeAgentTarget.CODEX,
                             current_module="pkg.entry")
    cold_model = RecordingModel()
    first = SkillForgePreparer(cold_model, WorkflowModel(), AntiPatternModel(), TendencyModel(),
                              semantic_cache_root=cache).prepare(request)
    baseline = load_baseline(tmp_path, first.evolution_baseline_path)
    warm_model = RecordingModel()
    second = SkillForgePreparer(warm_model, WorkflowModel(), AntiPatternModel(), TendencyModel(),
                               semantic_cache_root=cache).prepare(request)
    warm = load_baseline(tmp_path, second.evolution_baseline_path)
    assert first.structural_candidate_count == first.semantic_accepted_count == 1
    assert first.compiled_skill_count == 1 and len(first.selected_skill_ids) == 1
    assert cold_model.calls == 1 and warm_model.calls == 0
    assert baseline.source_dna == warm.source_dna
    assert baseline.source_compilation == warm.source_compilation
    assert baseline.source_dna.rules[0].support.examined == 3


@pytest.mark.parametrize("reject", [False, True])
def test_offline_semantic_dna_skill_registry_chain(tmp_path, reject):
    mined = StructuralRuleMiner().mine(facts_for(tmp_path, PUBLIC + "\ndef unused(): pass\n"))
    candidate, = mined.candidates

    class FakeSemantic:
        calls = 0

        def complete(self, request):
            self.calls += 1
            assert request.observation_kind == "static_delegation"
            assert request.delegation_topology == ("pkg.entry", "pkg.target.Alpha")
            assert request.support.examined == 3
            assert len(request.examined_operation_symbols) == 3
            refs = tuple(item.evidence_id for item in request.evidence)
            return SemanticStructuralResponse(
                request.repository_id, request.candidate_id,
                SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE if reject
                else SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
                "Current calls support a bounded owner observation", refs,
                request.source_package, request.target_package,
                () if reject else request.source_modules, () if reject else refs, (),
            )

    fixture = FakeSemantic()
    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, fixture)
    assert fixture.calls == 1
    assert not semantic.failures
    dna = ProjectDNABuilder().build(mined.repository_id, semantic_structural_rules=semantic)
    compiled = SkillCompiler().compile(dna, semantic)
    registry = SkillRegistryBuilder().build(compiled)
    if reject:
        assert len(semantic.rejections) == 1
        assert not dna.rules and not compiled.skills and not registry.skills
    else:
        assert len(semantic.accepted) == len(dna.rules) == len(registry.skills) == 1
        skill, = compiled.skills
        assert type(skill).__name__ == "RuleGuidanceSkill"
        assert skill.procedure.steps == delegation_decision_steps(candidate)
        assert skill.source_rule_id == dna.rules[0].rule_id
        rendered = _render_skill(skill, "fixture")
        assert "same observed family" in rendered and "[CONDITIONAL]" in rendered
