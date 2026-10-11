from dataclasses import replace

import pytest
from test_static_delegation import source

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralResponse,
)
from skillforge.domain.structural_rules import TESTING_METHOD
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_rules import SemanticStructuralRuleEvaluator
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


def fixture(root, *, operations=2, tested=2, internal=False, unrelated=False,
            generated=False, snapshot=False, assigned=False):
    source(root, operations=operations)
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    text = "# generated file\n" if generated else ""
    text += "import unittest\nfrom nebula.entry import Entry as Gateway\n"
    text += "from nebula.store import Vault\nfrom nebula.middle import Coordinator\n"
    text += "class Behavior(unittest.TestCase):\n def setUp(self):\n"
    ctor = "Vault()" if internal else "Gateway(Coordinator(Vault()))"
    text += f"  self.subject = {ctor}\n"
    for i in range(tested):
        text += f" def test_behavior_{i}(self):\n"
        call = f"self.subject.change_{i}('x')"
        if unrelated:
            text += f"  {call}\n  self.assertEqual(1, 1)\n"
        elif snapshot:
            text += f"  self.assertSnapshot({call})\n"
        elif assigned:
            text += f"  result = {call}\n  self.assertEqual(result, 'x')\n"
        else:
            text += f"  self.assertEqual({call}, 'x')\n"
    (tests / "test_behavior.py").write_text(text)
    return PythonStaticAnalyzer().analyze(root)


def _testing(facts):
    return tuple(c for c in StructuralRuleMiner().mine(facts).candidates
                 if c.method == TESTING_METHOD)


@pytest.mark.parametrize("assigned", [False, True])
def test_repeated_public_assertion_linkage_is_deterministic_and_real(tmp_path, assigned):
    facts = fixture(tmp_path, assigned=assigned)
    candidate, = _testing(facts)
    assert candidate == _testing(PythonStaticAnalyzer().analyze(tmp_path))[0]
    assert candidate.rule.support.supporting == candidate.rule.support.examined == 2
    symbols = {s.qualified_name for m in facts.modules for s in m.symbols}
    evidence = {e.evidence_id: e for e in facts.evidence}
    assert len(candidate.test_linkages) == 2
    for link in candidate.test_linkages:
        assert link.operation_symbol in symbols and link.test_symbol in symbols
        assert (tmp_path / link.path).is_file()
        refs = [evidence[key] for key in link.evidence_ids]
        for prefix in ("Python implementation", "Python public test call",
                       "Python behavior assertion", "Python test receiver binding"):
            assert any(e.description.startswith(prefix) for e in refs)
        assert all(e.location.line for e in refs)


@pytest.mark.parametrize("options", [
    {"operations": 1, "tested": 1}, {"tested": 1}, {"internal": True},
    {"unrelated": True}, {"operations": 3}, {"generated": True}, {"snapshot": True},
])
def test_insufficient_internal_unlinked_or_contradictory_has_no_testing_rule(tmp_path, options):
    assert _testing(fixture(tmp_path, **options)) == ()


def test_incidental_strings_are_not_calls(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "tests/test_behavior.py"
    path.write_text("def test_strings():\n assert 'change_0 change_1'\n")
    assert _testing(PythonStaticAnalyzer().analyze(tmp_path)) == ()


def test_private_helper_is_not_public_convention(tmp_path):
    fixture(tmp_path)
    for path in tmp_path.rglob("*.py"):
        path.write_text(path.read_text().replace("change_", "_change_"))
    assert _testing(PythonStaticAnalyzer().analyze(tmp_path)) == ()


def test_overwritten_receiver_has_no_linkage(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "tests/test_behavior.py"
    path.write_text(path.read_text().replace("  self.assertEqual", "  self.subject = unknown()\n"
                                           "  self.assertEqual"))
    assert _testing(PythonStaticAnalyzer().analyze(tmp_path)) == ()


class Model:
    def __init__(self, reject=False, missing=False):
        self.reject = reject
        self.missing = missing

    def complete(self, request):
        refs = tuple(e.evidence_id for e in request.evidence
                     if not self.missing or not e.description.startswith(
                         "Python behavior assertion"))
        return SemanticStructuralResponse(
            request.repository_id, request.candidate_id,
            SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE if self.reject else
            SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
            "Current public calls and assertions support a conditional convention",
            refs, request.source_package, request.target_package,
            () if self.reject else request.source_modules, () if self.reject else refs, (),
        )


@pytest.mark.parametrize("reject", [False, True])
def test_semantic_accept_reject_dna_compiler_and_delivery(tmp_path, reject):
    facts = fixture(tmp_path)
    mined = StructuralRuleMiner().mine(facts)
    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, Model(reject=reject))
    assert not semantic.failures
    dna = ProjectDNABuilder().build(facts.repository_id, semantic_structural_rules=semantic)
    compilation = SkillCompiler().compile(dna, semantic)
    if reject:
        assert not dna.rules and not compilation.skills
        return
    assert len(dna.rules) == len(compilation.skills) == 2
    registry = SkillRegistryBuilder().build(compilation)
    route = SkillRouter().route(registry, SkillRoutingRequest(
        facts.repository_id, "Add a public operation", current_file="src/nebula/entry.py"))
    plan = AgentSkillExporter().plan(route, CodeAgentTarget.CODEX)
    assert len(plan.packages) == 2
    for package in plan.packages:
        text = next(f.content for f in package.files if f.relative_path == "SKILL.md")
        for phrase in ("CONDITIONAL REQUIRED", "ACTION: Add or update", "DONE WHEN:",
                       "persistent repository test exists", "actually run and passes",
                       "temporary shell behavior probe does NOT satisfy", "SKIP ONLY IF:",
                       "regression tests alone do NOT satisfy", "owning implementation layer",
                       "tests/test_behavior.py", "test_behavior_0", "test_behavior_1"):
            assert phrase in text


def test_semantic_acceptance_cannot_omit_assertion_evidence(tmp_path):
    facts = fixture(tmp_path)
    result = SemanticStructuralRuleEvaluator().evaluate(
        StructuralRuleMiner().mine(facts), Model(missing=True))
    assert len(result.failures) == 1
    assert "cited behavior linkage" in result.failures[0].reason
    assert all(a.candidate.method != TESTING_METHOD for a in result.accepted)


def test_candidate_cannot_claim_single_or_missing_test_linkage(tmp_path):
    candidate, = _testing(fixture(tmp_path))
    with pytest.raises(ValueError, match="public behavior linkage"):
        replace(candidate, test_linkages=candidate.test_linkages[:1])


def test_pytest_direct_public_constructor_and_assert(tmp_path):
    source(tmp_path)
    path = tmp_path / "test_public.py"
    path.write_text("from nebula.entry import Entry\n"
                    "from nebula.middle import Coordinator\nfrom nebula.store import Vault\n"
                    "def test_first():\n subject = Entry(Coordinator(Vault()))\n"
                    " assert subject.change_0('x') == 'x'\n"
                    "def test_second():\n subject = Entry(Coordinator(Vault()))\n"
                    " assert subject.change_1('x') == 'x'\n")
    assert len(_testing(PythonStaticAnalyzer().analyze(tmp_path))) == 1


@pytest.mark.parametrize("framework", ["unittest", "pytest"])
def test_public_failure_behavior_is_linked_with_assertion_source(tmp_path, framework):
    fixture(tmp_path)
    path = tmp_path / "tests/test_behavior.py"
    text = path.read_text()
    if framework == "unittest":
        text += "\n def test_failure(self):\n  with self.assertRaises(KeyError):\n"
        text += "   self.subject.change_0('missing')\n"
    else:
        text += "\nimport pytest\ndef test_failure():\n"
        text += " subject = Gateway(Coordinator(Vault()))\n with pytest.raises(KeyError):\n"
        text += "  subject.change_0('missing')\n"
    path.write_text(text)
    candidate, = _testing(PythonStaticAnalyzer().analyze(tmp_path))
    link, = (r for r in candidate.test_linkages if r.assertion_kind == "raises")
    assert link.call_line > link.assertion_line


def test_fake_linkage_source_location_is_rejected(tmp_path):
    facts = fixture(tmp_path)
    with pytest.raises(ValueError, match="location mismatch"):
        StructuralRuleMiner().mine(replace(facts, test_linkages=(
            replace(facts.test_linkages[0], call_line=999), *facts.test_linkages[1:])))
