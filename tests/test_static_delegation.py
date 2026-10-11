from dataclasses import replace
from pathlib import Path

import pytest

from skillforge.domain.change import ChangePatternResult, GitHistoryResult
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticStepClassification,
    SemanticWorkflowResponse,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.integration.agent_skills import _render_skill
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    build_semantic_workflow_request,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.static_delegation import delegation_candidates
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def source(tmp_path, *, operations=2, inconsistent=False, private=False, helper=False):
    package = tmp_path / "src" / "nebula"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    top = "from nebula.middle import Coordinator\nclass Entry:\n"
    top += " def __init__(self, delegate: Coordinator):\n  self.delegate = delegate\n"
    mid = "from nebula.store import Vault\nclass Coordinator:\n"
    mid += " def __init__(self, vault: Vault):\n  self.vault = vault\n"
    low = "class Vault:\n"
    for index in range(operations):
        name = ("_" if private else "") + f"change_{index}"
        call = f"self.delegate.{name}(item_key)"
        if inconsistent and index == operations - 1:
            call = "item_key"
        top += f" def {name}(self, item_key: str) -> str:\n"
        top += f"  {call}\n  return item_key\n" if helper else f"  return {call}\n"
        mid += f" def {name}(self, item_key: str) -> str:\n"
        mid += f"  result = self.vault.{name}(item_key)\n  return result\n"
        low += f" def {name}(self, item_key: str) -> str:\n  return item_key\n"
    for name, text in (("entry", top), ("middle", mid), ("store", low)):
        (package / f"{name}.py").write_text(text)
    return PythonStaticAnalyzer().analyze(tmp_path)


def inputs(facts):
    history = GitHistoryResult(facts.repository_id, (), ())
    patterns = ChangePatternResult(facts.repository_id, (), (), ())
    mined = WorkflowCandidateMiner().mine(history, facts, patterns)
    return history, mined


def test_repeated_topology_has_actual_symbol_call_and_binding_evidence(tmp_path):
    facts = source(tmp_path)
    candidate, = delegation_candidates(facts)
    assert candidate.support.supporting == 2
    assert candidate.supporting_commits == ()
    refs = {e.evidence_id: e for e in facts.evidence}
    cited = [refs[key] for key in candidate.evidence_ids]
    assert sum(e.description.startswith("Python call ") for e in cited) == 4
    assert any("receiver binding" in e.description for e in cited)
    for e in cited:
        assert (tmp_path / e.location.path).is_file()
        assert e.location.line is not None
    assert all(edge.target_symbol in {s.qualified_name for m in facts.modules for s in m.symbols}
               for edge in facts.calls)
    assert candidate == delegation_candidates(PythonStaticAnalyzer().analyze(tmp_path))[0]
    assert "must" not in candidate.observation.lower()


@pytest.mark.parametrize("options", [
    {"operations": 1}, {"inconsistent": True}, {"private": True}, {"helper": True},
])
def test_single_inconsistent_private_or_nonreturning_helper_is_not_rule(tmp_path, options):
    assert delegation_candidates(source(tmp_path, **options)) == ()


def test_different_parameter_families_do_not_supply_independent_support(tmp_path):
    source(tmp_path)
    file = tmp_path / "src/nebula/entry.py"
    file.write_text(file.read_text().replace("change_1(self, item_key: str)",
                                           "change_1(self, item_key: int)"))
    assert delegation_candidates(PythonStaticAnalyzer().analyze(tmp_path)) == ()


def test_candidate_tampering_is_rejected_before_model(tmp_path):
    facts = source(tmp_path)
    history, mined = inputs(facts)
    candidate, = mined.candidates
    with pytest.raises(ValueError, match="current source facts"):
        build_semantic_workflow_request(history, facts, mined,
                                        replace(candidate, observation="invented"))


def decision(request, *, reject=False):
    refs = tuple(e.evidence_id for e in request.evidence)
    workflow = None
    if not reject:
        step = SemanticWorkflowStep("implement", "Inspect analogous operations and preserve "
                                    "their observed delegation for new operations of this family",
                                    refs)
        scope = SemanticWorkflowScope(SemanticWorkflowScopeKind.MODULE,
                                      request.current_modules, "Observed delegation family",
                                      "Bounded to the observed modules", refs)
        workflow = SemanticWorkflowResponse(
            request.repository_id, request.candidate_id, "Extend delegated operations", refs,
            "Add a new operation following the observed delegation family", refs, (), (step,),
            (), (SemanticStepClassification("implement", "Current structural examples", refs),),
            (), scope, (), 0.9,
        )
    return SemanticDecisionResponse(
        request.repository_id, request.candidate_id,
        SemanticDecisionKind.INCOHERENT if reject else SemanticDecisionKind.WORKFLOW,
        "Current implementations provide the structural examples", refs, 0.9,
        request.support, workflow,
    )


@pytest.mark.parametrize("reject", [False, True])
def test_existing_semantic_gate_dna_compiler_and_contract(tmp_path, reject):
    facts = source(tmp_path)
    history, mined = inputs(facts)

    class Model:
        def complete(self, request):
            assert request.kind.value == "static_delegation"
            assert request.supporting_commits == ()
            return decision(request, reject=reject)

    semantic = SemanticWorkflowAbstracter().abstract(history, facts, mined, Model())
    dna = ProjectDNABuilder().build(facts.repository_id, workflows=semantic)
    compiled = SkillCompiler().compile(dna)
    if reject:
        assert len(semantic.rejections) == 1
        assert dna.workflows == () and compiled.skills == ()
    else:
        assert len(dna.workflows) == 1
        skill, = compiled.skills
        rendered = _render_skill(skill, "fixture")
        assert "Decision procedure / Steps" in rendered
        assert "observed delegation" in rendered
        assert skill.procedure.steps == skill.steps
        assert "[OPTIONAL]" in rendered


def test_production_has_no_benchmark_specific_names():
    root = Path(__file__).resolve().parents[1] / "src/skillforge/repository"
    for name in ("static_calls.py", "static_delegation.py"):
        text = (root / name).read_text()
        for token in ("C05", "catalog", "remove_product", "rename_product", "feature_product"):
            assert token not in text
