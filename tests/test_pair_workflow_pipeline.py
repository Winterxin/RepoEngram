"""Repeated real Git pairs use the unchanged semantic and compiler gates."""
import subprocess
from dataclasses import replace

import pytest
from test_semantic_workflows import FakeModel, _valid_response

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
)
from skillforge.domain.structural_rules import StructuralRuleMiningResult
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout


def _mine(root, occurrences):
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Pair test")
    _git(root, "config", "user.email", "pair@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    for index in range(occurrences + 1):
        (root / "implementation.py").write_text(f"VALUE = {index}\n")
        (root / "test_public.py").write_text(f"# public test {index}\n")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "Update implementation and public tests")
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner(ubiquitous_min_occurrences=2).mine(history, facts)
    candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    return history, facts, patterns, candidates


@pytest.mark.parametrize("occurrences", [0, 1, 2])
def test_real_history_requires_two_nonbaseline_commits(tmp_path, occurrences):
    history, facts, patterns, candidates = _mine(tmp_path, occurrences)
    assert bool(candidates.candidates) == (occurrences == 2)
    if occurrences == 2:
        assert len(candidates.candidates) == 2  # file and module observations
        for candidate in candidates.candidates:
            assert len(candidate.members) == 2
            source = next(p for p in patterns.patterns if p.pattern_id == candidate.seed_pattern_id)
            assert candidate.support == source.support
            assert candidate.support.examined == len(history.changesets)
            assert set(source.evidence_ids) <= set(candidate.evidence_ids)
        again = WorkflowCandidateMiner().mine(
            history, facts, replace(patterns, patterns=tuple(reversed(patterns.patterns))),
        )
        assert candidates == again


@pytest.mark.parametrize("decision", ["accept", "reject", "invalid"])
def test_pair_semantic_gate_dna_compiler_and_contract(tmp_path, decision):
    history, facts, _, candidates = _mine(tmp_path, 2)
    candidate = next(c for c in candidates.candidates if c.kind.value == "file_co_change")

    assert candidate.noise_signals  # noisy pairs still require the semantic gate

    def response(request):
        eid = request.evidence[0].evidence_id
        if decision == "reject":
            return SemanticDecisionResponse(
                request.repository_id, request.candidate_id,
                SemanticDecisionKind.INSUFFICIENT_EVIDENCE,
                "Co-change alone does not establish stable engineering intent",
                (eid,), 0.4, request.support, None,
            )
        workflow = _valid_response(request)
        workflow = replace(workflow, scope=SemanticWorkflowScope(
            SemanticWorkflowScopeKind.FILE,
            tuple(m.value for m in request.members),
            "Implementation and public tests", "Supplied historical files", (eid,),
        ))
        return workflow

    class Model(FakeModel):
        def complete(self, request):
            result = super().complete(request)
            if decision == "invalid":
                return replace(result, decision_evidence_ids=("unknown",))
            return result

    model = Model(response)
    assembly = evaluate_semantic_knowledge(
        history.repository_id, facts, StructuralRuleMiningResult(history.repository_id, (), ()),
        None, history=history, candidates=candidates, selected_workflows=(candidate,),
        workflow_model=model,
    )
    dna = ProjectDNABuilder().build(history.repository_id, workflows=assembly.workflows)
    compilation = SkillCompiler().compile(dna)
    assert len(dna.workflows) == len(compilation.skills) == (1 if decision == "accept" else 0)
    if decision == "invalid":
        assert assembly.failures[0]["status"] == "INVALID"
        assert len(model.requests) == 2  # unchanged production typed retry
    if decision == "accept":
        route = SkillRouter().route(
            SkillRegistryBuilder().build(compilation),
            SkillRoutingRequest(history.repository_id, "Coordinated implementation workflow"),
        )
        plan = AgentSkillExporter().plan(route, CodeAgentTarget.CODEX)
        assert plan.packages
        text = next(f.content for f in plan.packages[0].files if f.relative_path == "SKILL.md")
        assert "Define the shared contract" in text
        assert "Coordinate the related changes" in text
        assert "# Shared execution protocol" not in text
        assert "COMPLETION GATE" not in text
        assert "conditional observations" in text
