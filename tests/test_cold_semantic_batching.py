"""Offline cold transport counts and full semantic/artifact equality."""

from dataclasses import replace

import pytest
from test_prepare import AntiPatternModel, FakeModel, WorkflowModel, _git
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import complete_identity

from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.domain.semantic_workflow import SemanticDecisionKind
from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.history_patch_context import enrich_selected_history
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_antipattern_evaluation import (
    select_semantic_antipattern_candidates,
)
from skillforge.repository.semantic_batch import request_id
from skillforge.repository.semantic_evaluation import select_semantic_candidates
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


@pytest.fixture
def eleven_inputs(historical_repository):
    root = historical_repository
    # Independent repeated collaborations produce distinct real miner candidates.
    for group in range(3):
        for occurrence in range(2):
            for relative in (f"src/app/api/route{group}.py",
                             f"src/app/service/unit{group % 2}.py"):
                path = root / relative
                path.write_text(path.read_text(encoding="utf-8") +
                                f"PAIR_{group}_{occurrence} = 1\n", encoding="utf-8")
            _git(root, "add", "src")
            _git(root, "commit", "-qm", f"Coordinate behavior {group}/{occurrence}")
    for index in range(2):
        path = root / f"src/app/service/unit{index}.py"
        path.write_text(path.read_text(encoding="utf-8") + "VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Fix service behavior")
    facts = PythonStaticAnalyzer().analyze(root)
    history = GitHistoryMiner().mine(root, limit=50)
    candidates = WorkflowCandidateMiner().mine(
        history, facts, ChangePatternMiner().mine(history, facts),
    )
    # Match ordinary prepare's bounded history before recording exact input fixtures.
    history = enrich_selected_history(
        root, history, facts, candidates, NegativePatternMiner().mine(history),
    ).history
    candidates = WorkflowCandidateMiner().mine(
        history, facts, ChangePatternMiner().mine(history, facts),
    )
    negative = NegativePatternMiner().mine(history)
    selected = select_semantic_candidates(candidates, limit=5)
    anti = select_semantic_antipattern_candidates(negative, limit=5)
    mined = StructuralRuleMiner().mine(facts)
    assert (len(selected), len(anti), len(mined.candidates)) == (5, 5, 1)
    return root, facts, history, candidates, negative, selected, anti, mined


@pytest.fixture
def twelve_inputs(eleven_inputs):
    root = eleven_inputs[0]
    data = root / "src/app/data"
    data.mkdir()
    (data / "__init__.py").write_text("")
    for index in range(2):
        (data / f"unit{index}.py").write_text("VALUE = 1\n")
    for index in range(3):
        path = root / f"src/app/api/route{index}.py"
        path.write_text(path.read_text("utf-8") + f"import app.data.unit{index % 2}\n", "utf-8")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Observe another independent package dependency")
    facts = PythonStaticAnalyzer().analyze(root)
    history = GitHistoryMiner().mine(root, limit=50)
    candidates = WorkflowCandidateMiner().mine(
        history, facts, ChangePatternMiner().mine(history, facts),
    )
    history = enrich_selected_history(
        root, history, facts, candidates, NegativePatternMiner().mine(history),
    ).history
    candidates = WorkflowCandidateMiner().mine(
        history, facts, ChangePatternMiner().mine(history, facts),
    )
    negative = NegativePatternMiner().mine(history)
    selected = select_semantic_candidates(candidates, limit=5)
    anti = select_semantic_antipattern_candidates(negative, limit=5)
    mined = StructuralRuleMiner().mine(facts)
    assert (len(selected), len(anti), len(mined.candidates)) == (5, 5, 2)
    return root, facts, history, candidates, negative, selected, anti, mined


class SingleFixture:
    def __init__(self, responder, mode="success"):
        self.responder, self.mode = responder, mode
        self.calls, self.batch_calls = 0, 0
        self.requests = []
        self.target = None
        self.semantic_cache_identity = complete_identity()
        self.semantic_cache_identity["response_parameters"]["mode"] = mode

    def response(self, request):
        self.requests.append(request)
        self.target = self.target or request_id(request)
        response = self.responder.complete(request)
        if request_id(request) != self.target:
            return response
        if self.mode == "malformed":
            raise ValueError("malformed fixture response")
        if self.mode == "transport_failure":
            raise RuntimeError("fixture transport unavailable")
        if self.mode in {"gate_failure", "batch_malformed_single_gate_failure"}:
            return replace(response, decision_evidence_ids=("unknown",))
        if self.mode == "gate_rejection":
            return replace(response, decision=SemanticDecisionKind.INCOHERENT, workflow=None)
        return response

    def complete(self, request):
        self.calls += 1
        return self.response(request)


class BatchFixture(SingleFixture):
    def complete_batch(self, requests):
        self.batch_calls += 1
        responses = {}
        for index, request in enumerate(requests):
            if self.mode == "partial" and index == 0:
                continue
            try:
                responses[request_id(request)] = self.response(request)
                if (self.mode == "batch_malformed_single_gate_failure"
                        and request_id(request) == self.target):
                    responses[request_id(request)] = ValueError("malformed batch only")
            except (ValueError, RuntimeError) as error:
                responses[request_id(request)] = error
        return responses


def run(inputs, model_type, mode="success", cache_root=None):
    root, facts, history, candidates, negative, selected, anti, mined = inputs
    models = (
        model_type(FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)),
        model_type(WorkflowModel(), mode), model_type(AntiPatternModel()),
    )
    cache = RepoSnapshotSemanticCache(root, cache_root) if cache_root else None
    wrapped = tuple(cache.wrap(model) for model in models) if cache else models
    structural, workflow, anti_pattern = wrapped
    assembly = evaluate_semantic_knowledge(
        facts.repository_id, facts, mined, structural, history=history,
        candidates=candidates, selected_workflows=selected,
        negative=negative, selected_anti_patterns=anti,
        workflow_model=workflow, anti_pattern_model=anti_pattern,
    )
    if cache:
        cache.publish(assembly.workflows, assembly.tendencies, assembly.anti_patterns,
                      assembly.structural)
    dna = ProjectDNABuilder().build(
        facts.repository_id, workflows=assembly.workflows, anti_patterns=assembly.anti_patterns,
        semantic_structural_rules=assembly.structural, semantic_tendencies=assembly.tendencies,
    )
    compiled = SkillCompiler().compile(dna, assembly.structural, assembly.tendencies)
    registry = SkillRegistryBuilder().build(compiled)
    routes = tuple(SkillRouter().route(registry, SkillRoutingRequest(
        facts.repository_id, task, current_module="app.api.route0",
    )) for task in ("update routes", "fix service behavior"))
    return assembly, (dna, compiled, registry, routes), models, cache


def assert_equal(single, batch):
    a, artifacts_a, models_a, _ = single
    b, artifacts_b, models_b, _ = batch
    assert (a.workflows, a.tendencies, a.anti_patterns, a.structural) == (
        b.workflows, b.tendencies, b.anti_patterns, b.structural,
    )
    assert [(f["channel"], f["candidate_id"], f["status"]) for f in a.failures] == [
        (f["channel"], f["candidate_id"], f["status"]) for f in b.failures
    ]
    assert artifacts_a == artifacts_b
    # Full requests include candidate identity and all original closed Evidence.
    assert {request_id(r) for m in models_a for r in m.requests} == {
        request_id(r) for m in models_b for r in m.requests
    }


@pytest.mark.parametrize("mode,old,new,fallback", [
    ("success", 11, 3, 0),
    ("malformed", 11, 4, 1),
    ("gate_rejection", 11, 3, 0),
    ("gate_failure", 12, 4, 1),
    ("batch_malformed_single_gate_failure", 12, 5, 2),
    ("transport_failure", 11, 4, 1),
    ("partial", 11, 4, 1),
])
def test_eleven_candidates_equal_and_isolated(eleven_inputs, mode, old, new, fallback):
    single = run(eleven_inputs, SingleFixture, mode)
    batch = run(eleven_inputs, BatchFixture, mode)
    assert_equal(single, batch)
    assert sum(m.calls for m in single[2]) == old
    assert sum(m.calls + m.batch_calls for m in batch[2]) == new
    assert sum(m.calls for m in batch[2]) == fallback
    if mode in {"malformed", "gate_failure", "transport_failure",
                "batch_malformed_single_gate_failure"}:
        assert len(batch[0].failures) == 1
        assert len(batch[0].workflows.workflows) == 4
    else:
        assert not batch[0].failures


def test_cold_batch_warm_exact_cache(eleven_inputs, tmp_path):
    directory = tmp_path.parent / (tmp_path.name + "-batch-cache")
    cold = run(eleven_inputs, BatchFixture, cache_root=directory)
    warm = run(eleven_inputs, BatchFixture, cache_root=directory)
    assert cold[:2] == warm[:2]
    assert sum(m.batch_calls for m in cold[2]) == 3
    assert sum(m.calls + m.batch_calls for m in warm[2]) == 0
    assert len(list(directory.glob("*.json"))) == warm[3].hits == 11


def test_failed_candidate_not_cached_and_warm_successes_retained(eleven_inputs, tmp_path):
    directory = tmp_path.parent / (tmp_path.name + "-failure-cache")
    cold = run(eleven_inputs, BatchFixture, "gate_failure", directory)
    warm = run(eleven_inputs, BatchFixture, "gate_failure", directory)
    assert cold[:2] == warm[:2]
    assert len(list(directory.glob("*.json"))) == 10
    assert warm[3].hits == 10
    assert sum(m.batch_calls for m in warm[2]) == 1
    assert sum(m.calls for m in warm[2]) == 1


def test_incomplete_identity_uses_single_path(eleven_inputs):
    class OpaqueBatch(BatchFixture):
        def __init__(self, *args):
            super().__init__(*args)
            self.semantic_cache_identity = {"model": "fixture"}

    result = run(eleven_inputs, OpaqueBatch)
    assert sum(m.batch_calls for m in result[2]) == 0
    assert sum(m.calls for m in result[2]) == 11


def test_unknown_partial_key_cannot_erase_known_successes(eleven_inputs):
    class UnknownSibling(BatchFixture):
        def complete_batch(self, requests):
            responses = super().complete_batch(requests)
            responses["unproven-request"] = ValueError("unknown sibling")
            return responses

    result = run(eleven_inputs, UnknownSibling)
    assert not result[0].failures
    assert sum(m.batch_calls for m in result[2]) == 3
    assert sum(m.calls for m in result[2]) == 0
