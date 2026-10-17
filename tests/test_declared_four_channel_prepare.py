"""Explicit four-channel observations preserve original gates and shared limits."""

import subprocess

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_cold_semantic_batching import twelve_inputs as twelve_inputs
from test_prepare import AntiPatternModel, FakeModel, WorkflowModel, _git
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_semantic_batch_transport import RecordedFixtureRunner, wire_response
from test_semantic_tendencies import response_for

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.change import ChangePatternKind
from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionKind
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.domain.semantic_tendency import TendencyDecisionKind
from skillforge.domain.semantic_workflow import SemanticDecisionKind
from skillforge.productization import prepare as p
from skillforge.productization.semantic_assembly import (
    build_semantic_antipattern_request,
    build_semantic_structural_request,
    build_semantic_tendency_request,
    build_semantic_workflow_request,
)
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository import semantic_execution
from skillforge.repository.history_patch_context import enrich_selected_history
from skillforge.repository.semantic_batch import _adapter

FOURTEEN = {"workflow": 3, "tendency": 2, "anti-pattern": 5, "structural": 4}


def test_explicit_four_channel_shape_keeps_original_limits_and_rejects_drift():
    budget = SemanticBudget(cold_shape=FOURTEEN)
    budget.validate_shape(FOURTEEN)
    assert budget.plan()["max_real_semantic_calls"] == 5
    assert budget.semantic_limit == 164.720042
    assert budget.prepare_limit == 349.862687
    for _ in range(5):
        budget.reserve()
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        budget.reserve()
    assert budget.calls == 5
    changed = {**FOURTEEN, "structural": 3}
    other = SemanticBudget(cold_shape=FOURTEEN)
    with pytest.raises(BudgetExceeded, match="differs"):
        other.validate_shape(changed)
    assert other.calls == 0
    with pytest.raises(BudgetExceeded, match="outside validated"):
        SemanticBudget().validate_shape(FOURTEEN)


def test_native_four_channel_cold_warm_and_checkpoint_preserve_gates(twelve_inputs, monkeypatch):
    root = twelve_inputs[0]
    for package in ("api", "service"):
        for i in range(3):
            path = root / f"src/app/{package}/added_{i}.py"
            path.write_text("VALUE = 1\n", "utf-8")
            _git(root, "add", "src")
            _git(root, "commit", "-qm", f"Introduce independent module {package}/{i}")
    facts = p.PythonStaticAnalyzer().analyze(root)
    history = p.GitHistoryMiner().mine(root, limit=p.HISTORY_LIMIT)
    patterns = p.ChangePatternMiner().mine(history, facts)
    candidates = p.WorkflowCandidateMiner().mine(history, facts, patterns)
    negative = p.NegativePatternMiner().mine(history)
    history = enrich_selected_history(root, history, facts, candidates, negative).history
    patterns = p.ChangePatternMiner().mine(history, facts)
    candidates = p.WorkflowCandidateMiner().mine(history, facts, patterns)
    negative = p.NegativePatternMiner().mine(history)
    mined = p.StructuralRuleMiner().mine(facts)
    requests = {name: [] for name in FOURTEEN}
    for candidate in p.select_semantic_candidates(candidates, limit=p.WORKFLOW_SEMANTIC_LIMIT):
        role = candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
        requests["tendency" if role else "workflow"].append(
            (build_semantic_tendency_request if role else build_semantic_workflow_request)(
                history, facts, candidates, candidate
            )
        )
    requests["anti-pattern"] = [
        build_semantic_antipattern_request(history, negative, candidate)
        for candidate in p.select_semantic_antipattern_candidates(
            negative, limit=p.ANTIPATTERN_SEMANTIC_LIMIT
        )
    ]
    requests["structural"] = [build_semantic_structural_request(mined, c) for c in mined.candidates]
    shape = {name: len(items) for name, items in requests.items()}
    assert all(shape.values()) and sum(shape.values()) > 11
    models = {
        "workflow": p.CodexCliSemanticModel(sandbox_mode="read-only"),
        "tendency": p.CodexCliTendencySemanticModel(sandbox_mode="read-only"),
        "anti-pattern": p.CodexCliAntiPatternSemanticModel(sandbox_mode="read-only"),
        "structural": p.CodexCliStructuralSemanticModel(sandbox_mode="read-only"),
    }
    responders = {
        "workflow": WorkflowModel(SemanticDecisionKind.INSUFFICIENT_EVIDENCE),
        "anti-pattern": AntiPatternModel(SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE),
        "structural": FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE),
    }
    records = {}
    for name, items in requests.items():
        for request in items:
            raw = wire_response(
                response_for(request, TendencyDecisionKind.INSUFFICIENT_EVIDENCE)
                if name == "tendency" else responders[name].complete(request)
            )
            if name == "tendency":
                raw.pop("repository_id")
                raw.pop("candidate_id")
                raw = {"decision": raw["decision"], "reason": raw["reason"],
                       "decision_evidence_id": request.evidence[0].evidence_id,
                       "citations": {
                           f"e{i}_id": None for i in range(len(request.structural_events))
                       }}
            _, prompt, _, _ = _adapter(models[name], request)
            records[prompt(request)] = request, raw
    runner = RecordedFixtureRunner(records)

    def offline(command, kwargs, timeout):
        assert timeout <= 164.720042 - 3
        return runner(command, **kwargs)

    monkeypatch.setattr(semantic_execution, "_execute_native", offline)
    preparer = p.SkillForgePreparer(
        models["structural"], models["workflow"], models["anti-pattern"], models["tendency"],
        semantic_cache_root=root / ".skillforge/test-cache",
    )
    request = p.PrepareRequest(root, "Update existing module behavior", CodeAgentTarget.CODEX)
    ledger = CostLedger()
    result = automatic_prepare(request, preparer=preparer, ledger=ledger, cold_shape=shape,
                               max_input_bytes=410000)
    assert len(runner.calls) == 1
    assert result.semantic_failure_count == 0
    assert result.workflow_rejection_count == shape["workflow"]
    assert result.tendency_rejection_count == shape["tendency"]
    assert result.anti_pattern_rejection_count == shape["anti-pattern"]
    assert result.semantic_rejection_count == shape["structural"]
    assert all(model._runner is subprocess.run for model in models.values())
    warm = automatic_prepare(request, preparer=preparer, ledger=ledger, cold_shape=shape)
    assert warm.dna_snapshot_id == result.dna_snapshot_id
    assert len(runner.calls) == 1
    assert ledger.events[-1]["cache"] == "hit"
    assert ledger.events[-1]["real_semantic_calls"] == 0
