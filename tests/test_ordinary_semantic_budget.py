"""Real production entry, offline transport: one shared native invocation budget."""

import json
import subprocess
import sys

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_semantic_batch_transport import recorded_models

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository import semantic_execution as execution
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel


def native_fixture(inputs, monkeypatch, mode="success"):
    _, runner, prototypes = recorded_models(inputs, mode)
    models = tuple(type(m)(runner=subprocess.run, sandbox_mode="read-only") for m in prototypes)
    models += (CodexCliTendencySemanticModel(sandbox_mode="read-only"),)
    calls = []

    def execute(command, kwargs, timeout):
        calls.append((command, timeout))
        return runner(command, **kwargs)

    monkeypatch.setattr(execution, "_execute_native", execute)
    return models, runner, calls


def test_ordinary_native_cold_warm_shared_budget_and_attributes_restored(
    eleven_inputs, monkeypatch
):
    models, runner, calls = native_fixture(eleven_inputs, monkeypatch)
    ledger = CostLedger()
    preparer = SkillForgePreparer(*models)
    request = PrepareRequest(eleven_inputs[0], "update routes", CodeAgentTarget.CODEX)
    cold = automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert len(calls) == 1 and "--json" in calls[0][0]
    assert calls[0][1] <= 164.720042 - 3
    assert all(not hasattr(m, "effect_budget") for m in models)
    warm = automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert len(runner.calls) == 1 and cold.dna_snapshot_id == warm.dna_snapshot_id
    assert ledger.events[-1]["cache"] == "hit"
    assert ledger.events[-1]["real_semantic_calls"] == 0
    assert len([e for e in ledger.events if e["kind"] == "semantic"]) == 1


def test_ordinary_native_failed_cohort_forbids_sixth_spawn(eleven_inputs, monkeypatch):
    models, runner, calls = native_fixture(eleven_inputs, monkeypatch, "batch_failure")
    ledger = CostLedger()
    # A failing cohort consumes the same budget as original single fallbacks.
    # Corrupt singles to keep the original one-decision repair path active.
    runner.corrupt = lambda response: {**response, "decision_evidence_ids": ["unknown"]}
    runner.target_channel = "workflow"
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        automatic_prepare(
            PrepareRequest(eleven_inputs[0], "update routes", CodeAgentTarget.CODEX),
            preparer=SkillForgePreparer(*models),
            ledger=ledger,
        )
    assert len(calls) == 5
    assert ledger.events[-1]["kind"] == "budget-stop"
    assert ledger.events[-1]["classification"] == "INVALID_BUDGET"
    assert not (
        eleven_inputs[0] / ".skillforge/productization/prepare/verified-baseline.json"
    ).exists()


def test_native_timeout_is_fatal_and_not_retried(eleven_inputs, monkeypatch):
    models, _, calls = native_fixture(eleven_inputs, monkeypatch)

    def timeout(command, kwargs, remaining):
        calls.append(command)
        raise BudgetExceeded("semantic deadline reached: INVALID_BUDGET")

    monkeypatch.setattr(execution, "_execute_native", timeout)
    with pytest.raises(BudgetExceeded, match="deadline"):
        automatic_prepare(
            PrepareRequest(eleven_inputs[0], "update routes", CodeAgentTarget.CODEX),
            preparer=SkillForgePreparer(*models),
        )
    assert len(calls) == 1


def test_native_cache_bypass_before_any_model(eleven_inputs, monkeypatch):
    models, _, calls = native_fixture(eleven_inputs, monkeypatch)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(BudgetExceeded, match="cache bypass"):
        automatic_prepare(
            PrepareRequest(eleven_inputs[0], "update routes", CodeAgentTarget.CODEX),
            preparer=SkillForgePreparer(*models),
        )
    assert not calls


def test_native_semantic_time_accumulates_and_usage_is_recorded(tmp_path, monkeypatch):
    model = CodexCliTendencySemanticModel(sandbox_mode="read-only")
    clock = [0.0]
    budget = SemanticBudget(semantic_seconds=4, clock=lambda: clock[0])
    monkeypatch.setattr(execution, "SemanticBudget", lambda: budget)
    command = ["codex", "exec", "--output-schema", str(tmp_path / "schema.json")]
    (tmp_path / "schema.json").write_text("{}")
    ledger = CostLedger()
    usages = {"input_tokens": 12, "cached_input_tokens": 0, "output_tokens": 3}

    def execute(argv, kwargs, remaining):
        budget.semantic_seconds = 4
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"type": "turn.completed", "usage": usages}), ""
        )

    monkeypatch.setattr(execution, "_execute_native", execute)
    with pytest.raises(BudgetExceeded, match="cumulative"):
        with execution.ordinary_governance([model], tmp_path, ledger):
            execution.run_semantic(model, command, input="fixture", cwd=tmp_path, timeout=180)
    assert ledger.events[-2]["usage"] == [usages]
    assert budget.calls == 1


def test_native_process_family_deadline_without_model(tmp_path):
    script = (
        "import subprocess,sys,time,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "pathlib.Path('child.pid').write_text(str(child.pid)); time.sleep(60)"
    )
    with pytest.raises(BudgetExceeded, match="deadline"):
        execution._execute_native([sys.executable, "-c", script], {"cwd": tmp_path, "input": ""}, 2)
    pid = (tmp_path / "child.pid").read_text()
    if sys.platform == "win32":
        listing = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/NH"])
        assert pid.encode() not in listing


def test_cli_budget_stop_is_bounded_explanatory_json(tmp_path, monkeypatch, capsys):
    from skillforge import cli

    def stopped(*args, **kwargs):
        raise BudgetExceeded("real semantic call budget exhausted BEFORE next spawn")

    monkeypatch.setattr("skillforge.productization.verified_prepare.automatic_prepare", stopped)
    assert (
        cli.main(["prepare", str(tmp_path), "--task", "task", "--target", "codex", "--json"]) == 2
    )
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["status"] == "INVALID_BUDGET"


def test_unconfirmed_native_cleanup_remains_fatal(tmp_path, monkeypatch):
    class Child:
        pid = 12345

        def communicate(self, *args, **kwargs):
            raise subprocess.TimeoutExpired("fixture", 0.1)

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Child())

    def failed_cleanup(*args, **kwargs):
        raise OSError("fixture cleanup unavailable")

    if sys.platform == "win32":
        monkeypatch.setattr(subprocess, "run", failed_cleanup)
    else:
        monkeypatch.setattr(execution.os, "killpg", failed_cleanup)
    with pytest.raises(BudgetExceeded, match="cleanup unconfirmed"):
        execution._execute_native(["fixture"], {"cwd": tmp_path, "input": ""}, 0.1)
