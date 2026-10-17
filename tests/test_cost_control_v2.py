"""No-model integration checks for the actual user and coordinator entries."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import sandbox_state
import u07_cost_control as campaign
from test_effect_cost_control import actual_models
from test_effect_cost_control import eleven_inputs as eleven_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_verify_runner import check, route
from test_verify_runner import repo as repo

from skillforge.cost_control import BudgetExceeded
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.verify import IndependentVerifier, VerifyPlanBuilder


def state(path, password):
    for relative in sandbox_state.FILES:
        (path / relative).parent.mkdir(parents=True, exist_ok=True)
    (path / sandbox_state.FILES[0]).write_text(
        json.dumps(
            {
                "version": 5,
                "offline_username": "CodexSandboxOffline",
                "online_username": "CodexSandboxOnline",
            }
        )
    )
    (path / sandbox_state.FILES[1]).write_text(
        json.dumps(
            {
                "version": 5,
                "offline": {"username": "CodexSandboxOffline", "password": password},
                "online": {"username": "CodexSandboxOnline", "password": password},
            }
        )
    )


def test_stale_global_record_rejected_healthy_existing_selected(tmp_path):
    authority, target, healthy = [tmp_path / n for n in ("global", "target", "healthy")]
    state(authority, "old-fixture")
    state(target, "old-fixture")
    state(healthy, "new-fixture")
    def login(r):
        return {
            "ok": r["password"] == "new-fixture",
            "win_error": 0 if r["password"] == "new-fixture" else 1326,
        }
    result = sandbox_state.recover(
        target,
        authority=authority,
        candidates=[healthy],
        evidence=tmp_path / "evidence",
        login=login,
    )
    assert result["source"] == str(healthy)
    assert result["rejected"][0]["checks"]["offline"]["win_error"] == 1326
    assert sandbox_state.health(target, login)["healthy"]
    assert "new-fixture" not in json.dumps(result)
    assert "old-fixture" not in json.dumps(result)
    assert not (authority / ".sandbox/skillforge-state.lock").exists()


def test_invalid_accounts_abort_without_overwriting_or_provisioning(tmp_path):
    authority, target = tmp_path / "authority", tmp_path / "target"
    state(authority, "bad")
    state(target, "existing")
    before = sandbox_state.fingerprint(target)
    with pytest.raises(RuntimeError, match="no healthy"):
        sandbox_state.recover(
            target,
            authority=authority,
            candidates=[],
            evidence=tmp_path / "out",
            login=lambda r: {"ok": False, "win_error": 1326},
        )
    assert sandbox_state.fingerprint(target) == before


def test_default_product_entry_reuses_without_mining_or_transport(
    eleven_inputs, tmp_path, monkeypatch
):
    models, transport, budget, ledger = actual_models(eleven_inputs, tmp_path)
    tendency = transport.bind(CodexCliTendencySemanticModel(runner=transport))
    preparer = SkillForgePreparer(
        *models, tendency, semantic_cache_root=tmp_path / ".skillforge/cache"
    )
    request = PrepareRequest(eleven_inputs[0], "update routes", CodeAgentTarget.CODEX)
    started = time.monotonic()
    cold = automatic_prepare(request, preparer=preparer, ledger=ledger)
    cold_seconds = time.monotonic() - started
    assert 3 <= budget.calls <= 5
    calls = budget.calls
    from skillforge.repository.git_history import GitHistoryMiner

    monkeypatch.setattr(
        GitHistoryMiner, "mine", lambda *a, **k: pytest.fail("warm rebuilt history")
    )
    started = time.monotonic()
    warm = automatic_prepare(request, preparer=preparer, ledger=ledger)
    warm_seconds = time.monotonic() - started
    assert budget.calls == calls
    assert warm.dna_snapshot_id == cold.dna_snapshot_id
    assert warm.selected_skill_ids == cold.selected_skill_ids
    assert warm.semantic_failures == cold.semantic_failures
    assert ledger.events[-1]["real_semantic_calls"] == 0
    print(
        json.dumps(
            {
                "fixture_cold_seconds": cold_seconds,
                "fixture_warm_seconds": warm_seconds,
                "cold_calls": calls,
                "warm_calls": 0,
                "warm_history_mines": 0,
            }
        )
    )
    (request.path / "source-drift.py").write_text("changed = 1")
    with pytest.raises(BudgetExceeded, match="drift"):
        automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert budget.calls == calls


def test_cli_prepare_uses_automatic_entry(tmp_path, monkeypatch):
    import skillforge.productization.verified_prepare as entry
    from skillforge import cli

    class Result:
        target = CodeAgentTarget.CODEX
        semantic_failure_count = 0
        evolution_baseline_path = None
        evolution_head_commit = None

        def to_dict(self):
            return {"mode": "tested-auto"}

    observed = []
    monkeypatch.setattr(
        entry, "automatic_prepare", lambda request, **kw: observed.append((request, kw)) or Result()
    )
    assert (
        cli.main(
            ["prepare", str(tmp_path), "--task", "natural task", "--target", "codex", "--json"]
        )
        == 0
    )
    assert observed[0][0].baseline is None and observed[0][1]["rebuild"] is False
    assert isinstance(observed[0][1]["preparer"], SkillForgePreparer)


def test_no_headroom_blocks_prepare_and_control_needs_no_prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.recovery, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(campaign, "validate_offline", lambda: None)
    monkeypatch.setattr(campaign, "forecast_before_control", lambda: None)
    monkeypatch.setattr(campaign.recovery, "require_preflight", lambda arm: None)
    calls = []
    monkeypatch.setattr(campaign.recovery, "freeze_and_run", lambda arm: calls.append(arm))
    monkeypatch.setattr(sys, "argv", ["coordinator", "control"])
    campaign.main()
    assert calls == ["control"]
    (tmp_path / "control").mkdir()
    (tmp_path / "control/scoring.json").write_text(
        json.dumps(
            {
                "score": {
                    "Functional": True,
                    "Regression": True,
                    "Project-conformant": True,
                    "agent_persisted_tests_pass": True,
                }
            }
        )
    )
    monkeypatch.setattr(campaign, "prepare", lambda: pytest.fail("unnecessary prepare"))
    with pytest.raises(RuntimeError, match="NO_HEADROOM"):
        campaign.budgeted_prepare()
    assert not (tmp_path / "prepare-worker-launch.json").exists()
    assert json.loads((tmp_path / "gate.json").read_text())["Treatment_prepare_runs"] == 0


def test_functional_pass_with_unknown_project_contract_does_not_spend_prepare(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign.recovery, "CAMPAIGN", tmp_path)
    (tmp_path / "control").mkdir()
    (tmp_path / "control/scoring.json").write_text(
        json.dumps(
            {"score": {"Functional": True, "Regression": True, "agent_persisted_tests_pass": True}}
        )
    )
    with pytest.raises(RuntimeError, match="CONTROL_REVIEW_REQUIRED"):
        campaign.require_headroom()
    assert json.loads((tmp_path / "gate.json").read_text())["Treatment_prepare_runs"] == 0


def test_coordinator_no_headroom_never_starts_prepare_or_treatment(tmp_path, monkeypatch):
    r = campaign.recovery
    monkeypatch.setattr(r, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(campaign, "validate_offline", lambda: None)
    monkeypatch.setattr(campaign, "forecast_before_control", lambda: None)
    calls = []
    monkeypatch.setattr(r, "require_preflight", lambda arm: calls.append("preflight " + arm))
    monkeypatch.setattr(r, "freeze_and_run", lambda arm: calls.append(arm))

    def grade(arm):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "scoring.json").write_text(
            json.dumps(
                {
                    "score": {
                        "Functional": True,
                        "Regression": True,
                        "Project-conformant": True,
                        "agent_persisted_tests_pass": True,
                    }
                }
            )
        )

    monkeypatch.setattr(r, "grade_arm", grade)
    monkeypatch.setattr(
        campaign, "budgeted_prepare", lambda: pytest.fail("prepare spent on NO_HEADROOM")
    )
    campaign.run_observation()
    assert calls == ["preflight control", "preflight treatment", "control"]
    assert json.loads((tmp_path / "coordinator-result.json").read_text())["status"] == "NO_HEADROOM"
    with pytest.raises(FileExistsError):
        campaign.run_observation()


def test_preflight_requires_exact_role_and_binding(tmp_path, monkeypatch):
    r = campaign.recovery
    monkeypatch.setattr(r, "RECOVERY", tmp_path)
    monkeypatch.setattr(r.harness, "frozen_context", lambda: (None, object()))
    monkeypatch.setattr(
        r,
        "task_row",
        lambda _: {"control_directory": str(tmp_path), "treatment_directory": str(tmp_path)},
    )
    monkeypatch.setattr(r, "preflight_binding", lambda *a: {"runtime": "current"})
    out = tmp_path / "attempt-old"
    out.mkdir()
    path = out / "preflight.json"
    path.write_text(
        json.dumps({"status": "PASS", "arm": "control", "binding": {"runtime": "current"}})
    )
    assert r.require_preflight("control") == path
    with pytest.raises(RuntimeError, match="current role"):
        r.require_preflight("treatment")
    path.write_text(
        json.dumps({"status": "PASS", "arm": "control", "binding": {"runtime": "stale"}})
    )
    with pytest.raises(RuntimeError):
        r.require_preflight("control")


def test_completed_control_uses_original_launch_proof_after_edits(tmp_path, monkeypatch):
    r = campaign.recovery
    monkeypatch.setattr(r, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(r.harness, "frozen_context", lambda: (None, object()))
    monkeypatch.setattr(
        r, "task_row", lambda _: {"prompt_sha256": "frozen", "control_directory": str(tmp_path)}
    )
    folder = tmp_path / "control"
    folder.mkdir()
    proof = tmp_path / "original-preflight.json"
    proof.write_text(
        json.dumps(
            {"status": "PASS", "arm": "control", "binding": {"workspace": str(tmp_path.resolve())}}
        )
    )
    freeze = {
        "preflight_evidence": str(proof),
        "preflight_sha256": r.digest(proof),
        "prompt_sha256": "frozen",
    }
    (folder / "freeze.json").write_text(json.dumps(freeze))
    (folder / "result.json").write_text(json.dumps({"exit_code": 0, "timeout": False}))
    (folder / "events.jsonl").write_text(
        "\n".join(json.dumps({"type": kind}) for kind in ("turn.started", "turn.completed")),
        encoding="utf-8",
    )
    (tmp_path / "edited-by-Control.py").write_text("new_behavior = 1")
    monkeypatch.setattr(
        r,
        "require_preflight",
        lambda *a: pytest.fail("post-edit snapshot incorrectly used as prelaunch gate"),
    )
    campaign.require_completed_control()
    proof.write_text("{}")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        campaign.require_completed_control()


def test_agent_deadline_uses_measured_common_ceiling_and_kills_family(monkeypatch):
    r = campaign.recovery
    waits, killed = [], []

    class Process:
        pid = 123

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)

    monkeypatch.setattr(r.subprocess, "run", lambda argv, **kw: killed.append(argv))
    assert r.AGENT_WALL_SECONDS == 315.1044808
    assert r.wait_actor(Process())
    assert waits == [315.1044808, 15]
    assert killed == [["taskkill", "/PID", "123", "/T", "/F"]]


def test_identical_check_single_flight(repo, monkeypatch):
    monkeypatch.setenv("TEMP", str(repo.parent))
    verifier = IndependentVerifier()
    plan = VerifyPlanBuilder().build(route(repo, False), (check("public", "print('checked')"),))
    started, release = threading.Event(), threading.Event()
    original = verifier._execute_check

    def held(*a, **kw):
        started.set()
        assert release.wait(10)
        return original(*a, **kw)

    monkeypatch.setattr(verifier, "_execute_check", held)
    result = []
    worker = threading.Thread(target=lambda: result.append(verifier.verify(repo, "HEAD", plan)))
    worker.start()
    assert started.wait(10)
    try:
        with pytest.raises(ValueError, match="already running"):
            IndependentVerifier().verify(repo, "HEAD", plan)
    finally:
        release.set()
        worker.join(10)
    assert len(result) == 1
    assert IndependentVerifier().verify(repo, "HEAD", plan) == result[0]


def test_real_shape_budget_is_checked_before_control_spend(tmp_path, monkeypatch):
    import audit_u07_shape

    monkeypatch.setattr(campaign, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(campaign, "validate_offline", lambda: None)
    monkeypatch.setattr(campaign.recovery, "require_preflight", lambda arm: None)
    monkeypatch.setattr(campaign.recovery.harness, "frozen_context", lambda: (None, object()))
    monkeypatch.setattr(
        campaign.recovery, "task_row", lambda _: {"treatment_directory": str(tmp_path)}
    )
    monkeypatch.setattr(
        audit_u07_shape,
        "forecast_shape",
        lambda root: {
            "counts": {"workflow": 5, "tendency": 0, "anti-pattern": 5, "structural": 2},
            "models": 0,
        },
    )
    monkeypatch.setattr(
        campaign.recovery,
        "freeze_and_run",
        lambda arm: pytest.fail("unsupported budget spent an Agent"),
    )
    monkeypatch.setattr(sys, "argv", ["coordinator", "control"])
    with pytest.raises(BudgetExceeded, match="candidate shape"):
        campaign.main()
    record = json.loads((tmp_path / "request-shape-plan.json").read_text())
    assert record["status"] == "INVALID_BUDGET" and record["Agent_runs"] == 0


def test_worker_budget_failure_keeps_invalid_budget_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.recovery, "CAMPAIGN", tmp_path)
    for name in ("validate_offline", "require_headroom", "require_completed_control"):
        monkeypatch.setattr(campaign, name, lambda: None)
    monkeypatch.setattr(campaign.recovery, "require_preflight", lambda arm: None)
    monkeypatch.setattr(
        campaign.recovery, "run_measured", lambda *a, **k: {"exit_code": 1, "timeout": False}
    )
    (tmp_path / "treatment").mkdir()
    (tmp_path / "treatment/prepare-result.json").write_text(
        json.dumps({"status": "INVALID_BUDGET", "reason": "actual cold shape unsupported"})
    )
    with pytest.raises(BudgetExceeded, match="actual cold shape unsupported"):
        campaign.budgeted_prepare()


def test_reviewed_resume_reuses_original_control_and_cannot_launch_another(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "RESUME_TREATMENT", True)
    monkeypatch.setattr(campaign, "ORIGINAL_CONTROL", tmp_path / "original/control")
    monkeypatch.setattr(campaign, "validate_offline", lambda: None)
    monkeypatch.setattr(
        campaign.recovery, "freeze_and_run", lambda arm: pytest.fail("duplicate Control")
    )
    assert campaign.control_folder() == tmp_path / "original/control"
    campaign.campaign_budget().validate_shape(campaign.VALIDATED_COLD_SHAPE)
    monkeypatch.setattr(sys, "argv", ["coordinator", "control"])
    with pytest.raises(RuntimeError, match="another Control"):
        campaign.main()
    monkeypatch.setattr(sys, "argv", ["coordinator", "run"])
    with pytest.raises(RuntimeError, match="another Control"):
        campaign.main()
