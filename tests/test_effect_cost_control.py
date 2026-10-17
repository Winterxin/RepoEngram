"""Offline proof for the actual transport and all fatal economic gates."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_cold_semantic_batching import twelve_inputs as twelve_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_semantic_batch_transport import recorded_models
from test_verify_runner import repo as repo, route, check

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from effect_transport import EffectTransport
from public_output import page
from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import SkillForgePreparer, PrepareRequest
from skillforge.productization.verified_prepare import VerifiedPreparer
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.verify import IndependentVerifier, VerifyPlanBuilder


def actual_models(inputs, tmp_path, mode="success", cold_shape=None):
    _, fixture, adapters = recorded_models(inputs, mode)
    # Fixture sees the EFFECTIVE command, so retains its unchanged parser and
    # response assertions while attesting the real stable runner normalization.
    def offline(command, **kwargs):
        assert kwargs["cwd"] == transport.directory
        compatible = list(command)
        compatible[2:2] = ["--sandbox", "read-only"]
        return fixture(compatible, **kwargs)
    budget, ledger = SemanticBudget(cold_shape=cold_shape), CostLedger()
    transport = EffectTransport(executable=sys.executable, directory=tmp_path / ".skillforge/transport",
                                environment={"FIXTURE": "no subprocess/no network"},
                                permissions=[], budget=budget, ledger=ledger, runner=offline)
    for adapter in adapters:
        adapter._runner = transport
        transport.bind(adapter)
    return adapters, transport, budget, ledger


TWELVE_SHAPE = {"workflow": 5, "tendency": 0, "anti-pattern": 5, "structural": 2}


@pytest.mark.parametrize("mode,expected_calls", [("success", 3), ("gate_failure", 4),
                                                ("partial", 4), ("batch_failure", 5)])
def test_declared_twelve_shape_preserves_requests_gates_and_shared_budget(twelve_inputs, tmp_path, mode, expected_calls):
    from test_cold_semantic_batching import run, SingleFixture
    models, transport, budget, ledger = actual_models(twelve_inputs, tmp_path, mode, cold_shape=TWELVE_SHAPE)
    root = twelve_inputs[0]
    request = PrepareRequest(root, "update routes", CodeAgentTarget.CODEX, "app.api.route0")
    preparer = SkillForgePreparer(*models, semantic_cache_root=tmp_path / ".skillforge/cache")
    controlled = VerifiedPreparer(preparer, models, config={}, budget=budget, ledger=ledger)
    if mode == "batch_failure":
        with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
            controlled.prepare(request)
        assert budget.calls == 5
        assert len([e for e in ledger.events if e["kind"] == "semantic"]) == 5
        return
    result = controlled.prepare(request)
    expected = run(twelve_inputs, SingleFixture, "gate_failure" if mode == "gate_failure" else "success")
    baseline = load_baseline(root, result.evolution_baseline_path)
    assert baseline.source_dna == expected[1][0]
    assert baseline.source_compilation == expected[1][1]
    assert baseline.semantic_structural == expected[0].structural
    assert result.structural_candidate_count == 2
    assert budget.calls == expected_calls
    cold_baseline = baseline
    transport.budget = warm_budget = SemanticBudget(max_calls=0, cold_shape=TWELVE_SHAPE)
    warm = VerifiedPreparer(preparer, models, config={}, budget=warm_budget, ledger=ledger).prepare(request)
    assert warm_budget.calls == 0
    assert load_baseline(root, warm.evolution_baseline_path) == cold_baseline
    assert warm.selected_skill_ids == result.selected_skill_ids
    assert warm.semantic_failures == result.semantic_failures


def test_declared_shape_is_exact_and_default_still_fails_closed():
    with pytest.raises(BudgetExceeded, match="candidate shape"):
        SemanticBudget().validate_shape(TWELVE_SHAPE)
    budget = SemanticBudget(cold_shape=TWELVE_SHAPE)
    budget.validate_shape(TWELVE_SHAPE)
    with pytest.raises(BudgetExceeded, match="differs"):
        budget.validate_shape({**TWELVE_SHAPE, "structural": 3})
    assert budget.calls == 0


@pytest.mark.parametrize("mode", ["success", "gate_failure"])
def test_actual_transport_cold_warm_baseline_canonical_and_provenance(eleven_inputs, tmp_path, mode):
    root = eleven_inputs[0]
    models, transport, budget, ledger = actual_models(eleven_inputs, tmp_path, mode)
    assert all(RepoSnapshotSemanticCache(root).wrap(m) is not m for m in models)
    s, w, a = models
    preparer = SkillForgePreparer(s, w, a, semantic_cache_root=tmp_path / ".skillforge/cache")
    controlled = VerifiedPreparer(preparer, models, config={"production": "unchanged"}, budget=budget, ledger=ledger)
    request = PrepareRequest(root, "update routes", CodeAgentTarget.CODEX, "app.api.route0")
    cold = controlled.prepare(request)
    assert 3 <= budget.calls <= 5
    baseline = load_baseline(root, cold.evolution_baseline_path)
    registry = SkillRegistryBuilder().build(baseline.source_compilation)
    expected_route = SkillRouter().route(registry, SkillRoutingRequest(cold.repository_id, request.task,
                                                                       current_module=request.current_module))
    warm_budget = SemanticBudget(max_calls=0)
    transport.budget = warm_budget
    warm = VerifiedPreparer(preparer, models, config=controlled.config, budget=warm_budget, ledger=ledger).prepare(request)
    assert warm_budget.calls == 0
    assert warm.dna_snapshot_id == cold.dna_snapshot_id
    assert warm.selected_skill_ids == cold.selected_skill_ids == tuple(m.skill.skill_id for m in expected_route.matches)
    assert load_baseline(root, warm.evolution_baseline_path) == baseline
    assert SkillRegistryBuilder().build(load_baseline(root, warm.evolution_baseline_path).source_compilation) == registry
    assert warm.semantic_failures == cold.semantic_failures
    assert warm.semantic_failure_count == cold.semantic_failure_count
    assert warm.to_dict()["semantic_channels"] == cold.to_dict()["semantic_channels"]
    assert cold.semantic_failure_count == (1 if mode == "gate_failure" else 0)
    next_task = replace(request, task="fix service behavior")
    cross_task = VerifiedPreparer(preparer, models, config=controlled.config, budget=warm_budget, ledger=ledger).prepare(next_task)
    canonical_next = SkillRouter().route(registry, SkillRoutingRequest(cold.repository_id, next_task.task,
                                                                     current_module=next_task.current_module))
    assert cross_task.selected_skill_ids == tuple(m.skill.skill_id for m in canonical_next.matches)
    assert cross_task.dna_snapshot_id == cold.dna_snapshot_id and warm_budget.calls == 0
    (root / "new-working-byte.txt").write_text("drift")
    with pytest.raises(BudgetExceeded, match="drift"):
        VerifiedPreparer(preparer, models, config=controlled.config, budget=warm_budget, ledger=ledger).prepare(request)


def test_bypass_aborts_zero_calls(eleven_inputs, tmp_path):
    models, transport, budget, ledger = actual_models(eleven_inputs, tmp_path)
    del models[0].semantic_cache_identity
    with pytest.raises(BudgetExceeded, match="bypass"):
        VerifiedPreparer(SkillForgePreparer(*models), models, config={}, budget=budget, ledger=ledger).prepare(
            PrepareRequest(eleven_inputs[0], "task", CodeAgentTarget.CODEX))
    assert budget.calls == 0


def test_budget_failure_cannot_fall_back_to_unlimited_singles(eleven_inputs, tmp_path):
    models, transport, budget, ledger = actual_models(eleven_inputs, tmp_path, "batch_failure")
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        SkillForgePreparer(*models).prepare(PrepareRequest(eleven_inputs[0], "task", CodeAgentTarget.CODEX))
    assert budget.calls == 5
    assert len([e for e in ledger.events if e["kind"] == "semantic"]) == 5


def test_wall_gates_pre_spawn_and_after_timed_call():
    times = [0.0]
    budget = SemanticBudget(clock=lambda: times[0], prepare_seconds=10, semantic_seconds=4)
    assert budget.reserve() == 4
    times[0] = 10
    with pytest.raises(BudgetExceeded, match="prepare"):
        budget.reserve()
    assert budget.calls == 1
    budget = SemanticBudget(clock=lambda: 0, semantic_seconds=4)
    budget.reserve()
    with pytest.raises(BudgetExceeded, match="semantic"):
        budget.finish(4)
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    assert budget.calls == 1


def test_verify_exact_check_reuse_and_env_source_invalidation(repo, monkeypatch):
    # Isolate persistent runtime history even when pytest reuses a basetemp path.
    monkeypatch.setenv("TEMP", str(repo.parent))
    plan = VerifyPlanBuilder().build(route(repo, False), (check("public", "print('checked')"),))
    first = IndependentVerifier().verify(repo, "HEAD", plan)
    assert IndependentVerifier().verify(repo, "HEAD", plan) == first
    log = IndependentVerifier.runtime_directory(repo) / "events.jsonl"
    assert [json.loads(line)["cache"] for line in log.read_text("utf-8").splitlines()] == ["miss", "hit"]
    monkeypatch.setenv("EFFECT_CHECK_ENV", "changed")
    IndependentVerifier().verify(repo, "HEAD", plan)
    (repo / "source.py").write_text("changed = 1")
    IndependentVerifier().verify(repo, "HEAD", plan)
    assert [json.loads(line)["cache"] for line in log.read_text("utf-8").splitlines()] == ["miss", "hit", "miss", "miss"]


def test_large_output_paged_and_ranges(tmp_path):
    output = tmp_path / "large.txt"
    output.write_text("x" * 415249)
    result = page(output)
    assert len(result["text"].encode()) == result["next_offset"] == 8192
    assert result["total_bytes"] == 415249
    assert len(page(output, result["next_offset"])["text"]) == 8192
    with pytest.raises(ValueError):
        page(output, limit=415249)


def test_actual_generic_command_guard(tmp_path, capsys):
    from bounded_command import run
    assert run([sys.executable, "-c", "print('x' * 415249)"], tmp_path) == 0
    captured = capsys.readouterr().out
    assert len(captured) < 9500
    assert "Read another page" in captured
    assert list(tmp_path.glob("*-stdout.txt"))[0].stat().st_size > 415249


def test_public_checker_requires_real_behavior_link_and_changed_ast(repo):
    from skillforge.verify.public_tests import behavioral_nodes, changed_tests
    source = 'from app.services.example import build\ndef helper(): return build()\ndef test_behavior():\n assert helper()\n'
    modules = ('app.services',)
    assert set(behavioral_nodes(source, production_modules=modules)) == {"test_behavior"}
    assert behavioral_nodes('def test_mirror():\n assert len([1]) == 1\n',
                            production_modules=modules) == {}
    folder = repo / "apps/api/tests";folder.mkdir(parents=True)
    (folder / "test_public.py").write_text(source)
    assert changed_tests(repo, 'apps/api/tests', production_modules=modules) == [
        "apps/api/tests/test_public.py::test_behavior"]


def test_coverage_mapping_does_not_invent_coverage(repo):
    from skillforge.verify.public_tests import test_execution_covers as covers
    from dataclasses import replace
    plan = VerifyPlanBuilder().build(route(repo))
    existing = plan.obligations[0]
    from skillforge.domain.verification import VerifyObligationKind
    from types import SimpleNamespace
    supported = replace(existing, target_ref=replace(existing.target_ref, kind=VerifyObligationKind.WORKFLOW_STEP),
                        description="Update corresponding tests", condition=None)
    arbitrary = replace(supported, description="Change business behavior")
    extra = replace(supported, condition="All outputs satisfy the freeform business policy")
    assert covers(SimpleNamespace(obligations=(supported, arbitrary, extra))) == [supported.target_ref]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shim integration")
def test_real_windows_search_shim_with_unicode_paths(tmp_path):
    import os
    import subprocess
    from u07_cost_control import install_output_guards
    root = tmp_path / "中文用户"
    root.mkdir()
    install_output_guards(root, None)
    source = root / "large.txt"
    source.write_text("x" * 415249, "utf-8")
    env = dict(os.environ, VIRTUAL_ENV=str(Path(sys.executable).parents[1]),
               TEMP=str(root / ".runtime/tmp"), PYTHONIOENCODING="utf-8")
    command = root / ".runtime/bin/rg.exe"
    result = subprocess.run([str(command), "x|y", str(source)], env=env,
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout) < 9500
    assert "Read another page" in result.stdout
