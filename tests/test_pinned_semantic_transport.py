"""No-model guarantees for the actual transport's frozen startup metadata."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from effect_transport import EffectTransport
from semantic_catalog import catalog_identity, freeze_catalog
from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache


def catalog(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"client_version": "original", "models": [
        {"slug": "gpt-6-luna", "base_instructions": "unchanged", "context_window": 12345},
        {"slug": "other", "base_instructions": "also preserved"}]}), "utf-8")
    target = tmp_path / "models.json"
    seal = freeze_catalog(source, target)
    assert json.loads(target.read_text())["models"] == json.loads(source.read_text())["models"]
    return target, seal


def transport(tmp_path, model_catalog=None, runner=None):
    return EffectTransport(executable=sys.executable, directory=tmp_path / "transport",
                           environment={"FIXTURE": "no-model"}, permissions=[],
                           budget=SemanticBudget(), ledger=CostLedger(), runner=runner,
                           model_catalog=model_catalog)


def test_native_implicit_metadata_aborts_before_execution(tmp_path):
    with pytest.raises(BudgetExceeded, match="requires frozen model catalog BEFORE model"):
        transport(tmp_path)


def test_truthful_identity_and_cache_eligibility_then_drift_aborts(tmp_path):
    # Cache eligibility requires an actual verifiable committed Git snapshot.
    # An empty non-Git directory correctly fails closed for this transport.
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-m", "fixture snapshot"],
                   cwd=tmp_path, check=True, capture_output=True)
    path, seal = catalog(tmp_path)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        pytest.fail("offline test reached transport")
    runner = transport(tmp_path, path, forbidden)
    model = runner.bind(CodexCliSemanticModel(runner=runner))
    assert runner.identity["response_parameters"]["model_catalog"]["sha256"] == seal["sha256"]
    assert "features.remote_plugin=false" in runner.prefix
    assert "model_catalog_json=" + json.dumps(str(path)) in runner.prefix
    cache = RepoSnapshotSemanticCache(tmp_path)
    assert cache.wrap(model) is not model
    runner.validate_identity()
    data = json.loads(path.read_text())
    data["models"][0]["context_window"] += 1
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(BudgetExceeded, match="catalog drift BEFORE cache/model"):
        cache.wrap(model)
    assert calls == [] and runner.budget.calls == 0


@pytest.mark.parametrize("models", [[], [{"slug": "gpt-6-luna"}] * 2, "invalid"])
def test_missing_ambiguous_or_invalid_catalog_fails_closed(tmp_path, models):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"models": models}))
    with pytest.raises(BudgetExceeded, match="unverifiable"):
        catalog_identity(path, "gpt-6-luna")


def test_existing_seal_cannot_be_overwritten(tmp_path):
    path, _ = catalog(tmp_path)
    source = tmp_path / "source.json"
    data = json.loads(source.read_text())
    data["models"][0]["base_instructions"] = "changed"
    source.write_text(json.dumps(data))
    original = path.read_bytes()
    with pytest.raises(BudgetExceeded, match="existing frozen model catalog drift"):
        freeze_catalog(source, path)
    assert path.read_bytes() == original


def original_call(runner, tmp_path):
    schema, output = tmp_path / "schema.json", tmp_path / "response.json"
    schema.write_text('{"type":"object"}')
    command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
               "--skip-git-repo-check", "--sandbox", "read-only", "-m", "gpt-6-luna",
               "-C", str(tmp_path), "--output-schema", str(schema), "-o", str(output), "-"]
    return runner(command, cwd=tmp_path, input="original prompt", text=True,
                  encoding="utf-8", capture_output=True, check=False, timeout=180)


def test_interrupted_transport_closes_ledger_without_fabricated_usage(tmp_path):
    def timeout(*args, **kwargs):
        (runner.directory / "001-stdout.jsonl").write_text('{"type":"turn.started"}\n')
        runner.budget.abort("fixture deadline")
    runner = transport(tmp_path, runner=timeout)
    with pytest.raises(BudgetExceeded, match="fixture deadline"):
        original_call(runner, tmp_path)
    event = runner.ledger.events[-1]
    assert runner.budget.calls == 1
    assert event["completion"] == "NOT_COMPLETED" and event["usage"] is None
    assert event["end_ns"] >= event["start_ns"]
    assert event["partial_stdout_jsonl_sha256"]


def test_cleanup_reserve_aborts_before_first_spawn(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("deadline guard reached runner")
    runner = transport(tmp_path, runner=forbidden)
    tick = [0.0]
    runner.budget = SemanticBudget(semantic_seconds=3, clock=lambda: tick[0])
    with pytest.raises(BudgetExceeded, match="BEFORE semantic spawn"):
        original_call(runner, tmp_path)
    assert runner.budget.calls == 0 and runner.ledger.events == []
