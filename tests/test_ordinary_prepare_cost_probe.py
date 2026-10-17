"""No-model safeguards for the explicit ordinary-use observation wrapper."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import ordinary_prepare_cost_probe as probe
from ordinary_prepare_cost_probe import OrdinaryTransport

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget, digest
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.semantic_batch import supports_semantic_batch


def transport(tmp_path, max_calls):
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"offline fixture, never executable")
    (tmp_path / "auth.json").write_text(json.dumps({"tokens": {"account_id": "fixture"}}))
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps({"models": [{"slug": "gpt-6-luna"}]}))
    port = OrdinaryTransport(executable=executable, directory=tmp_path / "transport",
        environment={"CODEX_HOME": str(tmp_path)}, permissions=[],
        budget=SemanticBudget(max_calls=max_calls), ledger=CostLedger(), model_catalog=catalog)
    return port


def test_ordinary_identity_declares_actual_full_access_argv(tmp_path):
    port = transport(tmp_path, 0)
    model = port.bind(CodexCliSemanticModel(runner=port))
    port.validate_identity()
    assert supports_semantic_batch(model)
    assert port.prefix[port.prefix.index("--sandbox") + 1] == "danger-full-access"
    assert 'approval_policy="never"' in port.prefix
    assert "unseen-sensitive" not in " ".join(port.prefix)
    assert port.identity["cli_config"]["argv_sha256"] == digest(port.prefix)
    port.prefix.append("unexpected")
    with pytest.raises(BudgetExceeded, match="drift"):
        port.validate_identity()


@pytest.mark.parametrize("max_calls", [0, 1])
def test_budget_forbids_next_execute_before_spawn(tmp_path, max_calls):
    port = transport(tmp_path, max_calls)
    schema, output = tmp_path / "schema.json", tmp_path / "output.json"
    schema.write_text("{}")
    command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "--skip-git-repo-check", "--sandbox", "read-only", "-m", "gpt-6-luna", "-C",
        str(tmp_path), "--output-schema", str(schema), "-o", str(output), "-"]
    executed = []
    def offline_execute(argv, kwargs, timeout):
        executed.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    port._execute = offline_execute
    kwargs = dict(cwd=tmp_path, input="fixture", timeout=180, text=True,
                  encoding="utf-8", capture_output=True, check=False)
    for _ in range(max_calls):
        port(command, **kwargs)
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        port(command, **kwargs)
    assert len(executed) == max_calls
    assert port.budget.calls == max_calls


def test_existing_observation_refuses_before_model_and_preserves_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "OUT", tmp_path)
    stopped = tmp_path / "stop.json"
    stopped.write_text('{"reason":"original stop"}')
    with pytest.raises(RuntimeError, match="no paid retry"):
        probe.observe()
    assert stopped.read_text() == '{"reason":"original stop"}'
    assert not (tmp_path / "observation-start.json").exists()
