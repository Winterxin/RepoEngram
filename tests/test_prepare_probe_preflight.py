"""Probe failures close model launch, clean the family, and preserve old evidence."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "evals/production_prepare_cost_smoke.py"
    spec = importlib.util.spec_from_file_location("prepare_probe_preflight_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["success", "exit_failure", "deadline"])
def test_permission_outcome_and_family_cleanup_before_model(probe, monkeypatch, tmp_path, mode):
    records, kills = [], []
    monkeypatch.setattr(probe, "OUT", tmp_path)

    class Process:
        pid = 123456
        returncode = 0
        calls = 0

        def communicate(self, timeout):
            self.calls += 1
            if mode == "deadline" and self.calls == 1:
                raise subprocess.TimeoutExpired("private argv", timeout)
            self.returncode = 1 if mode != "success" else 0
            return b"permission-spawn-ok", b""

    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(probe.subprocess, "run", lambda args, **kw: kills.append(args))
    monkeypatch.setattr(probe, "write", lambda name, value: records.append(value))
    if mode == "success":
        probe.permission_preflight(["codex", "sandbox"], Path("."), {})
    else:
        with pytest.raises(RuntimeError, match="INVALID_INFRASTRUCTURE"):
            probe.permission_preflight(["codex", "sandbox"], Path("."), {})
    assert records[0]["live_models"] == records[0]["Agent_runs"] == 0
    assert records[0]["status"] == ("PASS" if mode == "success" else "INVALID_INFRASTRUCTURE")
    assert bool(kills) == (mode == "deadline")
    if kills:
        assert kills == [["taskkill", "/PID", "123456", "/T", "/F"]]


def test_one_shot_refusal_cannot_overwrite_old_evidence_or_auth(probe, monkeypatch, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    stop = old / "stop.json"
    stop.write_text("original stop")
    home = tmp_path / "home"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_text("fixture not real credentials")
    monkeypatch.setattr(probe, "OUT", old)
    monkeypatch.setattr(probe, "HOME", home)
    monkeypatch.setattr(sys, "argv", ["probe.py"])

    def forbidden_setup():
        raise RuntimeError("one-shot directory already exists")

    monkeypatch.setattr(probe, "setup", forbidden_setup)
    with pytest.raises(RuntimeError, match="one-shot"):
        probe.main()
    assert stop.read_text() == "original stop"
    assert auth.read_text() == "fixture not real credentials"
