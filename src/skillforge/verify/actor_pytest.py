"""Actor-only pytest governance; external frozen scoring does not load this policy.

Activated through PYTEST_PLUGINS for console, python -m and pytest.main entries.
This is an execution policy, not a test/oracle selector or obligation generator.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from skillforge.cost_control import digest
from skillforge.filesystem_safety import no_symlink
from skillforge.repository.repo_snapshot_semantic_cache import _snapshot, _production_version

_STATE = pytest.StashKey()
_CAP = 8192


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _summary(value):
    # Bounded counts and short diagnostics; full terminal output stays in log.
    text = json.dumps(value, ensure_ascii=False)
    if len(text.encode("utf-8")) > _CAP:
        value = {**value, "failures": value.get("failures", [])[:1]}
        text = json.dumps(value, ensure_ascii=False)
    return text


def _decline(config, root, reason):
    start = time.monotonic_ns()
    directory = root / ".runtime/actor-checks"
    no_symlink(directory, root)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(config.invocation_params.args)).encode()
    _write(directory / f"policy-{start}.json", {"kind": "actor-check-policy", "start_ns": start,
        "end_ns": time.monotonic_ns(), "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(), "usage": None,
        "counter": 1 + len(list(directory.glob("policy-*.json"))), "executed_tests": 0,
        "cache": "not_applicable", "reason": reason})
    pytest.exit("CHECK_POLICY: " + reason, returncode=4)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    raw = os.environ.get("SKILLFORGE_ACTOR_ROOT")
    if not raw:
        return
    root = Path(raw).resolve()
    cwd = Path(config.invocation_params.dir).resolve()
    if not cwd.is_relative_to(root):
        _decline(config, root, "check cwd outside actor repository")
    selected = []
    for arg in config.args:
        file = (cwd / arg.split("::", 1)[0]).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            if not config.getoption("collectonly"):
                _decline(config, root, "iteration requires explicit public test files/nodes; whole suite is reserved for final external scoring")
        selected.append(file)
    public_files = {p.resolve() for p in root.rglob("test_*.py")
                    if not any(n in {".runtime", ".git", ".venv", ".skillforge"} for n in p.relative_to(root).parts)}
    if (not config.getoption("collectonly") and not config.option.keyword and not config.option.markexpr
            and all("::" not in arg for arg in config.args)
            and public_files and public_files.issubset(set(selected))):
        _decline(config, root, "whole public suite is reserved for final external scoring")
    runtime = root / ".runtime/actor-checks"
    no_symlink(runtime, root)
    runtime.mkdir(parents=True, exist_ok=True)
    identity = {"snapshot": _snapshot(root), "argv": list(config.invocation_params.args),
                "selected": list(config.args), "cwd": str(cwd), "env": digest(dict(os.environ)),
                "python": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
                "production": _production_version()}
    payload = json.dumps(identity, sort_keys=True).encode()
    key = digest(identity)
    directory = runtime / key[:20]
    no_symlink(directory, root)
    directory.mkdir(exist_ok=True)
    cached = directory / "result.json"
    try:
        lock = (directory / "execution.lock").open("x", encoding="ascii")
    except FileExistsError:
        pytest.exit("CHECK_POLICY: identical check already running; no duplicate execution", returncode=4)
    with lock:
        lock.write(str(os.getpid()))
    state = {"identity": identity, "key": key, "directory": directory, "start_ns": time.monotonic_ns(),
             "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
             "outcomes": {}, "failures": [], "cached": False, "summary": None}
    config.stash[_STATE] = state
    if cached.exists():
        entry = json.loads(cached.read_text(encoding="utf-8"))
        if entry.get("identity") != identity or entry.get("checksum") != digest(entry.get("result")):
            pytest.exit("CHECK_POLICY: cached check identity/checksum mismatch", returncode=4)
        state["cached"] = True
        state["summary"] = {**entry["result"], "cache": "hit", "executed_tests": 0}
        pytest.exit(_summary(state["summary"]), returncode=entry["result"]["exit_code"])
    # An explicit caller-selected temp directory is retained only if ignored.
    chosen = Path(config.getoption("basetemp")).resolve() if config.getoption("basetemp") else directory / "pytest-temp"
    no_symlink(chosen, root)
    ignored = __import__("subprocess").run(["git", "-c", f"safe.directory={root}", "check-ignore", "-q", str(chosen / "probe")], cwd=root, capture_output=True)
    if ignored.returncode:
        pytest.exit("CHECK_POLICY: pytest runtime directory must be Git-ignored inside actor repository", returncode=4)
    config.option.basetemp = str(chosen)
    terminal = config.pluginmanager.getplugin("terminalreporter")
    state["log"] = (directory / "complete.log").open("w", encoding="utf-8")
    if terminal:
        from _pytest._io import TerminalWriter
        state["terminal"], state["writer"] = terminal, terminal._tw
        terminal._tw = TerminalWriter(file=state["log"])
    # Also bound -s and arbitrary Python print output; normal pytest capture is
    # preserved, and its complete failure/summary rendering uses the log writer.
    state["stdout"], state["stderr"] = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = state["log"]


def pytest_runtest_logreport(report):
    # Locate state through the active config (set below); no coverage invented.
    config = getattr(pytest_runtest_logreport, "config", None)
    if config is None or _STATE not in config.stash:
        return
    state = config.stash[_STATE]
    if report.when == "call" or report.failed or report.skipped:
        state["outcomes"][report.nodeid] = report.outcome
        if report.failed:
            state["failures"].append({"node": report.nodeid[:200], "phase": report.when,
                                      "detail": str(report.longrepr)[-700:]})


def pytest_sessionstart(session):
    pytest_runtest_logreport.config = session.config


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if _STATE not in session.config.stash:
        return
    state = session.config.stash[_STATE]
    counts = {name: sum(v == name for v in state["outcomes"].values()) for name in ("passed", "failed", "skipped")}
    value = {"exit_code": int(exitstatus), "cache": "miss", "counts": counts,
             "executed_tests": len(state["outcomes"]), "collect_only": session.config.getoption("collectonly"),
             "failures": state["failures"][:4],
             "log": str(state["directory"] / "complete.log")}
    state["summary"] = value
    if _snapshot(Path(os.environ["SKILLFORGE_ACTOR_ROOT"])) == state["identity"]["snapshot"]:
        _write(state["directory"] / "result.json", {"identity": state["identity"],
                "result": value, "checksum": digest(value)})
    else:
        value["cache_bypass_reason"] = "source changed during check; result not published"


def pytest_unconfigure(config):
    if _STATE not in config.stash:
        return
    state = config.stash[_STATE]
    if "stdout" in state:
        sys.stdout, sys.stderr = state["stdout"], state["stderr"]
    if "terminal" in state:
        state["terminal"]._tw = state["writer"]
    if "log" in state:
        state["log"].close()
    end = time.monotonic_ns()
    event = {"kind": "actor-check", "start_ns": state["start_ns"], "end_ns": end,
             "payload_bytes": state["payload_bytes"], "payload_sha256": state["payload_sha256"],
             "usage": None, "cache": "hit" if state["cached"] else "miss",
             "counter": 1 + len(list(state["directory"].glob("event-*.json"))),
             "counter_scope": "exact_check_identity", "result": state["summary"]}
    # Each request has its own monotonic event, including cached invocations.
    _write(state["directory"] / f"event-{state['start_ns']}.json", event)
    (state["directory"] / "execution.lock").unlink()
    if state["summary"] is not None and not state["cached"]:
        print(_summary(state["summary"]))
