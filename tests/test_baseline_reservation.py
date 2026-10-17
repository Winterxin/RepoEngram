"""Real process ownership: reject duplicate work and release after interruption."""
import os
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from skillforge.cost_control import BudgetExceeded
from skillforge.productization.verified_prepare import _baseline_reservation, VerifiedPreparer


def test_legacy_sentinel_without_live_owner_is_recoverable(tmp_path):
    lock = tmp_path / ".skillforge/productization/prepare/verified-baseline.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("exclusive verified baseline preparation\n")
    original = lock.read_bytes()
    with _baseline_reservation(tmp_path):
        with pytest.raises(BudgetExceeded, match="already reserved"):
            with _baseline_reservation(tmp_path):
                pytest.fail("duplicate work acquired active reservation")
    assert lock.read_bytes() == original  # Retain inode; no delete-and-recreate race.
    with _baseline_reservation(tmp_path):
        pass


def test_killed_process_releases_reservation_without_deleting_sentinel(tmp_path):
    code = """import sys,os,json
from pathlib import Path
from skillforge.productization.verified_prepare import _baseline_reservation
with _baseline_reservation(Path(sys.argv[1])):
    print(json.dumps({'state':'LOCKED','owner_pid':os.getpid()}),flush=True)
    sys.stdin.readline()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    process = subprocess.Popen([sys.executable, "-u", "-c", code, str(tmp_path)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = queue.Queue()
    threading.Thread(target=lambda: output.put(process.stdout.readline()), daemon=True).start()
    try:
        owner=json.loads(output.get(timeout=10))
        assert owner['state'] == "LOCKED"
        with pytest.raises(BudgetExceeded, match="already reserved"):
            with _baseline_reservation(tmp_path):
                pytest.fail("second process must not run models")
        # Windows venv redirectors can be a different process from the actual
        # interpreter holding the lock. Terminate the owner, not just its launcher.
        if os.name == 'nt':
            subprocess.run(['taskkill','/PID',str(owner['owner_pid']),'/T','/F'],
                           capture_output=True,check=True,timeout=5)
        else:
            process.kill()
        process.wait(timeout=10)
        with _baseline_reservation(tmp_path):
            assert (tmp_path / ".skillforge/productization/prepare/verified-baseline.lock").exists()
    finally:
        if process.poll() is None: process.kill();process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr): stream.close()


def test_reservation_precedes_import_callback(tmp_path):
    from types import SimpleNamespace
    seen = []
    preparer = VerifiedPreparer(None, (), config={}, ledger=None)
    preparer._prepare = lambda request: seen.append("prepare")
    with _baseline_reservation(tmp_path):
        with pytest.raises(BudgetExceeded, match="already reserved"):
            preparer.prepare(SimpleNamespace(path=tmp_path), before_prepare=lambda: seen.append("import"))
    assert seen == []
    preparer.prepare(SimpleNamespace(path=tmp_path), before_prepare=lambda: seen.append("import"))
    assert seen == ["import", "prepare"]
