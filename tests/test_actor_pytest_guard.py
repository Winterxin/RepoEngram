"""Actual subprocess entry proofs, no Agent/model; external scoring stays normal."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def guarded(tmp_path):
    root = tmp_path / "repo"; root.mkdir()
    (root / ".gitignore").write_text(".runtime/\n__pycache__/\n.pytest_cache/\n")
    (root / "test_one.py").write_text("from pathlib import Path\ndef test_one():\n p=Path('.runtime/count'); p.parent.mkdir(exist_ok=True); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); print('x'*415249); assert True\n")
    (root / "test_other.py").write_text("def test_other(): assert True\n")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    env = dict(os.environ, SKILLFORGE_ACTOR_ROOT=str(root), PYTEST_PLUGINS="skillforge.verify.actor_pytest",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")  # Same explicit wire encoding as frozen effect actors.
    return root, env


def run(root, env, args):
    return subprocess.run([sys.executable, *args], cwd=root, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=25)


def test_native_module_and_import_entries_reuse_and_bound_output(guarded):
    root, env = guarded
    first = run(root, env, ["-m", "pytest", "-q", "-s", "test_one.py"])
    assert first.returncode == 0, first.stderr + first.stdout
    assert len(first.stdout.encode()) < 8192
    assert (root / ".runtime/count").read_text() == "1"
    second = run(root, env, ["-m", "pytest", "-q", "-s", "test_one.py"])
    assert second.returncode == 0 and '"cache": "hit"' in second.stdout + second.stderr
    imported = run(root, env, ["-c", "import pytest; raise SystemExit(pytest.main(['-q','-s','test_one.py']))"])
    assert imported.returncode == 0 and '"cache": "hit"' in imported.stdout + imported.stderr
    assert (root / ".runtime/count").read_text() == "1"
    logs = list((root / ".runtime/actor-checks").glob("*/complete.log"))
    assert len(logs) == 1 and logs[0].stat().st_size > 415249
    events = [json.loads(p.read_text(encoding="utf-8")) for p in (root / ".runtime/actor-checks").glob("*/event-*.json")]
    assert len(events) == 3 and sum(e["cache"] == "hit" for e in events) == 2
    assert all(e["start_ns"] <= e["end_ns"] and e["payload_bytes"] > 0 for e in events)
    with (root / "test_one.py").open("a") as stream: stream.write("\n# changed source\n")
    third = run(root, env, ["-m", "pytest", "-q", "-s", "test_one.py"])
    assert third.returncode == 0 and (root / ".runtime/count").read_text() == "2"


def test_heavy_suite_declined_before_tests_and_external_scoring_unaffected(guarded):
    root, env = guarded
    blocked = run(root, env, ["-m", "pytest", "-q"])
    assert blocked.returncode == 4 and "whole suite" in blocked.stdout + blocked.stderr
    assert not (root / ".runtime/count").exists()
    explicit_all = run(root, env, ["-m", "pytest", "-q", "test_one.py", "test_other.py"])
    assert explicit_all.returncode == 4 and "whole public suite" in explicit_all.stdout + explicit_all.stderr
    ordinary = dict(env); ordinary.pop("SKILLFORGE_ACTOR_ROOT")
    external = run(root, ordinary, ["-m", "pytest", "-q"])
    assert external.returncode == 0 and (root / ".runtime/count").read_text() == "1"


def test_failing_result_reused_without_false_pass(guarded):
    root, env = guarded
    (root / "test_one.py").write_text("def test_failure(): assert False, 'actual regression'\n")
    first = run(root, env, ["-m", "pytest", "-q", "test_one.py"])
    second = run(root, env, ["-m", "pytest", "-q", "test_one.py"])
    assert first.returncode == second.returncode == 1
    assert "actual regression" in first.stdout
    assert '"cache": "hit"' in second.stdout + second.stderr


def test_frozen_console_entry_uses_same_governance(guarded):
    root, env = guarded
    executable = Path.home() / "Desktop/skillforge-unseen-dependencies-v1/Scripts/pytest.exe"
    if not executable.exists():
        pytest.skip("frozen Windows console runtime unavailable")
    for expected in ("miss", "hit"):
        result = subprocess.run([str(executable), "-q", "-s", "test_one.py"], cwd=root,
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'"cache": "{expected}"' in result.stdout + result.stderr
        assert len((result.stdout + result.stderr).encode()) < 8192
    assert (root / ".runtime/count").read_text() == "1"
