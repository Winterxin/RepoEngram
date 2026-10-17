"""No-model launch policy validation; never invoke an effect actor."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "u06_policy", Path(__file__).parents[1] / "evals/u06_effect_cost_probe.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def policy(denied):
    return ["permissions.unseen-sensitive.filesystem={ "
            + json.dumps(str(denied)) + ' = "deny" }']


@pytest.mark.parametrize("resource", ["task", "src/server.py", "bin/python.exe"])
def test_launch_dependency_ancestor_denied(tmp_path, resource):
    with pytest.raises(RuntimeError, match="denied ancestor"):
        harness.assert_no_denied_ancestor(policy(tmp_path), [tmp_path / resource])


def test_exact_launch_dependency_denied(tmp_path):
    resource = tmp_path / "python.exe"
    with pytest.raises(RuntimeError):
        harness.assert_no_denied_ancestor(policy(resource), [resource])


def test_unrelated_sensitive_root_remains_denied(tmp_path):
    args = policy(tmp_path / "heldout-oracle")
    result = harness.assert_no_denied_ancestor(args, [tmp_path / "runtime/server.py"])
    assert result == [{"path": str(tmp_path / "heldout-oracle"), "access": "deny"}]
