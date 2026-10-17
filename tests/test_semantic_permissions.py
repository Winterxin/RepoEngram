"""Host-to-child permission propagation, without any model or Agent execution."""

import subprocess
import sys
from dataclasses import replace

import pytest
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import complete_identity
from test_semantic_evaluation import _request

from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_permissions import (
    resolve_semantic_sandbox,
    semantic_permission_args,
)

ADAPTERS = (CodexCliSemanticModel, CodexCliAntiPatternSemanticModel,
            CodexCliStructuralSemanticModel, CodexCliTendencySemanticModel)


@pytest.mark.parametrize("kind", ADAPTERS)
@pytest.mark.parametrize("profile,mode", [
    (":danger-full-access", "danger-full-access"),
    (":workspace", "workspace-write"),
    (":read-only", "read-only"),
])
def test_native_adapters_inherit_declared_builtin_host_profile(monkeypatch, kind, profile, mode):
    monkeypatch.delenv("SKILLFORGE_SEMANTIC_SANDBOX", raising=False)
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", profile)
    model = kind()
    assert model._sandbox_mode == mode
    args = semantic_permission_args(model)
    assert args[:2] == ["--sandbox", mode]
    assert ('approval_policy="never"' in args) == (mode == "danger-full-access")
    # Neither an environment change nor baseline's forbidden-runner substitution
    # may make actual argv disagree with the already captured cache identity.
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", ":read-only")
    model._runner = lambda *a, **k: pytest.fail("model execution forbidden")
    assert semantic_permission_args(model) == args


def test_unset_parent_keeps_existing_restricted_default(monkeypatch):
    monkeypatch.delenv("CODEX_PERMISSION_PROFILE", raising=False)
    monkeypatch.delenv("SKILLFORGE_SEMANTIC_SANDBOX", raising=False)
    assert resolve_semantic_sandbox(subprocess.run) == "read-only"


@pytest.mark.parametrize("profile", ["", "unknown", "unseen-sensitive"])
def test_unknown_named_profile_stops_before_any_spawn(monkeypatch, profile):
    monkeypatch.delenv("SKILLFORGE_SEMANTIC_SANDBOX", raising=False)
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", profile)
    with pytest.raises(ValueError, match="Unsupported CODEX_PERMISSION_PROFILE"):
        CodexCliSemanticModel()


def test_explicit_policy_and_launcher_override(monkeypatch):
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", ":danger-full-access")
    monkeypatch.setenv("SKILLFORGE_SEMANTIC_SANDBOX", "read-only")
    assert CodexCliSemanticModel()._sandbox_mode == "read-only"
    assert CodexCliSemanticModel(sandbox_mode="workspace-write")._sandbox_mode == "workspace-write"
    monkeypatch.setenv("SKILLFORGE_SEMANTIC_SANDBOX", "")
    with pytest.raises(ValueError, match="Semantic sandbox must"):
        CodexCliSemanticModel()


@pytest.mark.parametrize("kind", ADAPTERS)
def test_custom_runner_policy_stays_frozen_under_full_access_host(monkeypatch, kind):
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", ":danger-full-access")
    monkeypatch.setenv("SKILLFORGE_SEMANTIC_SANDBOX", "danger-full-access")
    runner = lambda *a, **k: pytest.fail("model execution forbidden")
    assert semantic_permission_args(kind(runner=runner)) == ["--sandbox", "read-only"]


@pytest.mark.parametrize("mode", ["read-only", "workspace-write", "danger-full-access"])
@pytest.mark.parametrize("batch", [False, True])
def test_real_single_and_batch_command_use_same_captured_policy(mode, batch):
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        # Capture transport invocation only; never launch the real executable.
        return subprocess.CompletedProcess(command, 1, "", "ERROR: offline capture only")

    model = CodexCliSemanticModel(runner=capture, sandbox_mode=mode)
    if batch:
        from skillforge.cost_control import digest
        model.semantic_cache_identity = complete_identity(model="gpt-6-luna")
        model.semantic_cache_identity["permission_wrapper"] = {
            "version": "offline-permission-capture-v1",
            "config_sha256": digest(semantic_permission_args(model)),
        }
    request = _request()
    with pytest.raises(RuntimeError, match="exit"):
        if batch:
            model.complete_batch((request, replace(request, candidate_id="another-candidate")))
        else:
            model.complete(request)
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--sandbox") + 1] == mode
    assert ('approval_policy="never"' in command) == (mode == "danger-full-access")
    assert {"--ignore-user-config", "--ignore-rules", "--ephemeral", "--output-schema"} <= set(command)


def test_native_cache_identity_separates_permission_modes(monkeypatch, repository):
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-m", "fixture snapshot"],
                   cwd=repository, check=True, capture_output=True)
    monkeypatch.delenv("SKILLFORGE_SEMANTIC_SANDBOX", raising=False)
    monkeypatch.setattr("skillforge.repository.repo_snapshot_semantic_cache.shutil.which",
                        lambda _: sys.executable)
    cache = RepoSnapshotSemanticCache(repository)
    configs = []
    for profile in (":read-only", ":danger-full-access"):
        monkeypatch.setenv("CODEX_PERMISSION_PROFILE", profile)
        model = CodexCliSemanticModel()
        wrapped = cache.wrap(model)
        assert wrapped is not model
        identity = wrapped.config["declared_identity"]
        assert identity["permission_argv"] == semantic_permission_args(model)
        configs.append(wrapped.config)
    assert configs[0] != configs[1]
