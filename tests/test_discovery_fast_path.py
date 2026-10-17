"""Real zero-model discovery, missing metadata, stale HEAD and lazy loading."""

import json
import os
import subprocess
import sys

import pytest
from test_autonomous_mcp import installed as installed
from test_prepare_evolution_baseline import prepared as prepared
from test_prepare_evolution_baseline import repository as repository

from skillforge.integration.autonomous import BOOTSTRAP, CONFIG, SNAPSHOT, register
from skillforge.integration.mcp_server import DiscoveryTools, dispatch

EXPECTED_TOOL_NAMES = {
    "get_project_summary",
    "get_related_workflow",
    "search_project_rules",
    "get_skill",
    "get_evidence",
    "bind_verify_check",
    "verify_patch",
}


def forbid_models(monkeypatch):
    from skillforge.productization.prepare import SkillForgePreparer

    monkeypatch.setattr(
        SkillForgePreparer, "prepare", lambda *a, **k: pytest.fail("Model preparation in discovery")
    )


@pytest.fixture
def connected_repository(repository):
    subprocess.run(["git", "init", "-q", str(repository)], check=True, capture_output=True)
    return repository


def test_connect_without_snapshot_never_prepares_and_preserves_user_settings(
    connected_repository, monkeypatch
):
    repository = connected_repository
    forbid_models(monkeypatch)
    from skillforge import cli

    assert cli.main(["connect", str(repository)]) == 0
    assert not (repository / SNAPSHOT).exists()
    before = (repository / CONFIG).read_bytes()
    register(repository)
    assert (repository / CONFIG).read_bytes() == before
    service = DiscoveryTools(repository)
    assert service.service is None
    tools = dispatch(service, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"][
        "tools"
    ]
    assert len(tools) == len(EXPECTED_TOOL_NAMES)
    assert {tool["name"] for tool in tools} == EXPECTED_TOOL_NAMES
    assert service.service is None
    assert service.call("get_project_summary", {})["knowledge_status"] == "MISSING"
    assert service.service is None
    with pytest.raises(ValueError, match="metadata unavailable"):
        service.call("get_related_workflow", {"task": "Implement a route"})


def test_protocol_starts_without_snapshot_real_child(connected_repository, monkeypatch):
    repository = connected_repository
    forbid_models(monkeypatch)
    register(repository)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_project_summary", "arguments": {}},
        },
    ]
    process = subprocess.run(
        [sys.executable, "-m", "skillforge.integration.mcp_server", "--repo", str(repository)],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"),
        },
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == 3
    tools = replies[1]["result"]["tools"]
    assert len(tools) == len(EXPECTED_TOOL_NAMES)
    assert {tool["name"] for tool in tools} == EXPECTED_TOOL_NAMES
    status = json.loads(replies[2]["result"]["content"][0]["text"])
    assert status["knowledge_status"] == "MISSING" and status["semantic_calls"] == 0


def test_matching_published_snapshot_loads_once_and_never_prepares(installed, monkeypatch):
    forbid_models(monkeypatch)
    root, baseline, _ = installed
    import skillforge.integration.mcp_server as mcp

    original = mcp.load_snapshot
    calls = []

    def counted(root):
        calls.append(root)
        return original(root)

    monkeypatch.setattr(mcp, "load_snapshot", counted)
    service = DiscoveryTools(root)
    dispatch(service, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    dispatch(service, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert not calls
    assert service.call("get_project_summary", {})["knowledge_status"] == "PUBLISHED_BASELINE"
    service.call("get_related_workflow", {"task": "Implement a route"})
    service.call("search_project_rules", {"query": "Implement a route"})
    assert len(calls) == 1


def test_stale_head_is_disclosed_and_cannot_be_routed(installed, monkeypatch):
    forbid_models(monkeypatch)
    root, _, _ = installed
    service = DiscoveryTools(root)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--allow-empty", "-qm", "New commit"],
        check=True,
        capture_output=True,
    )
    assert service.call("get_project_summary", {})["knowledge_status"] == "STALE_HEAD"
    with pytest.raises(ValueError, match="stale"):
        service.call("get_related_workflow", {"task": "Implement a route"})


def test_atomic_republish_reloads_without_model(installed, monkeypatch):
    forbid_models(monkeypatch)
    root, baseline, _ = installed
    from skillforge.integration.autonomous import install

    service = DiscoveryTools(root)
    before = service.call("get_project_summary", {})
    original = service.service
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    assert service.call("get_project_summary", {}) == before
    assert service.service is not original


def test_user_bootstrap_collision_is_preserved(connected_repository):
    repository = connected_repository
    path = repository / BOOTSTRAP
    path.parent.mkdir(parents=True)
    path.write_text("User content")
    with pytest.raises(ValueError, match="collision"):
        register(repository)
    assert path.read_text() == "User content"
    assert not (repository / CONFIG).exists()
