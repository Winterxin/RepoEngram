"""Actual CLI/prepare/evolution/discovery wiring, with deterministic model fixtures."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_prepare import AntiPatternModel, WorkflowModel, _git
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import RecordingModel, complete_identity

from skillforge import cli
from skillforge.cost_control import BudgetExceeded
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.semantic_tendency import (
    ScopedTendencyDecision,
    TendencyDecisionKind,
    TendencyTransportDecision,
)
from skillforge.integration.autonomous import install
from skillforge.integration.mcp_server import DiscoveryTools
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository.semantic_tendencies import tendency_request_fingerprint


class TendencyModel:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ScopedTendencyDecision(tendency_request_fingerprint(request),
            TendencyTransportDecision(TendencyDecisionKind.INSUFFICIENT_EVIDENCE,
                "Insufficient supplied observations.",
                tuple(e.evidence_id for e in request.evidence)))


def test_normal_cli_commit_refresh_and_zero_model_following_task(
        historical_repository, tmp_path, monkeypatch, capsys):
    root = historical_repository
    models = [RecordingModel(), WorkflowModel(), AntiPatternModel(), TendencyModel()]
    for model in models:
        model.semantic_cache_identity = deepcopy(complete_identity())
    preparer = SkillForgePreparer(*models,
        semantic_cache_root=tmp_path.parent / (tmp_path.name + "-semantic-cache"))
    monkeypatch.setattr(cli, "SkillForgePreparer", lambda: preparer)

    def prepare(task):
        code = cli.main(["prepare", str(root), "--target", "codex", "--task", task,
                         "--current-module", "app.api.route0", "--json"])
        output = capsys.readouterr()
        assert code == 0, output.err
        result = json.loads(output.out)
        assert result["semantic_failure_count"] == 0
        return result

    first = prepare("Add coordinated routes")
    baseline = load_baseline(root, first["evolution_baseline_path"])
    assert baseline.source_compilation.skills
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    client = DiscoveryTools(root)
    before = client.call("get_project_summary", {})
    for index in range(3):
        path = root / f"src/app/api/route{index}.py"
        path.write_text(path.read_text("utf-8") + "NEXT_VALUE = 2\n", "utf-8")
    _git(root, "add", "src")
    _git(root, "commit", "-m", "Update coordinated behavior")
    assert client.call("get_project_summary", {})["knowledge_status"] == "STALE_HEAD"
    second = prepare("Update coordinated behavior")
    assert second["prepare_mode"] == "incremental"
    assert second["discovery_publication"]["status"] == "REFRESHED"
    assert second["discovery_publication"]["semantic_calls"] == 0
    following = load_baseline(root, Path(second["evolution_baseline_path"]))
    assert following.head_commit != baseline.head_commit
    after = client.call("get_project_summary", {})
    assert after["knowledge_status"] == "PUBLISHED_BASELINE"
    assert after["snapshot_id"] == following.source_dna.snapshot_id
    assert after["snapshot_id"] != before["snapshot_id"]
    skill = following.source_compilation.skills[0]
    route = client.call("get_related_workflow", {
        "task": skill.trigger.task_description, "scope": {"current_module": "app.api.route0"}})
    assert skill.skill_id in {item["skill_id"] for item in route["skills"]}
    actual = client.call("get_skill", {"skill_id": skill.skill_id})
    assert actual["version"] == skill.version
    calls_before = models[0].calls + sum(len(m.requests) for m in models[1:])
    third = prepare("A following task")
    calls_after = models[0].calls + sum(len(m.requests) for m in models[1:])
    assert calls_after == calls_before
    assert third["discovery_publication"]["status"] == "UNCHANGED"
    assert third["evolution_baseline_id"] == second["evolution_baseline_id"]

    original = preparer.prepare

    def accidental_model_call(*args, **kwargs):
        models[1].complete(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(preparer, "prepare", accidental_model_call)
    with pytest.raises(BudgetExceeded, match="attempted model transport"):
        automatic_prepare(PrepareRequest(root, "Following task", CodeAgentTarget.CODEX),
                          preparer=preparer)
    assert "complete" not in models[1].__dict__
    assert models[0].calls + sum(len(m.requests) for m in models[1:]) == calls_after
