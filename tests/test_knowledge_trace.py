"""Real export decisions are local, bounded and absent from model-facing content."""

import json

import pytest
from test_source_constraints import _export, _files, _route

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.integration.knowledge_trace import (
    MAX_EVENTS,
    MAX_TRACE_BYTES,
    TRACE_DIRECTORY,
    TRACE_SLOTS,
    KnowledgeTrace,
)


def _logs(root):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in (root / TRACE_DIRECTORY).glob("trace-*.json")]


def test_export_records_filter_and_delivery_without_source_or_literal_values(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("api_key = 'PRIVATE_CANARY'\ndef f():\n    return 0\n", encoding="utf-8")
    plan = _export(tmp_path)
    trace, = _logs(tmp_path)
    assert trace["summary"]["outcome"] == "EXPORTED"
    assert trace["summary"]["object_id"] == plan.plan_id
    reasons = {row["reason"] for row in trace["events"]}
    assert {"SENSITIVE_EFFECT_EXCLUDED", "PARSED", "SELECTED", "COLD_ALLOWANCE"} <= reasons
    assert "PRIVATE_CANARY" not in json.dumps(trace)
    assert "return 0" not in json.dumps(trace)
    assert "trace-" not in json.dumps(_files(plan))
    assert "run_id" not in json.dumps(_files(plan))


def test_no_package_explains_parse_failure_and_unavailable_path(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("def broken(:\n", encoding="utf-8")
    assert _export(tmp_path) is None
    trace, = _logs(tmp_path)
    assert trace["summary"]["outcome"] == "NO_PACKAGE"
    assert any(row["reason"] == "READ_OR_PARSE_FAILED" and row["object_id"] == "SyntaxError"
               for row in trace["events"])


def test_historical_match_does_not_create_source_fallback_diagnostics(tmp_path):
    missing = tmp_path / "not-created"
    assert _export(missing, _route(matches=True)) is None
    assert not missing.exists()


def test_budget_rejections_are_visible_without_exporting_candidate_text(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("token = 'hidden'\n" + "".join(
        f"def f{i}(value):\n    if value == {i}:\n        return {i}\n" for i in range(90)
    ), encoding="utf-8")
    assert _export(tmp_path) is not None
    trace, = _logs(tmp_path)
    assert any(row["reason"] == "PACKAGE_SIZE_LIMIT" for row in trace["events"])
    assert "hidden" not in json.dumps(trace)


def test_log_failure_does_not_change_export_success(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("def f():\n    return 0\n", encoding="utf-8")
    (tmp_path / ".skillforge").write_text("user-owned-file", encoding="utf-8")
    assert _export(tmp_path) is not None
    assert (tmp_path / ".skillforge").read_text() == "user-owned-file"


def test_persistent_slots_and_event_limit_keep_terminal_outcome(tmp_path):
    for number in range(TRACE_SLOTS + 3):
        trace = KnowledgeTrace("repo")
        # Deterministic slots exercise replacement, not a probabilistic collision.
        trace.run_id = f"{number:032x}"
        for index in range(MAX_EVENTS + 10):
            trace.record("observation", "SELECTED", object_id=str(index))
        trace.finish("export", "EXPORTED")
        trace.persist(tmp_path)
    paths = list((tmp_path / TRACE_DIRECTORY).glob("trace-*.json"))
    assert len(paths) == TRACE_SLOTS
    assert not list((tmp_path / TRACE_DIRECTORY).glob("*.tmp"))
    for path in paths:
        data = json.loads(path.read_text())
        assert len(data["events"]) <= MAX_EVENTS
        assert data["dropped_events"] >= 10
        assert data["summary"]["outcome"] == "EXPORTED"
        assert path.stat().st_size <= MAX_TRACE_BYTES


def test_diagnostic_symlink_cannot_write_outside_repository(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    try:
        (root / ".skillforge").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    trace = KnowledgeTrace("repo")
    trace.finish("export", "EXPORTED")
    trace.persist(root)
    assert not list(outside.iterdir())
    # A link above the repository root must be rejected as well.
    (outside / "repo").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    trace.persist(alias / "repo")
    assert not list((outside / "repo").iterdir())


def test_mcp_routing_export_and_receipt_share_one_local_trace(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_skill_registry import compilation, skill

    from skillforge.integration.mcp_server import ProjectTools
    from skillforge.repository.skill_registry import SkillRegistryBuilder

    registry = SkillRegistryBuilder().build(compilation((skill("history", task="create route"),)))
    dna = SimpleNamespace()
    monkeypatch.setattr("skillforge.integration.mcp_server.repo_root", lambda cwd: tmp_path)
    monkeypatch.setattr("skillforge.integration.mcp_server.load_snapshot",
                        lambda root: (dna, registry, "base"))
    monkeypatch.setattr("skillforge.integration.mcp_server._head", lambda root: "base")
    monkeypatch.setattr("skillforge.integration.mcp_server.discovery_with_risks",
                        lambda response, *args: response)
    monkeypatch.setattr("skillforge.integration.mcp_server.current_integration",
                        lambda *args: None)
    tools = ProjectTools(tmp_path)
    routed = tools.call("get_related_workflow", {"task": "create route"})
    first, = _logs(tmp_path)
    skill_id = routed["skills"][0]["skill_id"]
    content = tools.call("get_skill", {"skill_id": skill_id})
    receipt = tools.call("get_skill", {"skill_id": skill_id})
    final, = _logs(tmp_path)
    assert final["run_id"] == first["run_id"]
    reasons = {row["reason"] for row in final["events"]}
    assert {"SELECTED", "ROUTED_SKILL", "FILE_READY", "CONTENT_RETURNED",
            "ALREADY_READ_RECEIPT"} <= reasons
    assert content["files"]["SKILL.md"]
    assert receipt["files"] == {}
    assert "knowledge_trace" not in content


def test_prepare_stages_and_export_share_one_trace_without_live_model(tmp_path):
    from test_prepare import FakeModel

    from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
    from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer

    preparer = SkillForgePreparer(FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE))
    preparer.prepare(PrepareRequest(tmp_path, "inspect project", CodeAgentTarget.CODEX))
    trace, = _logs(tmp_path)
    assert trace["summary"]["operation"] == "prepare"
    assert trace["summary"]["outcome"] == "COMPLETE"
    stages = {row["stage"] for row in trace["events"]}
    assert {"static_analysis", "semantic_knowledge", "compilation", "export",
            "publication"} <= stages
    assert "inspect project" not in json.dumps(trace)


def test_prepare_failure_retains_stage_and_error_type_without_error_text(tmp_path, monkeypatch):
    from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer

    def fail(*args):
        raise RuntimeError("PRIVATE_CANARY")

    monkeypatch.setattr("skillforge.productization.prepare.RepositoryAnalyzer.analyze", fail)
    with pytest.raises(RuntimeError, match="PRIVATE_CANARY"):
        SkillForgePreparer().prepare(PrepareRequest(tmp_path, "hidden task", CodeAgentTarget.CODEX))
    trace, = _logs(tmp_path)
    assert trace["summary"]["outcome"] == "FAILED"
    assert trace["summary"]["object_id"] == "RuntimeError"
    assert trace["progress"]["stage"] == "static_analysis"
    assert "PRIVATE_CANARY" not in json.dumps(trace)
    assert "hidden task" not in json.dumps(trace)
