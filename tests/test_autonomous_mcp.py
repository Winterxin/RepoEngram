"""Offline discovery, real stdio transport, production routing and Verify integration."""

import json
import os
import subprocess
import sys
import tomllib

import pytest
from test_prepare_evolution_baseline import prepared, repository  # noqa: F401

from skillforge.integration.autonomous import (
    BOOTSTRAP,
    CONFIG,
    SNAPSHOT,
    codex_registration_overrides,
    configure_codex_approval,
    install,
)
from skillforge.integration.mcp_server import TOOLS, ProjectTools, dispatch
from skillforge.productization.evolution_artifacts import load_baseline


@pytest.fixture
def installed(repository, prepared):  # noqa: F811
    baseline = load_baseline(repository, prepared.evolution_baseline_path)
    result = install(repository, baseline.source_dna, baseline.source_compilation,
                     baseline.head_commit)
    return repository, baseline, result


def test_install_idempotent_preserves_config_and_no_target_injection(installed):
    root, baseline, result = installed
    config = tomllib.loads((root / CONFIG).read_text("utf-8"))
    assert config["mcp_servers"]["skillforge"]["args"][-1] == str(root)
    assert not result["target_skill_preinjected"]
    text = (root / BOOTSTRAP).read_text("utf-8")
    for skill in baseline.source_compilation.skills:
        assert skill.skill_id not in text
        assert skill.procedure.steps[0].instruction not in text
    old = (root / CONFIG).read_bytes()
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    assert old == (root / CONFIG).read_bytes()
    overrides = " ".join(codex_registration_overrides(root))
    assert "mcp_servers.skillforge.command" in overrides
    assert "projects." not in overrides
    assert all(skill.skill_id not in overrides for skill in baseline.source_compilation.skills)


def test_conflicting_registration_preserves_files(installed):
    root, baseline, _ = installed
    (root / CONFIG).write_text('[mcp_servers.skillforge]\ncommand="user-owned"\n', "utf-8")
    before = (root / SNAPSHOT).read_bytes()
    with pytest.raises(ValueError, match="conflicts"):
        install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    assert before == (root / SNAPSHOT).read_bytes()


def test_metadata_does_not_claim_verify_commands_are_read_only():
    for item in TOOLS:
        assert item["annotations"]["readOnlyHint"] == (
            item["name"] not in {"verify_patch", "bind_verify_check"})
    verify = next(item for item in TOOLS if item["name"] == "verify_patch")
    assert verify["annotations"]["openWorldHint"]
    assert verify["annotations"]["destructiveHint"]


def test_opt_in_project_tool_approval_preserves_other_policy_and_reinstall(installed):
    root, baseline, _ = installed
    original = (root / CONFIG).read_text("utf-8")
    original += '\n[mcp_servers.other]\ncommand="other"\ndefault_tools_approval_mode="prompt"\n'
    (root / CONFIG).write_text(original, "utf-8")
    result = configure_codex_approval(root, "approve", ("get_related_workflow", "verify_patch"))
    after = (root / CONFIG).read_text("utf-8")
    assert after.startswith(original)
    parsed = tomllib.loads(after)
    assert parsed["mcp_servers"]["other"]["default_tools_approval_mode"] == "prompt"
    assert set(parsed["mcp_servers"]["skillforge"]["tools"]) == {
        "get_related_workflow", "verify_patch"}
    assert not result["global_settings_modified"]
    overrides = " ".join(codex_registration_overrides(root))
    assert 'mcp_servers.skillforge.tools.verify_patch.approval_mode="approve"' in overrides
    assert "approval_policy=" not in overrides
    configure_codex_approval(root, "approve", ("get_related_workflow", "verify_patch"))
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    assert (root / CONFIG).read_text("utf-8") == after
    with pytest.raises(ValueError, match="conflicts"):
        configure_codex_approval(root, "prompt", ("verify_patch",))
    with pytest.raises(ValueError, match="supported"):
        configure_codex_approval(root, "approve", ("other-tool",))
    assert (root / CONFIG).read_text("utf-8") == after


def test_real_route_skill_read_evidence_and_verify(installed):
    root, baseline, _ = installed
    service = ProjectTools(root / "src")
    with pytest.raises(ValueError, match="before"):
        service.call("get_skill", {"skill_id": "unknown"})
    related = service.call("get_related_workflow", {
        "task": "Implement a route", "scope": {"current_module": "app.api.route0"}})
    assert related["repository_id"] == baseline.repository_id
    assert related["router_trace"]["registry_id"] == service.registry.registry_id
    assert related["skills"]
    with pytest.raises(ValueError, match="read all"):
        service.call("verify_patch", {})
    for selected in related["skills"]:
        value = service.call("get_skill", {"skill_id": selected["skill_id"]})
        assert value["completion"]["tool"] == "verify_patch"
        assert value["completion"]["arguments"]["working_tree"] is True
        assert "Decision procedure" in value["files"]["SKILL.md"]
        assert service.call("get_evidence", {"rule_id": selected["skill_id"]})
    with pytest.raises(ValueError, match="not selected"):
        service.call("get_skill", {"skill_id": "foreign-skill"})
    service.call("search_project_rules", {"query": "unrelated"})
    checks_path = root / ".skillforge/autonomous/checks.json"
    checks_path.write_text(json.dumps({"checks": [{"check_id": "real-test", "kind": "TEST",
        "argv": [sys.executable, "-c", "raise SystemExit(1)"], "timeout_seconds": 10}]}), "utf-8")
    (root / "src/app/new.py").write_text("x = 1\n", "utf-8")
    result = service.call("verify_patch", {})
    assert result["report"]["checks"][0]["status"] == "FAILED"
    assert any(item["path"] == "src/app/new.py" for item in result["report"]["patch"]["files"])
    assert (root / result["artifact"]).is_file()


def test_skill_preview_uses_same_canonical_targets_and_does_not_cover_them(installed):
    from skillforge.verify import VerifyPlanBuilder
    root, _, _ = installed
    service = ProjectTools(root)
    related = service.call('get_related_workflow', {
        'task': 'Implement a route', 'scope': {'current_module': 'app.api.route0'}})
    for selected in related['skills']:
        value = service.call('get_skill', {'skill_id': selected['skill_id']})
        preview = value['completion']
        plan = VerifyPlanBuilder().for_skills(service.route, (selected['skill_id'],))
        expected = {(o.target_ref.kind.value, o.target_ref.ref_id) for o in plan.obligations}
        exposed = {(kind, ref) for kind, refs in preview['targets'].items() for ref in refs}
        assert exposed <= expected
        assert len(exposed) + preview['omitted_targets'] == len(expected)
        assert plan.checks == ()
        repeated = service.call('get_skill', {'skill_id': selected['skill_id']})
        assert repeated['files'] == {}
        assert repeated['completion'] == preview
    report = service.call('verify_patch', {})['report']
    assert report['overall_status'] == 'INCOMPLETE'
    assert report['checks'] == []


def test_formal_verify_catches_old_assertion_rewrite_even_when_configured_check_passes(installed):
    root, baseline, _ = installed
    folder = root / 'tests'
    folder.mkdir(exist_ok=True)
    target = folder / 'test_contract.py'
    target.write_text('def test_contract():\n    assert result == "original"\n', 'utf-8')
    subprocess.run(['git', 'add', str(target)], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'test expectations baseline'], cwd=root, check=True)
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    install(root, baseline.source_dna, baseline.source_compilation, base)
    service = ProjectTools(root)
    related = service.call('get_related_workflow', {
        'task': 'Implement a route', 'scope': {'current_module': 'app.api.route0'}})
    for selected in related['skills']:
        service.call('get_skill', {'skill_id': selected['skill_id']})
    checks_path = root / '.skillforge/autonomous/checks.json'
    checks_path.write_text(json.dumps({'checks': [{'check_id': 'passing-configured-check',
        'kind': 'TEST', 'argv': [sys.executable, '-c', 'pass']}]}), 'utf-8')
    target.write_text('def test_contract():\n    assert result == "changed"\n', 'utf-8')
    report = service.call('verify_patch', {})['report']
    checks = {check['check_id']: check for check in report['checks']}
    assert checks['passing-configured-check']['status'] == 'PASSED'
    guard = checks['zz-skillforge-preserve-test-expectations']
    assert guard['status'] == 'FAILED'
    assert guard['covered_refs'] == []
    assert report['overall_status'] == 'FAILED'


def test_repository_isolation_and_missing_identity(installed, tmp_path):
    root, _, _ = installed
    value = json.loads((root / SNAPSHOT).read_text("utf-8"))
    value["repository_id"] = "foreign"
    (root / SNAPSHOT).write_text(json.dumps(value), "utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        ProjectTools(root)
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q", str(empty)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="metadata unavailable"):
        ProjectTools(empty)


def test_stdio_registration_executable_trace_and_protocol(installed):
    root, _, _ = installed
    config = tomllib.loads((root / CONFIG).read_text("utf-8"))["mcp_servers"]["skillforge"]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "get_project_summary", "arguments": {}}},
    ]
    result = subprocess.run([config["command"], *config["args"]], cwd=root,
                            env={**os.environ, **config["env"]},
                            input="\n".join(json.dumps(item) for item in requests) + "\n",
                            encoding="utf-8", capture_output=True, timeout=30, check=True)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 3
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert len(responses[1]["result"]["tools"]) == len(TOOLS)
    assert not responses[2]["result"]["isError"]
    trace = root / ".skillforge/productization/verify/mcp-trace.jsonl"
    assert len(trace.read_text("utf-8").splitlines()) == len(requests)


@pytest.mark.parametrize("message", [[], {"jsonrpc": "1.0"},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []}])
def test_malformed_protocol_returns_error(message):
    assert "error" in dispatch(None, message)
