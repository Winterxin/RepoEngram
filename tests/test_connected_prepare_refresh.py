"""Successful preparation updates existing discovery without extra semantic work."""

import json
from types import SimpleNamespace

import pytest
from test_incremental_evolution import draft, evolution_fixture, rejected_anti

from skillforge import cli
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.integration.autonomous import BOOTSTRAP, SNAPSHOT, install, refresh_connected
from skillforge.integration.mcp_server import DiscoveryTools
from skillforge.productization.evolution_artifacts import make_baseline, save_baseline


@pytest.fixture
def connected_delta(tmp_path):
    evolution, state, dna, compiled, _, _, target, plan = evolution_fixture(tmp_path)
    selected = [c for c in target.workflow_candidates.candidates
                if c.candidate_id in plan.workflow_refresh_ids]
    drafts = tuple(draft(c, name="Revised workflow") for c in selected)
    by_id = {e.evidence_id: e for e in target.workflow_candidates.evidence}
    refs = sorted({key for d in drafts for key in d.evidence_ids})
    semantic = SemanticWorkflowResult(dna.repository_id, drafts,
                                     tuple(by_id[key] for key in refs))
    applied = evolution.apply_semantic(dna, compiled, target, plan, semantic,
                                       rejected_anti(target, plan))
    empty = SemanticStructuralRuleResult(dna.repository_id, (), (), (), ())
    next_baseline = make_baseline(
        tmp_path, target.head_commit, target.max_commits,
        applied.resulting_project_dna, applied.resulting_compilation,
        applied.resulting_history_state, empty,
    )
    path = save_baseline(tmp_path, next_baseline)
    install(tmp_path, dna, compiled, state.head_commit)
    return tmp_path, next_baseline, path, applied.skill_version_changed_ids


def prepared_result(root, baseline, path, **overrides):
    fields = dict(repository_root=root, target=CodeAgentTarget.CODEX,
                  semantic_failure_count=0, evolution_baseline_path=path,
                  evolution_head_commit=baseline.head_commit,
                  dna_snapshot_id=baseline.source_dna.snapshot_id,
                  to_dict=lambda: {"prepare_mode": "incremental"})
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_cli_prepare_publishes_changed_knowledge_and_existing_discovery_reloads(
        connected_delta, monkeypatch, capsys):
    from skillforge.productization import verified_prepare

    root, baseline, path, changed = connected_delta
    client = DiscoveryTools(root)
    before = client.call("get_project_summary", {})
    assert before["knowledge_status"] == "STALE_HEAD"
    old_bytes = (root / SNAPSHOT).read_bytes()
    monkeypatch.setattr(verified_prepare, "automatic_prepare",
                        lambda *_a, **_k: prepared_result(root, baseline, path))
    assert cli.main(["prepare", str(root), "--task", "new work",
                     "--target", "codex", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["discovery_publication"]["status"] == "REFRESHED"
    assert printed["discovery_publication"]["semantic_calls"] == 0
    after = client.call("get_project_summary", {})
    assert after["knowledge_status"] == "PUBLISHED_BASELINE"
    assert after["snapshot_id"] == baseline.source_dna.snapshot_id
    assert (root / SNAPSHOT).read_bytes() != old_bytes
    skill = next(s for s in baseline.source_compilation.skills if s.skill_id == changed[0])
    route = client.call("get_related_workflow", {"task": skill.trigger.task_description})
    assert skill.skill_id in {s["skill_id"] for s in route["skills"]}
    read = client.call("get_skill", {"skill_id": skill.skill_id})
    assert read["version"] == skill.version
    assert any("Revised workflow" in text for text in read["files"].values())


def test_unconnected_project_is_not_registered_or_published(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = refresh_connected(tmp_path, "missing.json", expected_head="unused",
                               expected_snapshot_id="unused")
    assert result["status"] == "NOT_CONNECTED"
    assert not (tmp_path / ".codex").exists()
    assert not (tmp_path / SNAPSHOT).exists()


def test_same_snapshot_does_not_rewrite_or_reload_baseline(connected_delta):
    root, baseline, _, _ = connected_delta
    install(root, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    before = (root / SNAPSHOT).stat().st_mtime_ns
    result = refresh_connected(root, "missing.json", expected_head=baseline.head_commit,
                               expected_snapshot_id=baseline.source_dna.snapshot_id)
    assert result["status"] == "UNCHANGED"
    assert (root / SNAPSHOT).stat().st_mtime_ns == before


@pytest.mark.parametrize("mismatch", ["head", "snapshot"])
def test_mismatched_prepare_identity_preserves_old_publication(connected_delta, mismatch):
    root, baseline, path, _ = connected_delta
    before = (root / SNAPSHOT).read_bytes()
    with pytest.raises(ValueError, match="changed after prepare|identity mismatch"):
        refresh_connected(
            root, path, expected_head="wrong" if mismatch == "head" else baseline.head_commit,
            expected_snapshot_id="wrong" if mismatch == "snapshot"
            else baseline.source_dna.snapshot_id,
        )
    assert (root / SNAPSHOT).read_bytes() == before


def test_invalid_existing_metadata_is_not_overwritten(connected_delta):
    root, baseline, path, _ = connected_delta
    (root / SNAPSHOT).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid autonomous snapshot"):
        refresh_connected(root, path, expected_head=baseline.head_commit,
                          expected_snapshot_id=baseline.source_dna.snapshot_id)
    assert (root / SNAPSHOT).read_text("utf-8") == "{}"


def test_user_bootstrap_collision_preserves_publication(connected_delta):
    root, baseline, path, _ = connected_delta
    before = (root / SNAPSHOT).read_bytes()
    (root / BOOTSTRAP).write_text("User-owned instructions", encoding="utf-8")
    with pytest.raises(ValueError, match="bootstrap collision"):
        refresh_connected(root, path, expected_head=baseline.head_commit,
                          expected_snapshot_id=baseline.source_dna.snapshot_id)
    assert (root / SNAPSHOT).read_bytes() == before
    assert (root / BOOTSTRAP).read_text("utf-8") == "User-owned instructions"


@pytest.mark.parametrize("overrides", [
    {"semantic_failure_count": 1}, {"target": CodeAgentTarget.CLAUDE_CODE},
])
def test_partial_prepare_and_other_target_preserve_publication(
        connected_delta, monkeypatch, capsys, overrides):
    from skillforge.productization import verified_prepare

    root, baseline, path, _ = connected_delta
    before = (root / SNAPSHOT).read_bytes()
    monkeypatch.setattr(verified_prepare, "automatic_prepare",
                        lambda *_a, **_k: prepared_result(root, baseline, path, **overrides))
    assert cli.main(["prepare", str(root), "--task", "new work",
                     "--target", "codex", "--json"]) == 0
    assert "discovery_publication" not in json.loads(capsys.readouterr().out)
    assert (root / SNAPSHOT).read_bytes() == before
