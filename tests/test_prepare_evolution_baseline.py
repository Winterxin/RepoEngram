"""Baseline-backed prepare re-enters the normal Agent task lifecycle."""

import json
from pathlib import Path

import pytest
from test_incremental_evolution import draft, evolution_fixture, rejected_anti
from test_prepare import FakeModel, _git
from test_prepare import repository as _repository_fixture

from skillforge import cli
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralRuleResult,
)
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.errors import SkillForgeError
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.repo_local import RepoLocalSkillSync
from skillforge.productization.evolution_artifacts import (
    load_baseline,
    make_baseline,
    save_baseline,
)
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify.artifacts import load_prepare_receipt


@pytest.fixture
def repository(tmp_path):
    return _repository_fixture.__wrapped__(tmp_path)


@pytest.fixture
def prepared(repository):
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SkillForge Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "add", "src")
    _git(repository, "commit", "-qm", "base")
    first = SkillForgePreparer(
        FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE), object(), object()
    ).prepare(PrepareRequest(
        repository, "unrelated work", CodeAgentTarget.CODEX, "app.api.route0",
    ))
    assert first.prepare_mode == "rich"
    return first


def _baseline_prepare(repository, prepared, *, task="unrelated work",
                      module="app.api.route0", target=CodeAgentTarget.CODEX):
    return SkillForgePreparer().prepare(PrepareRequest(
        repository, task, target, module, Path(prepared.evolution_baseline_path),
    ))


def test_reentry_routes_syncs_and_freezes_new_receipt(repository, prepared, monkeypatch, capsys):
    from skillforge.productization import prepare as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("baseline prepare invoked rich discovery")

    for name in (
        "RepositoryAnalyzer", "GitHistoryMiner", "ChangePatternMiner",
        "WorkflowCandidateMiner", "NegativePatternMiner", "StructuralRuleMiner",
        "evaluate_semantic_knowledge", "CodexCliSemanticModel",
        "CodexCliAntiPatternSemanticModel", "CodexCliStructuralSemanticModel",
        "ProjectDNABuilder",
    ):
        monkeypatch.setattr(module, name, forbidden)
    baseline = load_baseline(repository, prepared.evolution_baseline_path)
    existing = list((repository / ".skillforge/productization/evolution/baselines").glob("*.json"))
    result = _baseline_prepare(repository, prepared)
    receipt = load_prepare_receipt(repository, result.verify_receipt_path)
    assert result.prepare_mode == "evolution_baseline"
    assert result.evolution_baseline_id == baseline.baseline_id
    assert result.evolution_baseline_path == prepared.evolution_baseline_path
    assert result.verify_base_commit == baseline.head_commit
    assert result.selected_skill_ids
    assert receipt.plan.source_snapshot_id == baseline.source_dna.snapshot_id
    assert receipt.plan.selected_skills == tuple(
        (skill.skill_id, skill.version) for skill in baseline.source_compilation.skills
    )
    package = repository / result.installed_packages[0]
    marker = json.loads((package / ".skillforge.json").read_text(encoding="utf-8"))
    assert marker["source_snapshot_id"] == baseline.source_dna.snapshot_id
    assert marker["source_skill_version"] == baseline.source_compilation.skills[0].version
    assert (package / "SKILL.md").is_file()
    assert set(receipt.ignored_patch_paths) == set(result.unchanged_files)
    baseline_dir = repository / ".skillforge/productization/evolution/baselines"
    assert list(baseline_dir.glob("*.json")) == existing
    assert cli.main([
        "prepare", str(repository), "--task", "unrelated work", "--target", "codex",
        "--current-module", "app.api.route0", "--baseline",
        prepared.evolution_baseline_path, "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["prepare_mode"] == "evolution_baseline"


def test_head_and_static_drift_fail_before_sync(repository, prepared, monkeypatch):
    from skillforge.productization import prepare as module

    original = (repository / prepared.installed_packages[0] / "SKILL.md").read_bytes()
    calls = []
    monkeypatch.setattr(module.RepoLocalSkillSync, "sync", lambda *_args: calls.append(1))
    monkeypatch.setattr(module.IncrementalEvolution, "make_state", lambda *_args: object())
    with pytest.raises(SkillForgeError, match="static facts differ"):
        _baseline_prepare(repository, prepared)
    assert calls == []
    assert (repository / prepared.installed_packages[0] / "SKILL.md").read_bytes() == original
    monkeypatch.undo()
    monkeypatch.setattr(module.RepoLocalSkillSync, "sync", lambda *_args: calls.append(1))
    (repository / "src/app/api/route0.py").write_text(
        "import app.service.unit1\n", encoding="utf-8",
    )
    _git(repository, "add", "src")
    _git(repository, "checkout", "--orphan", "rewritten")
    _git(repository, "commit", "-qm", "next")
    with pytest.raises(SkillForgeError, match="not an ancestor"):
        _baseline_prepare(repository, prepared)
    assert calls == []


def test_zero_route_cleans_stale_and_preserves_user_owned(repository, prepared):
    managed = repository / prepared.installed_packages[0]
    user_skill = repository / ".codex/skills/skillforge-personal"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("# Mine\n", encoding="utf-8")
    result = _baseline_prepare(repository, prepared, task="words without a matching trigger",
                               module=None)
    assert result.selected_skill_ids == ()
    assert result.installed_packages == ()
    assert not managed.exists()
    assert user_skill.is_dir()
    receipt = load_prepare_receipt(repository, result.verify_receipt_path)
    assert receipt.plan.selected_skills == ()
    assert set(receipt.ignored_patch_paths) == set(result.to_dict()["sync"]["removed_files"])


def test_claude_target_uses_claude_root(repository, prepared):
    result = _baseline_prepare(repository, prepared, target=CodeAgentTarget.CLAUDE_CODE)
    assert result.installed_packages[0].startswith(".claude/skills/")
    marker = json.loads((repository / result.installed_packages[0] /
                         ".skillforge.json").read_text(encoding="utf-8"))
    assert marker["target"] == "claude_code"
    assert marker["source_snapshot_id"] == result.dna_snapshot_id


def test_invalid_and_foreign_baselines_fail_before_sync(repository, prepared, tmp_path,
                                                        monkeypatch):
    from skillforge.productization import prepare as module

    calls = []
    monkeypatch.setattr(module.RepoLocalSkillSync, "sync", lambda *_args: calls.append(1))
    with pytest.raises(SkillForgeError, match="Invalid Evolution baseline"):
        SkillForgePreparer().prepare(PrepareRequest(
            repository, "task", CodeAgentTarget.CODEX, baseline=Path("missing.json"),
        ))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(SkillForgeError, match="Invalid Evolution baseline"):
        SkillForgePreparer().prepare(PrepareRequest(
            foreign, "task", CodeAgentTarget.CODEX,
            baseline=repository / prepared.evolution_baseline_path,
        ))
    assert calls == []


def test_evolved_snapshot_and_skill_content_reach_agent_package(tmp_path):
    evolution, state, source, compilation, delta, _, target, plan = evolution_fixture(
        tmp_path,
    )
    selected = [item for item in target.workflow_candidates.candidates
                if item.candidate_id in plan.workflow_refresh_ids]
    drafts = tuple(draft(item, name="Revised workflow") for item in selected)
    by_id = {item.evidence_id: item for item in target.workflow_candidates.evidence}
    refs = sorted({key for item in drafts for key in item.evidence_ids})
    semantic = SemanticWorkflowResult(
        source.repository_id, drafts, tuple(by_id[key] for key in refs),
    )
    result = evolution.apply_semantic(
        source, compilation, target, plan, semantic, rejected_anti(target, plan),
    )
    assert result.result_snapshot_id != source.snapshot_id
    assert result.skill_version_changed_ids
    empty_structural = SemanticStructuralRuleResult(source.repository_id, (), (), (), ())
    baseline_a = make_baseline(tmp_path, state.head_commit, state.max_commits,
                               source, compilation, state, empty_structural)
    baseline_b = make_baseline(
        tmp_path, target.head_commit, target.max_commits,
        result.resulting_project_dna, result.resulting_compilation,
        result.resulting_history_state, empty_structural,
    )
    save_baseline(tmp_path, baseline_a)
    path_b = save_baseline(tmp_path, baseline_b)
    changed_id = result.skill_version_changed_ids[0]
    old_skill = next(item for item in compilation.skills if item.skill_id == changed_id)
    new_skill = next(item for item in baseline_b.source_compilation.skills
                     if item.skill_id == changed_id)
    task = new_skill.trigger.task_description
    registry_a = SkillRegistryBuilder().build(compilation)
    route_a = SkillRouter().route(
        registry_a, SkillRoutingRequest(source.repository_id, task),
    )
    RepoLocalSkillSync().sync(
        tmp_path, AgentSkillExporter().plan(route_a, CodeAgentTarget.CODEX),
    )
    old_package = next(item for item in AgentSkillExporter().plan(
        route_a, CodeAgentTarget.CODEX,
    ).packages if item.source_skill_id == changed_id)
    package_dir = tmp_path / ".codex/skills" / old_package.agent_skill_name
    old_text = (package_dir / "SKILL.md").read_text(encoding="utf-8")
    assert old_skill.version != new_skill.version

    captured = {}
    original_build = SkillRegistryBuilder.build
    original_route = SkillRouter.route

    def capture_build(builder, compiled):
        registry = original_build(builder, compiled)
        captured["registry"] = registry
        return registry

    def capture_route(router, registry, request):
        route = original_route(router, registry, request)
        captured["route"] = route
        return route

    from unittest.mock import patch

    with patch.object(SkillRegistryBuilder, "build", capture_build), patch.object(
        SkillRouter, "route", capture_route,
    ):
        prepared_b = SkillForgePreparer().prepare(PrepareRequest(
            tmp_path, task, CodeAgentTarget.CODEX, baseline=Path(path_b),
        ))
    marker = json.loads((package_dir / ".skillforge.json").read_text(encoding="utf-8"))
    receipt = load_prepare_receipt(tmp_path, prepared_b.verify_receipt_path)
    snapshot = baseline_b.source_dna.snapshot_id
    assert captured["registry"].source_snapshot_id == snapshot
    assert captured["route"].source_snapshot_id == snapshot
    assert marker["source_snapshot_id"] == snapshot
    assert receipt.plan.source_snapshot_id == snapshot
    assert marker["source_skill_version"] == new_skill.version
    assert (package_dir / "SKILL.md").read_text(encoding="utf-8") != old_text
    assert "Revised workflow" in (package_dir / "SKILL.md").read_text(encoding="utf-8")
    installed = (package_dir / "SKILL.md").read_bytes()
    (tmp_path / "alpha_a.py").unlink()
    with pytest.raises(SkillForgeError, match="static facts differ"):
        SkillForgePreparer().prepare(PrepareRequest(
            tmp_path, task, CodeAgentTarget.CODEX, baseline=Path(path_b),
        ))
    assert (package_dir / "SKILL.md").read_bytes() == installed
