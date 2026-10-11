"""Production parity and selective refresh using local Git and deterministic ports."""

import json
import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from test_evolution_productization import _git as git_command
from test_prepare import AntiPatternModel, FakeModel, WorkflowModel
from test_semantic_tendencies import response_for
from test_static_testing import fixture as build_static_source

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.evidence import Support
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionKind
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.domain.semantic_workflow import SemanticDecisionKind
from skillforge.domain.structural_rules import DELEGATION_METHOD, TESTING_METHOD
from skillforge.errors import SkillForgeError
from skillforge.productization.evolution_artifacts import (
    load_baseline,
    load_plan_context,
    load_result,
    make_baseline,
    save_baseline,
)
from skillforge.productization.evolve_apply import evolve_apply
from skillforge.productization.evolve_plan import evolve_plan
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.history_patch_context import strip_git_patch_evidence
from skillforge.repository.incremental_evolution import IncrementalEvolution, select_decisions
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer


def _git(root, *args):
    if args[0] != "commit":
        return git_command(root, *args)
    try:
        count = int(git_command(root, "rev-list", "--count", "HEAD"))
    except subprocess.CalledProcessError:
        count = 0
    stamp = (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=count)).isoformat()
    with patch.dict(os.environ, {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}):
        return git_command(root, *args)


class StructuralModel(FakeModel):
    def __init__(self):
        super().__init__(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        # This deterministic decision works for all current structural families.
        response = super().complete(request)
        return replace(response, reason="The supplied current source observations were evaluated.")


class TendencyModel:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return response_for(request)


def models():
    return StructuralModel(), WorkflowModel(), AntiPatternModel(), TendencyModel()


def calls(ports):
    return sum(len(port.requests) for port in ports)


def full(root, ports):
    return SkillForgePreparer(*ports).prepare(
        PrepareRequest(
            root,
            "Add coordinated routes",
            CodeAgentTarget.CODEX,
            "app.api.route0",
        )
    )


def apply(root, baseline_path, ports):
    planned = evolve_plan(root, Path(baseline_path)).summary
    result = evolve_apply(
        root,
        Path(baseline_path),
        Path(planned["plan_artifact"]),
        structural_model=ports[0],
        workflow_model=ports[1],
        antipattern_model=ports[2],
        tendency_model=ports[3],
    ).summary
    source = load_baseline(root, baseline_path)
    loaded = load_plan_context(root, planned["plan_artifact"], source)
    restored = load_result(root, result["result_artifact"], source, loaded)
    following = load_baseline(root, result["next_baseline"])
    assert restored.resulting_project_dna == following.source_dna
    assert restored.production_knowledge == following.production_knowledge
    return result, following, loaded


def equivalent(root, expected, actual):
    assert actual.source_state == expected.source_state
    assert actual.production_knowledge == expected.production_knowledge
    assert actual.source_dna == expected.source_dna
    assert actual.source_compilation == expected.source_compilation
    registry_a = SkillRegistryBuilder().build(expected.source_compilation)
    registry_b = SkillRegistryBuilder().build(actual.source_compilation)
    assert registry_a == registry_b
    for task, module in (
        ("Add coordinated routes", "app.api.route0"),
        ("Add a public operation", "nebula.entry"),
        ("unrelated work", None),
    ):
        request = SkillRoutingRequest(expected.repository_id, task, current_module=module)
        assert SkillRouter().route(registry_a, request) == SkillRouter().route(registry_b, request)
    # Derived patch refs are additional; the complete raw source history must remain exact.
    assert strip_git_patch_evidence(actual.source_state.history) == GitHistoryMiner().mine(
        root,
        limit=actual.history_limit,
    )


@pytest.fixture
def rich_repo(tmp_path, monkeypatch):
    # A saturated bounded window makes unrelated history metadata unchanged.
    monkeypatch.setattr("skillforge.productization.prepare.HISTORY_LIMIT", 8)
    build_static_source(tmp_path)
    for group in range(4):
        for package in (f"api{group}", f"service{group}"):
            folder = tmp_path / "src/app" / package
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("")
        for index in range(3):
            (tmp_path / f"src/app/api{group}/route{index}.py").write_text(
                f"import app.service{group}.unit{index % 2}\nVALUE = 0\n",
            )
        for index in range(2):
            (tmp_path / f"src/app/service{group}/unit{index}.py").write_text("VALUE = 0\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "SkillForge Offline")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "Establish source facts")
    for index in range(9):
        (tmp_path / f"checkpoint_{index}.txt").write_text(str(index))
        _git(tmp_path, "add", f"checkpoint_{index}.txt")
        _git(tmp_path, "commit", "-qm", "Record checkpoint")
    for index in range(3):
        for name in ("route0", "route1"):
            path = tmp_path / f"src/app/api0/{name}.py"
            path.write_text(path.read_text() + f"VALUE_{index} = {index}\n")
        _git(tmp_path, "add", "src")
        _git(tmp_path, "commit", "-qm", "Fix coordinated routes")
    return tmp_path


@pytest.mark.parametrize("scenario", ["unrelated", "local", "multi"])
def test_performance_and_exact_production_parity(rich_repo, scenario, monkeypatch, capsys):
    root = rich_repo
    prepared_a = full(root, models())
    old = load_baseline(root, prepared_a.evolution_baseline_path)
    if scenario == "unrelated":
        (root / "one-off.md").write_text("A single documentation addition.")
    else:
        for group in (1,) if scenario == "local" else (1, 2):
            path = root / f"src/app/api{group}/route0.py"
            # Same candidate identity; source Evidence location changes.
            path.write_text("# relocated import\n" + path.read_text())
    _git(root, "add", ".", ":!.skillforge", ":!.codex")
    _git(root, "commit", "-qm", "Record local update")
    ports_b = models()
    prepared_b = full(root, ports_b)
    expected = load_baseline(root, prepared_b.evolution_baseline_path)
    delta_calls = []
    original = GitHistoryMiner._read_file_changes

    def only_new(self, path, commit):
        assert commit.commit_id not in {c.commit_id for c in old.source_state.history.changesets}
        delta_calls.append(commit.commit_id)
        return original(self, path, commit)

    monkeypatch.setattr(GitHistoryMiner, "_read_file_changes", only_new)
    ports_i = models()
    summary, actual, loaded = apply(root, prepared_a.evolution_baseline_path, ports_i)
    monkeypatch.setattr(GitHistoryMiner, "_read_file_changes", original)
    equivalent(root, expected, actual)
    assert len(delta_calls) == 1
    assert calls(ports_b) == summary["full_candidate_count"]
    assert calls(ports_i) == summary["incremental_refresh_count"]
    assert calls(ports_i) < calls(ports_b)
    if scenario == "unrelated":
        assert calls(ports_i) == 0, summary["refresh_decisions"]
        assert summary["skills"]["version_changed"] == 0
    else:
        assert calls(ports_i) == (1 if scenario == "local" else 2)
        assert not ports_i[1].requests and not ports_i[2].requests
        assert summary["skills"]["version_changed"] == calls(ports_i)
    reused = {
        item.candidate_id
        for item in loaded.plan.production.decisions
        if item.channel == "structural" and item.reason == "UNCHANGED_REUSED"
    }
    assert select_decisions(old.semantic_structural, reused) == select_decisions(
        actual.semantic_structural,
        reused,
    )
    print(
        json.dumps(
            {
                "scenario": scenario,
                "full_candidate_count": calls(ports_b),
                "incremental_refresh_count": calls(ports_i),
                "reused_knowledge_count": summary["reused_knowledge_count"],
                "semantic_calls_full": calls(ports_b),
                "semantic_calls_incremental": calls(ports_i),
                "reduction_percent": round(100 * (1 - calls(ports_i) / calls(ports_b)), 2),
            }
        )
    )
    # Keep the real measured numbers visible with pytest -s.
    output = capsys.readouterr().out
    with capsys.disabled():
        print(output.strip())


def test_four_channels_tendency_and_changed_history_refresh(rich_repo):
    root = rich_repo
    package = root / "src/app/plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Establish plugin package")
    for index in range(3):
        (package / f"unit_{index}.py").write_text("VALUE = 1\n")
        _git(root, "add", "src")
        _git(root, "commit", "-qm", "Add independent child module")
    first = full(root, models())
    baseline = load_baseline(root, first.evolution_baseline_path)
    assert baseline.production_knowledge.tendencies.accepted
    assert baseline.production_knowledge.workflows.workflows
    assert baseline.production_knowledge.anti_patterns.anti_patterns
    assert baseline.semantic_structural.accepted
    (package / "unit_3.py").write_text("VALUE = 2\n")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Add another independent child module")
    expected = load_baseline(root, full(root, models()).evolution_baseline_path)
    ports = models()
    summary, actual, _ = apply(root, first.evolution_baseline_path, ports)
    equivalent(root, expected, actual)
    assert ports[3].requests
    assert summary["semantic_refresh"]["tendency_candidates"] >= 1
    assert any(
        item["reason"] in {"SOURCE_EVIDENCE_CHANGED", "CANDIDATE_CHANGED"}
        for item in summary["refresh_decisions"]
        if item["channel"] == "tendency"
    )


@pytest.mark.parametrize("source", ["delegation", "testing", "structural"])
def test_removed_source_removes_rules_and_old_skills(rich_repo, source):
    root = rich_repo
    first = full(root, models())
    baseline = load_baseline(root, first.evolution_baseline_path)
    if source == "delegation":
        path = root / "src/nebula/entry.py"
        path.write_text(
            path.read_text().replace("return self.delegate.change_1(item_key)", "return item_key")
        )
        removed = {
            a.candidate_id
            for a in baseline.semantic_structural.accepted
            if a.candidate.method == DELEGATION_METHOD
        }
    elif source == "testing":
        (root / "tests/test_behavior.py").write_text("def test_unrelated():\n assert True\n")
        removed = {
            a.candidate_id
            for a in baseline.semantic_structural.accepted
            if a.candidate.method == TESTING_METHOD
        }
    else:
        for index in range(3):
            (root / f"src/app/api1/route{index}.py").write_text("VALUE = 1\n")
        removed = {
            a.candidate_id
            for a in baseline.semantic_structural.accepted
            if a.candidate.source_package == "app.api1"
        }
    assert removed
    _git(root, "add", "src", "tests")
    _git(root, "commit", "-qm", "Remove observed source relationship")
    expected = load_baseline(root, full(root, models()).evolution_baseline_path)
    summary, actual, _ = apply(root, first.evolution_baseline_path, models())
    equivalent(root, expected, actual)
    assert not removed & {r.rule_id for r in actual.source_dna.rules}
    assert summary["skills"]["removed"] >= len(removed)
    assert all(
        any(
            d["candidate_id"] == identifier and d["reason"] == "REMOVED_SOURCE"
            for d in summary["refresh_decisions"]
        )
        for identifier in removed
    )


def test_new_candidate_and_prepare_descendant_entry(rich_repo):
    root = rich_repo
    # Remove one family before A, then introduce the real source relationship at B.
    for index in range(3):
        path = root / f"src/app/api3/route{index}.py"
        path.write_text("VALUE = 0\n")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Remove a local relationship")
    first = full(root, models())
    for index in range(3):
        (root / f"src/app/api3/route{index}.py").write_text(
            f"import app.service3.unit{index % 2}\n",
        )
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "Introduce a local relationship")
    expected = load_baseline(root, full(root, models()).evolution_baseline_path)
    result = SkillForgePreparer(*models()).prepare(
        PrepareRequest(
            root,
            "Add coordinated routes",
            CodeAgentTarget.CODEX,
            baseline=Path(first.evolution_baseline_path),
        )
    )
    assert result.prepare_mode == "incremental"
    assert any(d["reason"] == "NEW_CANDIDATE" for d in result.evolution_refresh)
    actual = load_baseline(root, result.evolution_baseline_path)
    equivalent(root, expected, actual)
    second = SkillForgePreparer(*models()).prepare(
        PrepareRequest(
            root,
            "another task",
            CodeAgentTarget.CODEX,
            baseline=Path(result.evolution_baseline_path),
        )
    )
    assert second.prepare_mode == "evolution_baseline"


def test_uncommitted_static_drift_and_invalid_baseline_fail_before_models(rich_repo):
    first = full(rich_repo, models())
    planned = evolve_plan(rich_repo, Path(first.evolution_baseline_path)).summary
    path = rich_repo / "src/app/api1/route0.py"
    path.write_text("# changed Evidence\n" + path.read_text())
    ports = models()
    with pytest.raises(ValueError, match="static facts changed"):
        evolve_apply(
            rich_repo,
            Path(first.evolution_baseline_path),
            Path(planned["plan_artifact"]),
            structural_model=ports[0],
        )
    assert calls(ports) == 0
    with pytest.raises(SkillForgeError, match="static facts differ"):
        SkillForgePreparer(*ports).prepare(
            PrepareRequest(
                rich_repo,
                "task",
                CodeAgentTarget.CODEX,
                baseline=Path(first.evolution_baseline_path),
            )
        )
    assert calls(ports) == 0


def test_incremental_history_is_the_existing_evolution_state(rich_repo):
    first = full(rich_repo, models())
    baseline = load_baseline(rich_repo, first.evolution_baseline_path)
    facts = PythonStaticAnalyzer().analyze(rich_repo)
    assert (
        IncrementalEvolution().make_state(
            baseline.source_state.history,
            baseline.head_commit,
            baseline.history_limit,
            facts,
        )
        == baseline.source_state
    )


def test_rejected_decisions_are_reused_without_model_calls(rich_repo):
    def rejecting_ports():
        ports = (
            StructuralModel(),
            WorkflowModel(SemanticDecisionKind.INCOHERENT),
            AntiPatternModel(SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE),
            TendencyModel(),
        )
        ports[0].decision = SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE
        return ports

    first = full(rich_repo, rejecting_ports())
    (rich_repo / "one-off.md").write_text("A documentation addition.")
    _git(rich_repo, "add", "one-off.md")
    _git(rich_repo, "commit", "-qm", "Record documentation")
    expected = load_baseline(rich_repo, full(rich_repo, rejecting_ports()).evolution_baseline_path)
    ports = rejecting_ports()
    summary, actual, _ = apply(rich_repo, first.evolution_baseline_path, ports)
    equivalent(rich_repo, expected, actual)
    assert calls(ports) == 0 and summary["reused_knowledge_count"] > 0
    assert actual.production_knowledge.workflows.rejections
    assert actual.production_knowledge.anti_patterns.rejections
    assert actual.semantic_structural.rejections


def test_failed_refresh_is_isolated_and_retried_without_stale_rule(rich_repo):
    class FailingStructural(StructuralModel):
        def complete(self, request):
            if request.source_package == "app.api1":
                self.requests.append(request)
                raise RuntimeError("offline fixture channel failure")
            return super().complete(request)

    first = full(rich_repo, models())
    path = rich_repo / "src/app/api1/route0.py"
    path.write_text("# relocated Evidence\n" + path.read_text())
    _git(rich_repo, "add", "src")
    _git(rich_repo, "commit", "-qm", "Relocate one import")
    failing = (FailingStructural(), WorkflowModel(), AntiPatternModel(), TendencyModel())
    expected = load_baseline(rich_repo, full(rich_repo, failing).evolution_baseline_path)
    failing = (FailingStructural(), WorkflowModel(), AntiPatternModel(), TendencyModel())
    summary, actual, _ = apply(rich_repo, first.evolution_baseline_path, failing)
    equivalent(rich_repo, expected, actual)
    assert len(summary["semantic_failures"]) == 1
    assert summary["skills"]["removed"] == 1
    assert all(
        a.candidate.source_package != "app.api1" for a in actual.semantic_structural.accepted
    )
    retry = models()
    retried, repaired, _ = apply(rich_repo, summary["next_baseline"], retry)
    assert calls(retry) == 1
    assert any(d["reason"] == "MISSING_DECISION" for d in retried["refresh_decisions"])
    expected = load_baseline(rich_repo, full(rich_repo, models()).evolution_baseline_path)
    equivalent(rich_repo, expected, repaired)


def test_new_and_changed_workflow_candidates(rich_repo):
    first = full(rich_repo, models())
    for index in range(2):
        for group in (0, 2):
            for name in ("route0", "route1"):
                path = rich_repo / f"src/app/api{group}/{name}.py"
                path.write_text(path.read_text() + f"ADDED_{index} = {index}\n")
        _git(rich_repo, "add", "src")
        _git(rich_repo, "commit", "-qm", "Extend coordinated routes")
    expected = load_baseline(rich_repo, full(rich_repo, models()).evolution_baseline_path)
    summary, actual, _ = apply(rich_repo, first.evolution_baseline_path, models())
    equivalent(rich_repo, expected, actual)
    reasons = {d["reason"] for d in summary["refresh_decisions"] if d["channel"] == "workflow"}
    assert "NEW_CANDIDATE" in reasons
    assert reasons & {"SOURCE_EVIDENCE_CHANGED", "CANDIDATE_CHANGED"}
    assert summary["knowledge"]["workflows_refreshed"] >= 1
    assert summary["knowledge"]["workflows_added"] >= 1


def test_expired_history_removes_old_workflow_and_antipattern_knowledge(rich_repo):
    first = full(rich_repo, models())
    old = load_baseline(rich_repo, first.evolution_baseline_path)
    assert old.production_knowledge.anti_patterns.anti_patterns
    for index in range(8):
        name = f"later_{index}.md"
        (rich_repo / name).write_text("Independent documentation addition.")
        _git(rich_repo, "add", name)
        _git(rich_repo, "commit", "-qm", "Record documentation")
    expected = load_baseline(rich_repo, full(rich_repo, models()).evolution_baseline_path)
    ports = models()
    summary, actual, _ = apply(rich_repo, first.evolution_baseline_path, ports)
    equivalent(rich_repo, expected, actual)
    assert calls(ports) == 0
    assert not actual.production_knowledge.anti_patterns.anti_patterns
    assert summary["skills"]["removed"] >= 2
    assert summary["knowledge"]["workflows_unresolved"] >= 2


def test_legacy_forward_baseline_requires_explicit_full_prepare(rich_repo):
    first = full(rich_repo, models())
    baseline = load_baseline(rich_repo, first.evolution_baseline_path)
    # This source has Delegation; drop its synthetic-only static Workflow from the
    # legacy history state to model a genuine baseline from before that capability.
    from skillforge.domain.change import ChangePatternKind
    from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
    from skillforge.repository.project_dna import ProjectDNABuilder
    from skillforge.repository.skill_compiler import SkillCompiler

    candidate_result = baseline.source_state.workflow_candidates
    candidates = tuple(
        c for c in candidate_result.candidates if c.kind != ChangePatternKind.STATIC_DELEGATION
    )
    refs = {key for c in candidates for key in c.evidence_ids}
    state = replace(
        baseline.source_state,
        workflow_candidates=replace(
            candidate_result,
            candidates=candidates,
            evidence=tuple(e for e in candidate_result.evidence if e.evidence_id in refs),
        ),
    )
    structural = SemanticStructuralRuleResult(baseline.repository_id, (), (), (), ())
    dna = ProjectDNABuilder().build(baseline.repository_id)
    legacy = make_baseline(
        rich_repo,
        baseline.head_commit,
        baseline.history_limit,
        dna,
        SkillCompiler().compile(dna),
        state,
        structural,
    )
    supplied = save_baseline(rich_repo, legacy)
    (rich_repo / "independent.md").write_text("Documentation.")
    _git(rich_repo, "add", "independent.md")
    _git(rich_repo, "commit", "-qm", "Record documentation")
    with pytest.raises(ValueError, match="Legacy baseline.*full prepare"):
        evolve_plan(rich_repo, Path(supplied))


def test_same_snapshot_cache_can_serve_a_planned_incremental_refresh(rich_repo, tmp_path):
    from copy import deepcopy

    from test_repo_snapshot_semantic_cache import complete_identity

    cache = tmp_path.parent / (tmp_path.name + "-semantic-cache")

    def cached_full():
        ports = models()
        for port in ports:
            port.semantic_cache_identity = deepcopy(complete_identity())
        prepared = SkillForgePreparer(*ports, semantic_cache_root=cache).prepare(
            PrepareRequest(
                rich_repo,
                "task",
                CodeAgentTarget.CODEX,
            )
        )
        return prepared

    first = cached_full()
    path = rich_repo / "src/app/api1/route0.py"
    path.write_text("# moved import\n" + path.read_text())
    _git(rich_repo, "add", "src")
    _git(rich_repo, "commit", "-qm", "Relocate one import")
    expected = load_baseline(rich_repo, cached_full().evolution_baseline_path)
    ports = models()
    for port in ports:
        port.semantic_cache_identity = deepcopy(complete_identity())
    planned = evolve_plan(rich_repo, Path(first.evolution_baseline_path)).summary
    summary = evolve_apply(
        rich_repo,
        Path(first.evolution_baseline_path),
        Path(planned["plan_artifact"]),
        structural_model=ports[0],
        workflow_model=ports[1],
        antipattern_model=ports[2],
        tendency_model=ports[3],
        semantic_cache_root=cache,
    ).summary
    assert summary["incremental_refresh_count"] == 1
    assert summary["semantic_cache_hits"] == 1
    assert summary["semantic_calls_incremental"] == calls(ports) == 0
    equivalent(rich_repo, expected, load_baseline(rich_repo, summary["next_baseline"]))


def test_repository_change_during_semantic_refresh_does_not_publish(rich_repo):
    first = full(rich_repo, models())
    path = rich_repo / "src/app/api1/route0.py"
    path.write_text("# moved import\n" + path.read_text())
    _git(rich_repo, "add", "src")
    _git(rich_repo, "commit", "-qm", "Relocate one import")
    planned = evolve_plan(rich_repo, Path(first.evolution_baseline_path)).summary
    artifact_dir = rich_repo / ".skillforge/productization/evolution"
    existing = set(artifact_dir.rglob("*.json"))

    class ChangingModel(StructuralModel):
        def complete(self, request):
            response = super().complete(request)
            path.write_text("# changed during evaluation\n" + path.read_text())
            return response

    with pytest.raises(ValueError, match="changed during semantic refresh"):
        evolve_apply(
            rich_repo,
            Path(first.evolution_baseline_path),
            Path(planned["plan_artifact"]),
            structural_model=ChangingModel(),
        )
    assert set(artifact_dir.rglob("*.json")) == existing


def test_forged_refresh_reason_and_shadow_semantic_artifact_are_rejected(rich_repo):
    first = full(rich_repo, models())
    path = rich_repo / "src/app/api1/route0.py"
    path.write_text("# moved import\n" + path.read_text())
    _git(rich_repo, "add", "src")
    _git(rich_repo, "commit", "-qm", "Relocate one import")
    summary, _, loaded = apply(rich_repo, first.evolution_baseline_path, models())
    source = load_baseline(rich_repo, first.evolution_baseline_path)
    artifact = rich_repo / summary["result_artifact"]
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["semantic_workflows"]["workflows"] = []
    value["semantic_workflows"]["evidence"] = []
    value["semantic_workflows"]["rejections"] = []
    artifact.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic decision artifacts disagree"):
        load_result(rich_repo, summary["result_artifact"], source, loaded)
    plan_path = rich_repo / summary["plan"]
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    fresh = next(
        item
        for item in value["plan"]["production"]["decisions"]
        if item["reason"] != "UNCHANGED_REUSED"
    )
    fresh["reason"] = "UNCHANGED_REUSED"
    plan_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="production refresh decisions differ"):
        load_plan_context(rich_repo, plan_path, source)


def test_production_result_cannot_bypass_candidate_support_gate(rich_repo):
    first = full(rich_repo, models())
    source = load_baseline(rich_repo, first.evolution_baseline_path)
    for name in ("route0", "route1"):
        path = rich_repo / f"src/app/api0/{name}.py"
        path.write_text(path.read_text() + "ADDED = 1\n")
    _git(rich_repo, "add", "src")
    _git(rich_repo, "commit", "-qm", "Extend coordinated routes")
    _, actual, loaded = apply(rich_repo, first.evolution_baseline_path, models())
    knowledge = actual.production_knowledge
    draft = next(
        item
        for item in knowledge.workflows.workflows
        if item.candidate_id in loaded.plan.workflow_refresh_ids
    )
    support = Support(draft.support.supporting + 1, draft.support.examined + 1)
    forged = replace(draft, support=support, confidence=replace(draft.confidence, support=support))
    workflows = replace(
        knowledge.workflows,
        workflows=tuple(
            forged if item.draft_id == draft.draft_id else item
            for item in knowledge.workflows.workflows
        ),
    )
    with pytest.raises(ValueError, match="candidate"):
        IncrementalEvolution().apply_production_semantic(
            source.source_dna,
            source.source_compilation,
            loaded.target_state,
            loaded.plan,
            source.production_knowledge,
            replace(knowledge, workflows=workflows),
        )
