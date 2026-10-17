"""Frozen evolution apply: local Git fixtures and fake semantic models only."""

import json
from pathlib import Path

import pytest
from test_evolution_productization import _git, _prepare
from test_incremental_evolution import RejectModel, evolution_fixture

from skillforge import cli
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
)
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralResponse,
    SemanticStructuralRuleResult,
)
from skillforge.domain.skills import RuleGuidanceSkill
from skillforge.productization.evolution_artifacts import (
    load_baseline,
    load_plan_context,
    load_result,
    make_baseline,
    save_baseline,
    save_plan,
)
from skillforge.productization.evolve_apply import evolve_apply
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.incremental_evolution import IncrementalEvolution
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "SkillForge Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "add", "sample.py")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


class FailIfCalled:
    def complete(self, _request):
        raise AssertionError("unplanned semantic call")


class RejectAntiModel:
    def __init__(self):
        self.calls = []

    def complete(self, request):
        self.calls.append(request.candidate_id)
        return SemanticAntiPatternDecisionResponse(
            request.repository_id, request.candidate_id,
            SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
            "insufficient", tuple(item.evidence_id for item in request.evidence),
            0.2, None,
        )


def _planned_empty(root: Path):
    prepared = _prepare(root)
    assert cli.main(["evolve-plan", str(root), "--baseline",
                     prepared.evolution_baseline_path, "--json"]) == 0
    plan = next((root / ".skillforge/productization/evolution/plans").glob("*.json"))
    return prepared.evolution_baseline_path, plan.relative_to(root)


def test_zero_refresh_result_next_baseline_and_tamper(repo, capsys):
    baseline_path, plan_path = _planned_empty(repo)
    capsys.readouterr()
    workflow = FailIfCalled()
    anti = FailIfCalled()
    outcome = evolve_apply(
        repo,
        Path(baseline_path),
        plan_path,
        allow_legacy_v1=True,
        workflow_model=workflow,
        antipattern_model=anti,
    ).summary
    source = load_baseline(repo, baseline_path)
    loaded = load_plan_context(repo, plan_path, source)
    result = load_result(repo, outcome["result_artifact"], source, loaded)
    following = load_baseline(repo, outcome["next_baseline"])
    assert result.resulting_project_dna == source.source_dna
    assert result.resulting_compilation == source.source_compilation
    assert following.source_dna == result.resulting_project_dna
    assert following.source_compilation == result.resulting_compilation
    assert following.source_state == loaded.target_state
    assert following.head_commit == loaded.plan.target_head_commit
    assert following.semantic_structural == source.semantic_structural
    assert cli.main(["evolve-apply", str(repo), "--baseline", baseline_path,
                     "--plan", str(plan_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result_id"] == result.result_id
    assert cli.main(["evolve-plan", str(repo), "--baseline",
                     outcome["next_baseline"], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["new_commits"] == 0
    result_path = repo / outcome["result_artifact"]
    original = result_path.read_bytes()
    tampered = json.loads(original)
    tampered["result"]["skill_unchanged_ids"] = ["forged"]
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        load_result(repo, outcome["result_artifact"], source, loaded)
    result_path.write_bytes(original)


def test_head_and_static_drift_rejected_before_semantic(tmp_path, capsys):
    _, state, dna, compiled, delta, _, target, plan = evolution_fixture(tmp_path)
    semantic = SemanticStructuralRuleResult(dna.repository_id, (), (), (), ())
    baseline = make_baseline(tmp_path, state.head_commit, state.max_commits,
                             dna, compiled, state, semantic)
    baseline_path = save_baseline(tmp_path, baseline)
    plan_path = save_plan(tmp_path, baseline, target, delta, plan, None, {})
    (tmp_path / "alpha_a.py").write_text("import alpha_b\nnumber = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="static facts changed"):
        evolve_apply(
            tmp_path,
            Path(baseline_path),
            Path(plan_path),
            allow_legacy_v1=True,
            workflow_model=FailIfCalled(),
            antipattern_model=FailIfCalled(),
        )
    assert not list((tmp_path / ".skillforge/productization/evolution").glob("results/*.json"))
    _git(tmp_path, "add", "alpha_a.py")
    _git(tmp_path, "commit", "-qm", "changed")
    with pytest.raises(ValueError, match="HEAD changed"):
        evolve_apply(
            tmp_path,
            Path(baseline_path),
            Path(plan_path),
            allow_legacy_v1=True,
            workflow_model=FailIfCalled(),
            antipattern_model=FailIfCalled(),
        )
    assert cli.main(["evolve-apply", str(tmp_path), "--baseline", baseline_path,
                     "--plan", plan_path]) == 2
    assert "rerun evolve-plan" in capsys.readouterr().err


def test_selective_refresh_and_legal_rejection(tmp_path):
    _, state, dna, compiled, delta, _, target, plan = evolution_fixture(tmp_path)
    semantic = SemanticStructuralRuleResult(dna.repository_id, (), (), (), ())
    baseline = make_baseline(tmp_path, state.head_commit, state.max_commits,
                             dna, compiled, state, semantic)
    baseline_path = save_baseline(tmp_path, baseline)
    plan_path = save_plan(tmp_path, baseline, target, delta, plan, None, {})
    workflow = RejectModel()
    anti = RejectAntiModel()
    outcome = evolve_apply(
        tmp_path,
        Path(baseline_path),
        Path(plan_path),
        allow_legacy_v1=True,
        workflow_model=workflow,
        antipattern_model=anti,
    ).summary
    assert tuple(workflow.calls) == plan.workflow_refresh_ids
    assert tuple(anti.calls) == plan.antipattern_refresh_ids
    loaded = load_plan_context(tmp_path, plan_path, baseline)
    result = load_result(tmp_path, outcome["result_artifact"], baseline, loaded)
    assert result.resulting_project_dna.rules == dna.rules
    assert result.resulting_project_dna == dna
    assert result.unresolved_workflow_ids
    assert result.skill_unchanged_ids == tuple(sorted(item.skill_id for item in compiled.skills))


def test_invalid_semantic_response_saves_nothing(tmp_path):
    _, state, dna, compiled, delta, _, target, plan = evolution_fixture(tmp_path)
    semantic = SemanticStructuralRuleResult(dna.repository_id, (), (), (), ())
    baseline = make_baseline(tmp_path, state.head_commit, state.max_commits,
                             dna, compiled, state, semantic)
    baseline_path = save_baseline(tmp_path, baseline)
    plan_path = save_plan(tmp_path, baseline, target, delta, plan, None, {})

    class Invalid:
        def complete(self, _request):
            return object()

    with pytest.raises((ValueError, TypeError)):
        evolve_apply(
            tmp_path,
            Path(baseline_path),
            Path(plan_path),
            allow_legacy_v1=True,
            workflow_model=Invalid(),
            antipattern_model=FailIfCalled(),
        )
    assert not list((tmp_path / ".skillforge/productization/evolution").glob("results/*.json"))


def test_structural_rule_guidance_survives_apply(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "SkillForge Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    for package in ("api", "service"):
        path = tmp_path / "src" / "app" / package / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for index in range(3):
        (tmp_path / "src" / "app" / "api" / f"route{index}.py").write_text(
            f"import app.service.unit{index % 2}\n", encoding="utf-8")
    for index in range(2):
        (tmp_path / "src" / "app" / "service" / f"unit{index}.py").write_text(
            "", encoding="utf-8")
    _git(tmp_path, "add", "src")
    _git(tmp_path, "commit", "-qm", "base")
    static = PythonStaticAnalyzer().analyze(tmp_path)
    mined = StructuralRuleMiner().mine(static)
    request = build_semantic_structural_request(mined, mined.candidates[0])
    refs = tuple(item.evidence_id for item in request.evidence)
    response = SemanticStructuralResponse(
        request.repository_id, request.candidate_id,
        SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
        "Observed imports connect app.api to app.service across source modules.",
        refs, request.source_package, request.target_package,
        request.source_modules, refs, (),
    )

    class StructuralModel:
        def complete(self, _request):
            return response

    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, StructuralModel())
    dna = ProjectDNABuilder().build(mined.repository_id,
                                    semantic_structural_rules=semantic)
    compiled = SkillCompiler().compile(dna, semantic)
    assert any(isinstance(item, RuleGuidanceSkill) for item in compiled.skills)
    head = _git(tmp_path, "rev-parse", "HEAD")
    state = IncrementalEvolution().make_state(
        GitHistoryMiner().mine(tmp_path, 10), head, 10, static)
    baseline = make_baseline(tmp_path, head, 10, dna, compiled, state, semantic)
    baseline_path = save_baseline(tmp_path, baseline)
    delta = GitHistoryMiner().mine_delta(tmp_path, head, head)
    plan = IncrementalEvolution().plan(dna, compiled, state, state, delta,
                                       source_semantic_structural=semantic)
    plan_path = save_plan(tmp_path, baseline, state, delta, plan, None, {})
    summary = evolve_apply(
        tmp_path,
        Path(baseline_path),
        Path(plan_path),
        allow_legacy_v1=True,
        workflow_model=FailIfCalled(),
        antipattern_model=FailIfCalled(),
    ).summary
    result = load_result(tmp_path, summary["result_artifact"], baseline,
                         load_plan_context(tmp_path, plan_path, baseline))
    assert result.resulting_compilation == compiled
    assert result.resulting_project_dna.rules == dna.rules
    assert load_baseline(tmp_path, summary["next_baseline"]).semantic_structural == semantic
