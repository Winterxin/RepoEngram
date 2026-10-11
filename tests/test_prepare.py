"""Normal-usage orchestration without live model execution."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from skillforge import cli
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.change import ChangePatternKind
from skillforge.domain.project_dna import ProjectKnowledgeStatus
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
    SemanticAntiPatternResponse,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralResponse,
)
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticEntityReferences,
    SemanticStepClassification,
    SemanticWorkflowResponse,
    SemanticWorkflowRole,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.errors import InvalidRepositoryPath
from skillforge.productization.evolution_artifacts import load_baseline, load_plan
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.identity import repository_identity
from skillforge.verify.artifacts import load_prepare_receipt


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for package in ("api", "service"):
        path = tmp_path / "src" / "app" / package / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for index in range(3):
        (tmp_path / "src" / "app" / "api" / f"route{index}.py").write_text(
            f"import app.service.unit{index % 2}\n", encoding="utf-8"
        )
    for index in range(2):
        (tmp_path / "src" / "app" / "service" / f"unit{index}.py").write_text(
            "", encoding="utf-8"
        )
    return tmp_path


class FakeModel:
    def __init__(self, decision: SemanticStructuralDecisionKind) -> None:
        self.decision = decision

    def complete(self, request):
        refs = tuple(item.evidence_id for item in request.evidence)
        accepted = self.decision == SemanticStructuralDecisionKind.ARCHITECTURE_RULE
        return SemanticStructuralResponse(
            request.repository_id, request.candidate_id, self.decision,
            "Observed imports connect the source and target packages." if accepted
            else "The observations are insufficient for a rule.",
            refs, request.source_package, request.target_package,
            request.source_modules if accepted else (), refs if accepted else (), (),
        )


def _prepare(repository, decision, *, target=CodeAgentTarget.CODEX,
             current_module="app.api.route0"):
    return SkillForgePreparer(FakeModel(decision)).prepare(PrepareRequest(
        repository, "unrelated lexical wording", target, current_module,
    ))


def test_positive_structural_prepare_and_idempotency(repository):
    first = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    assert first.structural_candidate_count == 1
    assert first.semantic_accepted_count == 1
    assert first.compiled_skill_count == 1
    assert len(first.selected_skill_ids) == 1
    assert len(first.installed_packages) == 1
    package = repository / first.installed_packages[0]
    assert (package / "SKILL.md").is_file()
    assert (package / "references" / "evidence.md").is_file()
    assert (package / ".skillforge.json").is_file()
    second = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    assert second.written_files == ()
    assert len(second.unchanged_files) == 3
    assert second.removed_packages == ()


def test_git_prepare_persists_deterministic_exact_verify_seed(repository, monkeypatch, capsys):
    from skillforge.verify import VerifyPlanBuilder

    actual_build = VerifyPlanBuilder.build
    seeds = []

    def capture_seed(builder, route, checks=()):
        seed = actual_build(builder, route, checks)
        seeds.append(seed)
        return seed

    monkeypatch.setattr(VerifyPlanBuilder, "build", capture_seed)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SkillForge Test")
    _git(repository, "config", "user.email", "skillforge@example.invalid")
    _git(repository, "add", "src")
    _git(repository, "commit", "-qm", "base")
    base = _git(repository, "rev-parse", "HEAD")

    first = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    baseline_artifact = load_baseline(repository, first.evolution_baseline_path)
    assert baseline_artifact.source_compilation.source_snapshot_id == first.dna_snapshot_id
    assert len(baseline_artifact.source_compilation.skills) == first.compiled_skill_count
    assert baseline_artifact.source_state.head_commit == base
    baseline_bytes = (repository / first.evolution_baseline_path).read_bytes()
    assert cli.main(["evolve-plan", str(repository), "--baseline",
                     first.evolution_baseline_path, "--json"]) == 0
    empty_plan_summary = json.loads(capsys.readouterr().out)
    assert empty_plan_summary["new_commits"] == 0
    assert load_plan(repository, empty_plan_summary["plan_artifact"],
                     baseline_artifact).plan_id == empty_plan_summary["plan_id"]
    receipt_path = repository / first.verify_receipt_path
    original = receipt_path.read_bytes()
    receipt = load_prepare_receipt(repository, first.verify_receipt_path)
    assert receipt.plan == seeds[0]
    assert first.verify_available is True
    assert first.verify_plan_id == receipt.plan.plan_id
    assert first.verify_base_commit == base == receipt.base_commit
    assert receipt.repository_id == repository_identity(repository)
    assert receipt.plan.repository_id == first.repository_id
    assert receipt.registry_id == receipt.plan.registry_id
    assert receipt.source_snapshot_id == receipt.plan.source_snapshot_id
    assert receipt.plan.source_snapshot_id == first.dna_snapshot_id
    assert tuple(skill_id for skill_id, _ in receipt.plan.selected_skills) == (
        first.selected_skill_ids
    )
    marker = json.loads(
        (repository / first.installed_packages[0] / ".skillforge.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt.plan.selected_skills == (
        (marker["source_skill_id"], marker["source_skill_version"]),
    )
    assert receipt.plan.checks == ()
    assert tuple(item.evidence_id for item in receipt.plan.evidence) == tuple(sorted(
        {evidence_id for item in receipt.plan.obligations for evidence_id in item.evidence_ids}
    ))

    second = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    assert second.evolution_baseline_id == first.evolution_baseline_id
    assert (repository / second.evolution_baseline_path).read_bytes() == baseline_bytes
    assert receipt.plan == seeds[1]
    assert second.verify_receipt_path == first.verify_receipt_path
    assert (repository / second.verify_receipt_path).read_bytes() == original

    empty = _prepare(repository, SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE)
    assert seeds[2].selected_skills == ()
    empty_receipt = load_prepare_receipt(repository, empty.verify_receipt_path)
    assert empty.verify_available is True
    assert empty_receipt.plan.selected_skills == ()
    assert empty_receipt.plan.obligations == ()
    assert empty_receipt.plan.evidence == ()
    assert empty.verify_receipt_path != first.verify_receipt_path


@pytest.mark.parametrize(("target", "target_dir"), [
    (CodeAgentTarget.CODEX, ".codex"),
    (CodeAgentTarget.CLAUDE_CODE, ".claude"),
])
def test_git_prepare_verify_excludes_only_managed_agent_skill_files(
    repository, target, target_dir, capsys,
):
    user_skill = repository / target_dir / "skills" / "my-own-skill" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("user version one", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SkillForge Test")
    _git(repository, "config", "user.email", "skillforge@example.invalid")
    _git(repository, "add", "src", target_dir)
    _git(repository, "commit", "-qm", "base")

    result = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
                      target=target)
    receipt = load_prepare_receipt(repository, result.verify_receipt_path)
    expected_managed = tuple(sorted(
        package + name for package in result.installed_packages
        for name in (".skillforge.json", "SKILL.md", "references/evidence.md")
    ))
    assert receipt.ignored_patch_paths == expected_managed
    assert all((repository / path).is_file() for path in expected_managed)

    user_skill.write_text("user version two", encoding="utf-8")
    extra_file = repository / result.installed_packages[0] / "user-note.txt"
    extra_file.write_text("user addition", encoding="utf-8")
    agent_file = repository / "src/app/api/route0.py"
    agent_file.write_text(agent_file.read_text(encoding="utf-8") + "\nVALUE = 1\n",
                          encoding="utf-8")
    assert cli.main([
        "verify", str(repository), "--receipt", result.verify_receipt_path, "--json",
    ]) == 1
    patch = json.loads(capsys.readouterr().out)["report"]["patch"]
    assert {item["path"]: item["kind"] for item in patch["files"]} == {
        user_skill.relative_to(repository).as_posix(): "MODIFIED",
        extra_file.relative_to(repository).as_posix(): "UNTRACKED",
        agent_file.relative_to(repository).as_posix(): "MODIFIED",
    }


def test_prepare_stale_tracked_managed_package_deletion_is_not_agent_patch(repository, capsys):
    first = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SkillForge Test")
    _git(repository, "config", "user.email", "skillforge@example.invalid")
    _git(repository, "add", "src", ".codex")
    _git(repository, "commit", "-qm", "base with managed Skill")

    second = _prepare(repository, SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE)
    assert second.removed_packages
    receipt = load_prepare_receipt(repository, second.verify_receipt_path)
    expected = tuple(sorted(
        package + name for package in first.installed_packages
        for name in (".skillforge.json", "SKILL.md", "references/evidence.md")
    ))
    assert receipt.ignored_patch_paths == expected
    assert cli.main([
        "verify", str(repository), "--receipt", second.verify_receipt_path, "--json",
    ]) == 1
    assert json.loads(capsys.readouterr().out)["report"]["patch"]["files"] == []


def test_semantic_rejection_creates_no_fallback(repository):
    result = _prepare(repository, SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE)
    assert result.semantic_accepted_count == 0
    assert result.semantic_rejection_count == 1
    assert result.compiled_skill_count == 0
    assert result.selected_skill_ids == ()
    assert list((repository / ".codex" / "skills").iterdir()) == []


def test_empty_route_cleans_only_previous_managed_package(repository):
    first = _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    own = repository / ".codex" / "skills" / "my-own-skill" / "SKILL.md"
    own.parent.mkdir(parents=True)
    own.write_text("mine", encoding="utf-8")
    second = _prepare(repository, SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE)
    assert second.selected_skill_ids == ()
    assert len(second.removed_packages) == 1
    assert len(second.to_dict()["sync"]["removed_files"]) == 3
    assert not (repository / first.installed_packages[0]).exists()
    assert own.read_text(encoding="utf-8") == "mine"


def test_unrelated_scope_selects_nothing(repository):
    result = _prepare(
        repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
        current_module="app.service.unit0",
    )
    assert result.compiled_skill_count == 1
    assert result.selected_skill_ids == ()


def test_claude_target_and_user_owned_skill(repository):
    own = repository / ".claude" / "skills" / "my-own-skill" / "SKILL.md"
    own.parent.mkdir(parents=True)
    own.write_text("mine", encoding="utf-8")
    result = _prepare(
        repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
        target=CodeAgentTarget.CLAUDE_CODE,
    )
    assert result.installed_packages[0].startswith(".claude/skills/")
    assert (repository / result.installed_packages[0] / "SKILL.md").is_file()
    assert own.read_text(encoding="utf-8") == "mine"
    assert not (repository / ".codex").exists()


def test_codex_user_owned_skill_is_preserved(repository):
    own = repository / ".codex" / "skills" / "my-own-skill" / "SKILL.md"
    own.parent.mkdir(parents=True)
    own.write_text("mine", encoding="utf-8")
    _prepare(repository, SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    assert own.read_text(encoding="utf-8") == "mine"


def test_unavailable_structural_channel_records_failure_without_knowledge(repository):
    class BrokenModel:
        def complete(self, request):
            raise RuntimeError("model unavailable")

    result = SkillForgePreparer(BrokenModel()).prepare(PrepareRequest(
        repository, "task", CodeAgentTarget.CODEX,
    ))
    assert result.compiled_skill_count == 0
    assert result.installed_packages == ()
    assert result.semantic_failure_count == 1
    assert result.semantic_failures[0]["status"] == "CHANNEL_FAILED"
    assert "model unavailable" in result.semantic_failures[0]["reason"]


def test_missing_path_is_cli_error(repository, capsys):
    status = cli.main([
        "prepare", str(repository / "missing"), "--task", "task", "--target", "codex",
    ])
    assert status == 2
    assert "not a directory" in capsys.readouterr().err
    with pytest.raises(InvalidRepositoryPath):
        SkillForgePreparer().prepare(PrepareRequest(
            repository / "missing", "task", CodeAgentTarget.CODEX,
        ))


def test_cli_json_without_live_model(repository, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "SkillForgePreparer",
        lambda: SkillForgePreparer(FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)),
    )
    status = cli.main([
        "prepare", str(repository), "--task", "unrelated lexical wording", "--target",
        "codex", "--current-module", "app.api.route0", "--json",
    ])
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "codex"
    assert payload["selected_skill_count"] == 1
    assert payload["current_module"] == "app.api.route0"
    assert payload["history_participated"] is False
    assert payload["workflow_semantic_retry_count"] == 0
    assert payload["anti_pattern_semantic_retry_count"] == 0
    assert payload["project_dna"]["rule_count"] == 1
    assert payload["project_dna"]["snapshot_id"]
    assert len(payload["sync"]["written_files"]) == 3


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def historical_repository(repository: Path) -> Path:
    _git(repository, "init")
    _git(repository, "config", "user.name", "SkillForge Test")
    _git(repository, "config", "user.email", "skillforge@example.invalid")
    _git(repository, "add", "src")
    _git(repository, "commit", "-m", "Initial modules")
    paths = (
        repository / "src/app/api/route0.py",
        repository / "src/app/api/route1.py",
        repository / "src/app/api/route2.py",
    )
    for index, message in enumerate(("Add coordinated routes", "Fix coordinated routes")):
        for path in paths:
            path.write_text(path.read_text(encoding="utf-8") + f"VALUE_{index} = {index}\n",
                            encoding="utf-8")
        _git(repository, "add", "src")
        _git(repository, "commit", "-m", message)
    return repository


class WorkflowModel:
    def __init__(self, decision=SemanticDecisionKind.WORKFLOW):
        self.decision = decision
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        evidence = (request.evidence[0].evidence_id,)
        if self.decision != SemanticDecisionKind.WORKFLOW:
            return SemanticDecisionResponse(
                request.repository_id, request.candidate_id, self.decision,
                "The historical facts do not establish a reusable workflow.",
                evidence, 0.5, request.support, None,
            )
        file_kind = request.kind == ChangePatternKind.FILE_CO_CHANGE
        scope = SemanticWorkflowScope(
            SemanticWorkflowScopeKind.FILE if file_kind else SemanticWorkflowScopeKind.MODULE,
            (request.members[0].value if file_kind else request.current_modules[0],),
            "The selected member establishes this scope.",
            "The supplied candidate contains the selected member.", evidence,
        )
        step = SemanticWorkflowStep("step-1", "Coordinate related changes", evidence)
        workflow = SemanticWorkflowResponse(
            request.repository_id, request.candidate_id, "Coordinated routes",
            evidence, "Add coordinated routes", evidence,
            (SemanticWorkflowRole("role-1", "Implementer", "Changes routes", evidence),),
            (step,), (SemanticStepClassification("step-1", "Observed changes", evidence),),
            (), (), scope, (), 0.8,
            SemanticEntityReferences(),
        )
        return SemanticDecisionResponse(
            request.repository_id, request.candidate_id, self.decision,
            "The repeated changes support a reusable workflow.", evidence,
            0.8, request.support, workflow,
        )


class AntiPatternModel:
    def __init__(self, decision=SemanticAntiPatternDecisionKind.ANTI_PATTERN):
        self.decision = decision
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        evidence = tuple(item.evidence_id for item in request.evidence)
        if self.decision == SemanticAntiPatternDecisionKind.ANTI_PATTERN:
            scope = SemanticAntiPatternScope(
                SemanticAntiPatternScopeKind.FILE, request.affected_paths,
                "The affected file is evidenced.", "The signal names the file.", evidence,
            )
            draft = SemanticAntiPatternResponse(
                "Repeated correction", evidence, "A correction followed the earlier edit.",
                evidence, "Avoid repeating the corrected behavior.", evidence,
                "The repository history records a correction.", evidence,
                scope, (), evidence,
            )
        else:
            draft = None
        return SemanticAntiPatternDecisionResponse(
            request.repository_id, request.candidate_id, self.decision,
            "The observed history was evaluated.", evidence, 0.8, draft,
        )


def _rich_prepare(path, workflow_model, anti_pattern_model, *, task="Add coordinated routes",
                  current_module="app.api.route0"):
    return SkillForgePreparer(
        FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE),
        workflow_model, anti_pattern_model,
    ).prepare(PrepareRequest(path, task, CodeAgentTarget.CODEX,
                             current_module=current_module))


def test_git_history_workflow_compiles_routes_and_closes_evidence(
    historical_repository, monkeypatch,
):
    import skillforge.productization.prepare as prepare_module

    actual_builder = prepare_module.ProjectDNABuilder
    captured = []

    class RecordingBuilder(actual_builder):
        def build(self, *args, **kwargs):
            dna = super().build(*args, **kwargs)
            captured.append(dna)
            return dna

    monkeypatch.setattr(prepare_module, "ProjectDNABuilder", RecordingBuilder)
    workflow = WorkflowModel()
    anti_pattern = AntiPatternModel()
    first = _rich_prepare(historical_repository, workflow, anti_pattern)
    dna, = captured
    assert first.history_participated and first.history_commit_count == 3
    assert first.change_pattern_count > 0
    assert first.workflow_candidate_count >= 2
    assert first.workflow_semantic_selected_count == len(workflow.requests) <= 5
    assert first.workflow_semantic_retry_count == 0
    assert first.anti_pattern_semantic_selected_count == len(anti_pattern.requests) <= 5
    assert first.anti_pattern_semantic_retry_count == 0
    assert first.workflow_accepted_count >= 1
    assert first.anti_pattern_accepted_count >= 1
    assert first.dna_workflow_count == len(dna.workflows) == first.workflow_accepted_count
    assert first.dna_anti_pattern_count == len(dna.anti_patterns)
    assert all(item.status == ProjectKnowledgeStatus.CANDIDATE
               for item in (*dna.workflows, *dna.anti_patterns))
    assert dna.repository_id == repository_identity(historical_repository)
    assert {item.evidence_id for item in dna.evidence} == {
        evidence_id for rule in dna.rules for evidence_id in rule.evidence_ids
    } | {
        evidence_id for item in (*dna.workflows, *dna.anti_patterns)
        for evidence_id in item.draft.evidence_ids
    }
    assert first.dna_snapshot_id == dna.snapshot_id
    assert first.compiled_skill_count >= 2
    assert first.selected_skill_ids
    assert first.installed_packages
    assert any(
        "Avoid repeating the corrected behavior." in
        (historical_repository / package / "SKILL.md").read_text(encoding="utf-8")
        for package in first.installed_packages
    )
    second = _rich_prepare(historical_repository, WorkflowModel(), AntiPatternModel())
    assert second.dna_snapshot_id == first.dna_snapshot_id
    assert second.written_files == ()
    assert second.removed_packages == ()


def test_unrelated_rich_route_creates_no_fallback(historical_repository):
    result = _rich_prepare(
        historical_repository, WorkflowModel(), AntiPatternModel(),
        task="unrelated request", current_module="app.service.unit0",
    )
    assert result.compiled_skill_count > 0
    assert result.selected_skill_ids == ()
    assert result.installed_packages == ()


def test_non_git_prepare_has_no_history_or_fallback(repository):
    result = _rich_prepare(repository, WorkflowModel(), AntiPatternModel(), task="unrelated")
    assert not result.history_participated
    assert result.history_commit_count == result.workflow_candidate_count == 0
    assert result.anti_pattern_candidate_count == 0
    assert result.selected_skill_ids == ()
    assert result.verify_available is False
    assert result.verify_receipt_path is None
    assert result.verify_plan_id is None
    assert result.verify_base_commit is None


def test_empty_git_history_skips_history_models(repository):
    _git(repository, "init")

    class UnusedModel:
        def complete(self, request):
            raise AssertionError("empty history must not call a semantic model")

    result = _rich_prepare(repository, UnusedModel(), UnusedModel())
    assert result.history_participated
    assert result.history_commit_count == 0
    assert result.workflow_semantic_selected_count == 0
    assert result.anti_pattern_semantic_selected_count == 0


@pytest.mark.parametrize("rejected", ["workflow", "anti_pattern"])
def test_legal_semantic_rejection_is_not_failure(historical_repository, rejected):
    workflow = WorkflowModel(
        SemanticDecisionKind.INSUFFICIENT_EVIDENCE
        if rejected == "workflow" else SemanticDecisionKind.WORKFLOW
    )
    anti_pattern = AntiPatternModel(
        SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE
        if rejected == "anti_pattern" else SemanticAntiPatternDecisionKind.ANTI_PATTERN
    )
    result = _rich_prepare(historical_repository, workflow, anti_pattern)
    assert result.workflow_rejection_count == (
        len(workflow.requests) if rejected == "workflow" else 0
    )
    assert result.anti_pattern_rejection_count == (
        len(anti_pattern.requests) if rejected == "anti_pattern" else 0
    )
    assert result.dna_workflow_count == (0 if rejected == "workflow" else len(workflow.requests))
    assert result.dna_anti_pattern_count == (
        0 if rejected == "anti_pattern" else len(anti_pattern.requests)
    )
    assert result.compiled_skill_count == result.dna_workflow_count
    assert result.workflow_semantic_retry_count == 0
    assert result.anti_pattern_semantic_retry_count == 0


@pytest.mark.parametrize("broken", ["workflow", "anti_pattern"])
@pytest.mark.parametrize("invalid", [False, True])
def test_semantic_failure_isolated_from_other_channel(historical_repository, broken, invalid):
    class BrokenModel:
        calls = 0

        def complete(self, request):
            self.calls += 1
            if invalid:
                return object()
            raise RuntimeError("model unavailable")

    failed = BrokenModel()
    workflow = failed if broken == "workflow" else WorkflowModel()
    anti_pattern = failed if broken == "anti_pattern" else AntiPatternModel()
    result = _rich_prepare(historical_repository, workflow, anti_pattern)
    assert failed.calls >= 1
    assert result.semantic_failures
    assert all(f["status"] == ("INVALID" if invalid else "CHANNEL_FAILED")
               for f in result.semantic_failures)
    if broken == "workflow":
        assert result.dna_workflow_count == 0 and result.dna_anti_pattern_count > 0
    else:
        assert result.dna_anti_pattern_count == 0 and result.dna_workflow_count > 0


class FlakyWorkflowModel(WorkflowModel):
    def __init__(self, decision, invalid_attempts):
        super().__init__(decision)
        self.invalid_attempts = invalid_attempts

    def complete(self, request):
        response = super().complete(request)
        if self.invalid_attempts:
            self.invalid_attempts -= 1
            return replace(response, reason="An invented/path/name is involved.")
        return response


class FlakyAntiPatternModel(AntiPatternModel):
    def __init__(self, decision, invalid_attempts):
        super().__init__(decision)
        self.invalid_attempts = invalid_attempts

    def complete(self, request):
        response = super().complete(request)
        if self.invalid_attempts:
            self.invalid_attempts -= 1
            return replace(response, reason="An invented/path/name is involved.")
        return response


@pytest.mark.parametrize("decision", [
    SemanticDecisionKind.INSUFFICIENT_EVIDENCE, SemanticDecisionKind.WORKFLOW,
])
def test_workflow_invalid_decision_retries_once_and_keeps_legal_result(
    historical_repository, decision,
):
    workflow = FlakyWorkflowModel(decision, invalid_attempts=1)
    result = _rich_prepare(
        historical_repository, workflow,
        AntiPatternModel(SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE),
    )
    assert result.workflow_semantic_retry_count == 1
    assert len(workflow.requests) == result.workflow_semantic_selected_count + 1
    if decision == SemanticDecisionKind.WORKFLOW:
        assert result.dna_workflow_count == result.workflow_semantic_selected_count
        assert result.compiled_skill_count > 0
    else:
        assert result.workflow_rejection_count == result.workflow_semantic_selected_count
        assert result.dna_workflow_count == 0


def test_workflow_twice_invalid_isolated_after_original_retry(historical_repository):
    workflow = FlakyWorkflowModel(SemanticDecisionKind.WORKFLOW, invalid_attempts=2)
    result = _rich_prepare(historical_repository, workflow, AntiPatternModel())
    assert result.workflow_semantic_retry_count == 1
    assert len(workflow.requests) == result.workflow_semantic_selected_count + 1
    assert result.workflow_accepted_count == result.workflow_semantic_selected_count - 1
    assert result.semantic_failures[0]["status"] == "INVALID"
    assert "unknown path" in result.semantic_failures[0]["reason"]


def test_anti_pattern_invalid_decision_retries_once(historical_repository):
    anti_pattern = FlakyAntiPatternModel(
        SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE, invalid_attempts=1,
    )
    result = _rich_prepare(
        historical_repository,
        WorkflowModel(SemanticDecisionKind.INSUFFICIENT_EVIDENCE), anti_pattern,
    )
    assert result.anti_pattern_semantic_retry_count == 1
    assert len(anti_pattern.requests) == result.anti_pattern_semantic_selected_count + 1
    assert result.anti_pattern_rejection_count == result.anti_pattern_semantic_selected_count
    assert result.dna_anti_pattern_count == 0


def test_anti_pattern_twice_invalid_isolated_after_original_retry(historical_repository):
    anti_pattern = FlakyAntiPatternModel(
        SemanticAntiPatternDecisionKind.ANTI_PATTERN, invalid_attempts=2,
    )
    result = _rich_prepare(
        historical_repository,
        WorkflowModel(SemanticDecisionKind.INSUFFICIENT_EVIDENCE), anti_pattern,
    )
    assert result.anti_pattern_semantic_retry_count == 1
    assert len(anti_pattern.requests) == result.anti_pattern_semantic_selected_count + 1
    assert result.dna_anti_pattern_count == result.anti_pattern_semantic_selected_count - 1
    assert result.semantic_failures[0]["status"] == "INVALID"
    assert "unknown path" in result.semantic_failures[0]["reason"]
