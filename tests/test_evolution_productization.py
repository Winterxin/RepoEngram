"""Deterministic prepare baseline and evolution planning contracts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_verify_plan import complex_route

from skillforge import cli
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.verification import PrepareVerifyReceipt, fingerprint
from skillforge.feedback.artifacts import LEDGER_PATH, source_path
from skillforge.productization.evolution_artifacts import (
    _verify_source_path,
    load_baseline,
    load_plan,
)
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.identity import repository_identity
from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import load_verify_artifact, save_prepare_receipt
from skillforge.verify.git_patch import GitPatchCollector


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "SkillForge Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "add", "sample.py")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _prepare(root: Path):
    return SkillForgePreparer(object(), object(), object()).prepare(
        PrepareRequest(root, "unrelated work", CodeAgentTarget.CODEX),
    )


def test_non_git_prepare_has_no_evolution_baseline(tmp_path):
    (tmp_path / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = _prepare(tmp_path)
    assert not result.evolution_available
    assert result.evolution_baseline_id is None
    assert result.evolution_baseline_path is None
    assert result.evolution_head_commit is None


def test_unborn_git_prepare_has_no_evolution_baseline(tmp_path):
    (tmp_path / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    result = _prepare(tmp_path)
    assert not result.evolution_available
    assert result.evolution_baseline_path is None


def test_prepare_freezes_exact_source_and_repeats(repo):
    first = _prepare(repo)
    baseline = load_baseline(repo, first.evolution_baseline_path)
    content = (repo / first.evolution_baseline_path).read_bytes()
    assert first.evolution_available
    assert baseline.repository_id == repository_identity(repo)
    assert baseline.source_dna.snapshot_id == first.dna_snapshot_id
    assert baseline.source_compilation.source_snapshot_id == first.dna_snapshot_id
    assert baseline.source_state.head_commit == _git(repo, "rev-parse", "HEAD")
    assert baseline.source_state.history.changesets[0].commit_id == baseline.head_commit
    assert first.selected_skill_ids == ()
    second = _prepare(repo)
    assert second.evolution_baseline_id == first.evolution_baseline_id
    assert (repo / second.evolution_baseline_path).read_bytes() == content


@pytest.mark.parametrize(
    "field",
    [
        "source_dna",
        "source_compilation",
        "source_state",
        "head_commit",
        "baseline_id",
    ],
)
def test_baseline_rejects_tampering(repo, field):
    result = _prepare(repo)
    path = repo / result.evolution_baseline_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if field == "source_dna":
        value[field]["snapshot_id"] = "tampered"
    elif field == "source_compilation":
        value[field]["source_snapshot_id"] = "tampered"
    elif field == "source_state":
        value[field]["state_id"] = "tampered"
    else:
        value[field] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(repo, result.evolution_baseline_path)


def test_evolve_plan_empty_and_new_commits(repo, capsys):
    prepared = _prepare(repo)
    args = ["evolve-plan", str(repo), "--baseline", prepared.evolution_baseline_path, "--json"]
    assert cli.main(args) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["new_commits"] == 0
    baseline = load_baseline(repo, prepared.evolution_baseline_path)
    plan = load_plan(repo, empty["plan_artifact"], baseline)
    assert plan.target_head_commit == baseline.head_commit
    assert empty["workflow_candidates"] == {"changed": 0, "new": 0, "missing": 0}

    (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-qm", "change 2")
    assert cli.main(args) == 0
    one_commit = json.loads(capsys.readouterr().out)
    assert one_commit["new_commits"] == 1
    assert one_commit["target_head_commit"] == _git(repo, "rev-parse", "HEAD")
    (repo / "sample.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-qm", "change 3")
    assert cli.main(args) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["new_commits"] == 2
    assert updated["target_head_commit"] == _git(repo, "rev-parse", "HEAD")
    assert load_plan(repo, updated["plan_artifact"], baseline).plan_id == updated["plan_id"]
    plan_path = repo / updated["plan_artifact"]
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    data["plan_id"] = "tampered"
    plan_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(repo, updated["plan_artifact"], baseline)


def test_diverged_history_and_patch_exclusion(repo, capsys):
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-qm", "first")
    prepared = _prepare(repo)
    _git(repo, "reset", "--hard", parent)
    (repo / "sample.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-qm", "alternate")
    assert cli.main(["evolve-plan", str(repo), "--baseline", prepared.evolution_baseline_path]) == 2
    assert "not an ancestor" in capsys.readouterr().err
    patch = GitPatchCollector().snapshot(repo, repository_identity(repo), parent)
    assert all(
        not item.path.startswith(".skillforge/productization/evolution/") for item in patch.files
    )


def test_evolution_artifacts_do_not_change_verify_patch_hash(repo):
    base = _git(repo, "rev-parse", "HEAD")
    identity = repository_identity(repo)
    before = GitPatchCollector().snapshot(repo, identity, base)
    prepared = _prepare(repo)
    artifact = repo / prepared.evolution_baseline_path
    untracked = GitPatchCollector().snapshot(repo, identity, base)
    assert untracked.snapshot_id == before.snapshot_id
    _git(repo, "add", prepared.evolution_baseline_path)
    _git(repo, "commit", "-qm", "track evolution artifact")
    tracked = GitPatchCollector().snapshot(repo, identity, base)
    assert tracked.files == before.files
    assert tracked.tracked_diff_sha256 == before.tracked_diff_sha256
    artifact.write_text("changed", encoding="utf-8")
    modified = GitPatchCollector().snapshot(repo, identity, base)
    assert modified.files == before.files
    assert modified.tracked_diff_sha256 == before.tracked_diff_sha256
    artifact.unlink()
    deleted = GitPatchCollector().snapshot(repo, identity, base)
    assert deleted.files == before.files
    assert deleted.tracked_diff_sha256 == before.tracked_diff_sha256


def test_verify_feedback_is_planned_for_same_snapshot(repo, capsys):
    prepared = _prepare(repo)
    checks = repo / "checks.json"
    checks.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "check_id": "probe",
                        "kind": "TEST",
                        "argv": [sys.executable, "-c", "print('ok')"],
                        "cwd": ".",
                        "timeout_seconds": 10,
                        "covers": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "verify",
                str(repo),
                "--receipt",
                prepared.verify_receipt_path,
                "--checks",
                "checks.json",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    report = next((repo / ".skillforge/productization/verify").glob("*.json"))
    report = report.relative_to(repo).as_posix()
    _, report_value = load_verify_artifact(repo, report)
    _verify_source_path(report, report_value)  # current content-addressed v2 path
    legacy_key = fingerprint(
        ("verify-report-artifact-v1", report_value.plan_id, report_value.patch.snapshot_id)
    )
    legacy_path = f".skillforge/productization/verify/{legacy_key}.json"
    legacy_file = repo / legacy_path
    legacy_file.write_bytes((repo / report).read_bytes())
    _verify_source_path(legacy_path, report_value)  # historical v1 key remains readable
    tampered_report = report_value.to_dict()
    tampered_report["overall_status"] = "tampered"
    wrong_v2_key = fingerprint(
        (
            "verify-report-artifact-v2",
            report_value.plan_id,
            report_value.patch.snapshot_id,
            tampered_report,
        )
    )
    with pytest.raises(ValueError):
        _verify_source_path(f".skillforge/productization/verify/{wrong_v2_key}.json", report_value)
    assert source_path(repo, report) == report
    assert (
        cli.main(
            [
                "feedback",
                str(repo),
                "--report",
                report,
                "--kind",
                "FALSE_POSITIVE",
                "--rationale",
                "reviewed",
                "--check-id",
                "probe",
                "--json",
            ]
        )
        == 0
    )
    feedback = json.loads(capsys.readouterr().out)
    assert (
        cli.main(
            ["evolve-plan", str(repo), "--baseline", prepared.evolution_baseline_path, "--json"]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["feedback_total_active_count"] == 1
    assert summary["feedback_applicable_count"] == 1
    assert summary["feedback_ignored_stale_count"] == 0
    baseline = load_baseline(repo, prepared.evolution_baseline_path)
    plan_path = repo / summary["plan_artifact"]
    original_plan = plan_path.read_bytes()
    tampered = json.loads(original_plan)
    source = next(iter(tampered["feedback_sources"]))
    tampered["feedback_sources"][source]["report"]["plan_id"] = "tampered"
    plan_path.write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_plan(repo, summary["plan_artifact"], baseline)
    plan_path.write_bytes(original_plan)
    (repo / report).unlink()
    plan = load_plan(repo, summary["plan_artifact"], baseline)
    assert plan.verify_review_feedback_ids == (feedback["feedback_id"],)


def test_stale_feedback_is_ignored_without_changing_ledger(repo, capsys):
    prepared = _prepare(repo)
    other = VerifyPlanBuilder().build(complex_route(repository_identity(repo)))
    assert other.source_snapshot_id != prepared.dna_snapshot_id
    receipt = save_prepare_receipt(
        repo,
        PrepareVerifyReceipt.create(other, _git(repo, "rev-parse", "HEAD")),
    )
    checks = repo / "checks.json"
    checks.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "check_id": "probe",
                        "kind": "TEST",
                        "argv": [sys.executable, "-c", "print('ok')"],
                        "cwd": ".",
                        "timeout_seconds": 10,
                        "covers": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli.main(["verify", str(repo), "--receipt", receipt, "--checks", "checks.json", "--json"])
        == 1
    )
    report = json.loads(capsys.readouterr().out)["summary"]["report_path"]
    report = Path(report).as_posix()
    assert (
        cli.main(
            [
                "feedback",
                str(repo),
                "--report",
                report,
                "--kind",
                "FALSE_POSITIVE",
                "--rationale",
                "stale",
                "--check-id",
                "probe",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    ledger_before = (repo / LEDGER_PATH).read_bytes()
    assert (
        cli.main(
            ["evolve-plan", str(repo), "--baseline", prepared.evolution_baseline_path, "--json"]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["feedback_total_active_count"] == 1
    assert summary["feedback_applicable_count"] == 0
    assert summary["feedback_ignored_stale_count"] == 1
    assert (repo / LEDGER_PATH).read_bytes() == ledger_before
    baseline = load_baseline(repo, prepared.evolution_baseline_path)
    plan = load_plan(repo, summary["plan_artifact"], baseline)
    assert plan.verify_review_feedback_ids == ()
