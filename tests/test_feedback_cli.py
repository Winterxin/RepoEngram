"""Repository-local Verify feedback normal use and artifact integrity."""

import json
import subprocess
import sys

import pytest
from test_verify_plan import complex_route

from skillforge import cli
from skillforge.domain.verification import PrepareVerifyReceipt, VerifyPlan, VerifyReport
from skillforge.feedback.artifacts import LEDGER_PATH, load_feedback_ledger
from skillforge.repository.identity import repository_identity
from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import (
    load_verify_artifact,
    save_prepare_receipt,
)
from skillforge.verify.git_patch import GitPatchCollector


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "add", "module.py"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "base"),
                   cwd=root, check=True)
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    seed = VerifyPlanBuilder().build(complex_route(repository_identity(root)))
    receipt = save_prepare_receipt(root, PrepareVerifyReceipt.create(seed, head))
    ref = seed.obligations[0].target_ref
    checks = root / "checks.json"
    checks.write_text(json.dumps({"checks": [{"check_id": "probe", "kind": "TEST",
        "argv": [sys.executable, "-c", "print('ok')"],
        "covers": [{"kind": ref.kind.value, "skill_id": ref.skill_id,
                    "ref_id": ref.ref_id}]}]}), encoding="utf-8")
    assert cli.main(["verify", str(root), "--receipt", receipt,
                     "--checks", "checks.json", "--json"]) == 1
    return root, ref


def report_path(root):
    return next((root / ".skillforge/productization/verify").glob("*.json"))


def feedback(root, report, kind="FALSE_POSITIVE", rationale="reviewed", *selectors,
             supersedes=None):
    argv = ["feedback", str(root), "--report", str(report), "--kind", kind,
            "--rationale", rationale, *selectors, "--json"]
    if supersedes:
        argv.extend(("--supersedes-feedback-id", supersedes))
    return cli.main(argv)


def obligation(ref):
    return ("--target-kind", ref.kind.value, "--skill-id", ref.skill_id,
            "--ref-id", ref.ref_id)


def rewrite(path, mutator):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def test_verify_artifact_recovers_real_plan_report_and_rejects_tampering(prepared):
    root, _ = prepared
    path = report_path(root)
    plan, report = load_verify_artifact(root, path)
    assert isinstance(plan, VerifyPlan) and isinstance(report, VerifyReport)
    original = path.read_bytes()
    for change in (
        lambda value: value["plan"].update(plan_id="0" * 64),
        lambda value: value["report"].update(overall_status="PASSED"),
        lambda value: value["report"].update(repository_id="wrong"),
    ):
        rewrite(path, change)
        with pytest.raises(ValueError):
            load_verify_artifact(root, path)
        path.write_bytes(original)


def test_record_append_duplicate_and_supersession(prepared, capsys):
    root, ref = prepared
    path = report_path(root)
    assert feedback(root, path, "FALSE_POSITIVE", "first", "--check-id", "probe") == 0
    first = json.loads(capsys.readouterr().out)
    ledger, _ = load_feedback_ledger(root)
    assert len(ledger.records) == 1
    assert feedback(root, path, "FALSE_NEGATIVE", "second", *obligation(ref)) == 0
    capsys.readouterr()
    ledger, _ = load_feedback_ledger(root)
    assert len(ledger.records) == 2
    assert feedback(root, path, "FALSE_POSITIVE", "first", "--check-id", "probe") == 2
    assert feedback(root, path, "MODULE_EXCEPTION", "replacement", "--check-id", "probe",
                    supersedes=first["feedback_id"]) == 0
    capsys.readouterr()
    ledger, _ = load_feedback_ledger(root)
    assert len(ledger.records) == 3 and len(ledger.active_records) == 2
    assert feedback(root, path, "FALSE_POSITIVE", "unknown", "--check-id", "probe",
                    supersedes="0" * 64) == 2
    second = next(item for item in ledger.records if item.kind.value == "FALSE_NEGATIVE")
    assert feedback(root, path, "FALSE_POSITIVE", "wrong target", "--check-id", "probe",
                    supersedes=second.feedback_id) == 2


def test_selectors_and_kinds_are_validated(prepared):
    root, ref = prepared
    path = report_path(root)
    assert feedback(root, path, "FALSE_POSITIVE", "bad", "--check-id", "missing") == 2
    assert feedback(root, path, "FALSE_NEGATIVE", "bad", "--target-kind", "RULE") == 2
    assert feedback(root, path, "FALSE_NEGATIVE", "bad", "--target-kind", "RULE",
                    "--skill-id", ref.skill_id, "--ref-id", "unknown") == 2
    assert feedback(root, path, "FALSE_POSITIVE", "good", "--check-id", "probe",
                    *obligation(ref)) == 0
    other = next(item.target_ref for item in load_verify_artifact(root, path)[0].obligations
                 if item.target_ref != ref)
    assert feedback(root, path, "FALSE_POSITIVE", "bad", "--check-id", "probe",
                    *obligation(other)) == 2
    with pytest.raises(SystemExit):
        feedback(root, path, "CONFIRM", "invalid", "--check-id", "probe")


def test_ledger_and_source_tampering_rejected(prepared):
    root, _ = prepared
    path = report_path(root)
    assert feedback(root, path, "FALSE_POSITIVE", "first", "--check-id", "probe") == 0
    ledger_path = root / LEDGER_PATH
    original = ledger_path.read_bytes()
    for change in (
        lambda value: value.update(ledger_id="0" * 64),
        lambda value: value["records"][0].update(feedback_id="0" * 64),
        lambda value: value.update(repository_id="other"),
    ):
        rewrite(ledger_path, change)
        with pytest.raises(ValueError):
            load_feedback_ledger(root)
        ledger_path.write_bytes(original)
    rewrite(path, lambda value: value["report"].update(overall_status="PASSED"))
    with pytest.raises(ValueError):
        load_feedback_ledger(root)


def test_feedback_artifact_is_excluded_from_patch(prepared):
    root, _ = prepared
    path = report_path(root)
    assert feedback(root, path, "FALSE_POSITIVE", "first", "--check-id", "probe") == 0
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    before = GitPatchCollector().snapshot(root, repository_identity(root), head)
    (root / LEDGER_PATH).write_text("changed", encoding="utf-8")
    after = GitPatchCollector().snapshot(root, repository_identity(root), head)
    assert before.snapshot_id == after.snapshot_id
    assert all(not item.path.startswith(".skillforge/productization/feedback/")
               for item in after.files)
