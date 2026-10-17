"""Normal Verify CLI use consumes only the frozen prepare receipt."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_verify_plan import complex_route

from skillforge import cli
from skillforge.domain.verification import PrepareVerifyReceipt
from skillforge.repository.identity import repository_identity
from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import load_prepare_receipt, save_prepare_receipt


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "add", "module.py"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "base"), cwd=root, check=True)
    return root


def _receipt(root: Path) -> str:
    seed = VerifyPlanBuilder().build(complex_route(repository_identity(root)))
    return save_prepare_receipt(root, PrepareVerifyReceipt.create(seed, _head(root)))


def _head(root: Path) -> str:
    return subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _checks(root: Path, argv: list[str], covers, *, check_id: str = "explicit") -> Path:
    path = root / "checks.json"
    value = {
        "checks": [{
            "check_id": check_id,
            "kind": "TEST",
            "argv": argv,
            "covers": [
                {"kind": ref.kind.value, "skill_id": ref.skill_id, "ref_id": ref.ref_id}
                for ref in covers
            ],
        }],
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_verify_cli_uses_receipt_base_and_only_explicit_covered_check(
    git_repo, capsys, monkeypatch,
):
    root = git_repo
    receipt_path = _receipt(root)
    receipt = load_prepare_receipt(root, receipt_path)
    (root / "module.py").write_text("value = 2\n", encoding="utf-8")
    check_path = _checks(
        root, [sys.executable, "-c", "print('ok')"],
        tuple(item.target_ref for item in receipt.plan.obligations),
    )

    class UnexpectedPrepare:
        def __init__(self, *args, **kwargs):
            raise AssertionError("verify must not enter Rich Prepare")

    monkeypatch.setattr(cli, "SkillForgePreparer", UnexpectedPrepare)
    monkeypatch.setattr(cli, "RepositoryAnalyzer", UnexpectedPrepare)
    status = cli.main([
        "verify", str(root), "--receipt", receipt_path, "--checks", str(check_path), "--json",
    ])
    output = json.loads(capsys.readouterr().out)
    assert status == 0
    assert output["summary"]["base_commit"] == receipt.base_commit == _head(root)
    assert output["report"]["patch"]["resolved_base_commit"] == receipt.base_commit
    assert output["summary"]["changed_file_count"] == 1
    assert [item["path"] for item in output["report"]["patch"]["files"]] == ["module.py"]
    assert output["report"]["checks"][0]["status"] == "PASSED"
    assert output["summary"]["overall_status"] == "PASSED"
    assert (root / output["summary"]["report_path"]).is_file()
    first_patch_id = output["report"]["patch"]["snapshot_id"]
    assert cli.main([
        "verify", str(root), "--receipt", receipt_path, "--checks", str(check_path), "--json",
    ]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["report"]["patch"]["snapshot_id"] == first_patch_id
    assert repeated["summary"]["report_path"] == output["summary"]["report_path"]


@pytest.mark.parametrize(
    ("argv", "expected_status"),
    [([sys.executable, "-c", "raise SystemExit(3)"], "FAILED"), (None, "INCOMPLETE")],
)
def test_failed_and_incomplete_reports_are_saved(git_repo, capsys, argv, expected_status):
    root = git_repo
    receipt_path = _receipt(root)
    receipt = load_prepare_receipt(root, receipt_path)
    if argv is None:
        command = ["verify", str(root), "--receipt", receipt_path]
    else:
        checks = _checks(root, argv, tuple(
            item.target_ref for item in receipt.plan.obligations
        ))
        command = ["verify", str(root), "--receipt", receipt_path, "--checks", str(checks)]
    assert cli.main(command + ["--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["overall_status"] == expected_status
    assert (root / output["summary"]["report_path"]).is_file()
    if expected_status == "INCOMPLETE":
        assert output["summary"]["unverified_obligation_count"] == len(receipt.plan.obligations)


def test_unknown_coverage_is_rejected_before_check_execution(git_repo, capsys):
    root = git_repo
    receipt_path = _receipt(root)
    marker = root / "should-not-run"
    code = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    checks = root / "bad-checks.json"
    checks.write_text(json.dumps({"checks": [{
        "check_id": "unsafe-coverage",
        "kind": "CUSTOM",
        "argv": [sys.executable, "-c", code],
        "covers": [{"kind": "RULE", "skill_id": "unknown", "ref_id": "missing"}],
    }]}), encoding="utf-8")
    assert cli.main([
        "verify", str(root), "--receipt", receipt_path, "--checks", str(checks),
    ]) == 2
    assert not marker.exists()
    assert "unknown obligation" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["plan_id", "evidence", "selected_skill"])
def test_tampered_receipt_is_rejected(git_repo, capsys, field):
    root = git_repo
    receipt_path = _receipt(root)
    payload = json.loads((root / receipt_path).read_text(encoding="utf-8"))
    plan = payload["plan"]
    if field == "plan_id":
        plan["plan_id"] = "0" * 64
    elif field == "evidence":
        plan["evidence"][0]["description"] = "tampered evidence"
    else:
        plan["selected_skills"][0][0] = "tampered-skill"
    (root / receipt_path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["verify", str(root), "--receipt", receipt_path]) == 2
    assert "Verify input or repository error" in capsys.readouterr().err


def test_receipt_cannot_be_reused_at_another_repository(git_repo, tmp_path):
    source = git_repo
    receipt_path = _receipt(source)
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=other, check=True)
    foreign = other / receipt_path
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes((source / receipt_path).read_bytes())
    with pytest.raises(ValueError, match="repository identity mismatch"):
        load_prepare_receipt(other, receipt_path)


def test_receipt_fingerprint_binds_exact_managed_paths(git_repo):
    root = git_repo
    seed = VerifyPlanBuilder().build(complex_route(repository_identity(root)))
    managed = ".codex/skills/skillforge-" + "a" * 32 + "/SKILL.md"
    without_paths = PrepareVerifyReceipt.create(seed, _head(root))
    with_paths = PrepareVerifyReceipt.create(seed, _head(root), (managed,))
    assert with_paths.receipt_id != without_paths.receipt_id
    relative = save_prepare_receipt(root, with_paths)
    assert load_prepare_receipt(root, relative).ignored_patch_paths == (managed,)

    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    payload["ignored_patch_paths"] = [managed.replace("/SKILL.md", "/.skillforge.json")]
    (root / relative).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt_id mismatch"):
        load_prepare_receipt(root, relative)


@pytest.mark.parametrize("paths", [
    ("../escape",), ("/absolute",), ("C:/absolute",),
    (".codex/skills/my-own-skill/SKILL.md",),
    (".codex/skills/skillforge-" + "a" * 32 + "/SKILL.md",) * 2,
])
def test_receipt_rejects_invalid_managed_exclusions(git_repo, paths):
    seed = VerifyPlanBuilder().build(complex_route(repository_identity(git_repo)))
    with pytest.raises(ValueError):
        PrepareVerifyReceipt.create(seed, _head(git_repo), paths)


def test_receipt_rejects_managed_path_symlink_escape(git_repo, tmp_path):
    root = git_repo
    managed_dir = root / ".codex" / "skills" / ("skillforge-" + "a" * 32)
    managed_dir.parent.mkdir(parents=True)
    try:
        managed_dir.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    seed = VerifyPlanBuilder().build(complex_route(repository_identity(root)))
    managed = f".codex/skills/{managed_dir.name}/SKILL.md"
    relative = save_prepare_receipt(
        root, PrepareVerifyReceipt.create(seed, _head(root), (managed,)),
    )
    with pytest.raises(ValueError, match="link or reparse point"):
        load_prepare_receipt(root, relative)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"argv": ["bash", "-c", "echo unsafe"]},
        {"cwd": "../outside"},
    ],
)
def test_checks_json_still_obeys_command_contract(git_repo, capsys, unsafe):
    root = git_repo
    receipt_path = _receipt(root)
    check = {"check_id": "bad", "kind": "CUSTOM", "argv": [sys.executable, "-V"]}
    check.update(unsafe)
    checks = root / "unsafe.json"
    checks.write_text(json.dumps({"checks": [check]}), encoding="utf-8")
    assert cli.main([
        "verify", str(root), "--receipt", receipt_path, "--checks", str(checks),
    ]) == 2
    assert "Verify input or repository error" in capsys.readouterr().err
