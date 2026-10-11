"""Independent command results, coverage, and report status in a real Git repo."""

import subprocess
import sys
from dataclasses import replace

import pytest
from test_verify_plan import complex_route

from skillforge.domain.routing import SkillRegistry, SkillRoutingRequest, registry_fingerprint
from skillforge.domain.verification import (
    VerifyCheckKind,
    VerifyCheckStatus,
    VerifyCommandSpec,
    VerifyFinding,
    VerifyFindingSeverity,
    VerifyOverallStatus,
)
from skillforge.repository.identity import repository_identity
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify import IndependentVerifier, VerifyPlanBuilder


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-qm", "base"), cwd=root, check=True)
    return root


def route(root, selected=True):
    if selected:
        return complex_route(repository_identity(root))
    repo_id = repository_identity(root)
    registry = SkillRegistry(repo_id, "snap", registry_fingerprint(repo_id, "snap", ()), (), ())
    return SkillRouter().route(registry, SkillRoutingRequest(repo_id, "create route"))


def check(key, code, *, covers=(), timeout=300):
    return VerifyCommandSpec(key, VerifyCheckKind.CUSTOM,
                             (sys.executable, "-c", code),
                             timeout_seconds=timeout, covers=covers)


def run(root, plan):
    return IndependentVerifier().verify(root, "HEAD", plan)


def test_pass_all_covered_and_no_checks_incomplete(repo):
    selected = route(repo)
    empty_plan = VerifyPlanBuilder().build(selected)
    assert run(repo, empty_plan).overall_status == VerifyOverallStatus.INCOMPLETE
    refs = tuple(item.target_ref for item in empty_plan.obligations)
    complete = VerifyPlanBuilder().build(selected, (check("pass", "pass", covers=refs),))
    result = run(repo, complete)
    assert result.overall_status == VerifyOverallStatus.PASSED
    assert result.checks[0].status == VerifyCheckStatus.PASSED
    assert result.findings == ()
    assert tuple(item.evidence_id for item in result.evidence) == (
        "risk-ev", "rule-ev", "step-ev"
    )
    assert result.to_dict() == run(repo, complete).to_dict()
    with pytest.raises(ValueError, match="overall status"):
        replace(result, plan=complete, overall_status=VerifyOverallStatus.INCOMPLETE)
    with pytest.raises(ValueError, match="Evidence closure"):
        replace(result, plan=complete, evidence=result.evidence[:-1])


def test_uncovered_candidate_risk_and_completion_are_incomplete(repo):
    selected = route(repo)
    plan = VerifyPlanBuilder().build(selected, (check("pass", "pass"),))
    result = run(repo, plan)
    assert result.overall_status == VerifyOverallStatus.INCOMPLETE
    assert len(result.findings) == len(plan.obligations)
    assert all(item.severity == VerifyFindingSeverity.CANDIDATE for item in result.findings)
    assert all(item.code == "UNVERIFIED_OBLIGATION" for item in result.findings)


@pytest.mark.parametrize("code,status", (
    ("import sys; sys.exit(3)", VerifyCheckStatus.FAILED),
    ("import time; time.sleep(2)", VerifyCheckStatus.TIMED_OUT),
))
def test_fail_and_timeout_produce_error_with_obligation_evidence(repo, code, status):
    selected = route(repo)
    ref = VerifyPlanBuilder().build(selected).obligations[0].target_ref
    plan = VerifyPlanBuilder().build(selected, (check("bad", code, covers=(ref,), timeout=1),))
    report = run(repo, plan)
    assert report.overall_status == VerifyOverallStatus.FAILED
    assert report.checks[0].status == status
    assert any(item.target_ref == ref and item.check_id == "bad" and
               item.severity == VerifyFindingSeverity.ERROR for item in report.findings)


def test_missing_executable_is_error_and_empty_route_statuses(repo):
    empty = route(repo, selected=False)
    empty_report = run(repo, VerifyPlanBuilder().build(empty))
    assert empty_report.overall_status == VerifyOverallStatus.INCOMPLETE
    passed = VerifyPlanBuilder().build(empty, (check("ok", "pass"),))
    assert run(repo, passed).overall_status == VerifyOverallStatus.PASSED
    missing = VerifyCommandSpec("missing", VerifyCheckKind.CUSTOM,
                                ("skillforge-executable-does-not-exist-936592",))
    report = run(repo, VerifyPlanBuilder().build(empty, (missing,)))
    assert report.checks[0].status == VerifyCheckStatus.ERROR
    assert report.overall_status == VerifyOverallStatus.FAILED


def test_output_bounds_shell_literals_and_order(repo):
    code = "import sys; print('x'*70000); print('y'*70000, file=sys.stderr)"
    literal = "a; echo SHOULD_NOT_EXECUTE"
    literal_code = "import sys; assert sys.argv[1] == 'a; echo SHOULD_NOT_EXECUTE'"
    checks = (
        VerifyCommandSpec("z", VerifyCheckKind.CUSTOM,
                          (sys.executable, "-c", literal_code, literal)),
        check("a", code),
    )
    report = run(repo, VerifyPlanBuilder().build(route(repo, False), checks))
    assert report.overall_status == VerifyOverallStatus.PASSED
    assert tuple(item.check_id for item in report.checks) == ("a", "z")
    assert "[output truncated" in report.checks[0].stdout
    assert "[output truncated" in report.checks[0].stderr
    assert len(report.checks[0].stdout) < 66000


def test_cwd_link_escape_is_rejected(repo, tmp_path):
    external = tmp_path / "outside"
    external.mkdir()
    link = repo / "linked"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    spec = replace(check("x", "pass"), cwd="linked")
    plan = VerifyPlanBuilder().build(route(repo, False), (spec,))
    with pytest.raises(ValueError, match="link or reparse"):
        run(repo, plan)


def test_report_direct_construction_is_bound_to_exact_plan_and_checks(repo):
    selected = route(repo)
    refs = tuple(item.target_ref for item in VerifyPlanBuilder().build(selected).obligations)
    plan = VerifyPlanBuilder().build(selected, (
        check("a", "pass", covers=refs[:2]),
        check("b", "pass", covers=refs[2:]),
    ))
    report = run(repo, plan)
    assert report.overall_status == VerifyOverallStatus.PASSED
    assert "plan" not in report.to_dict()
    assert replace(report, plan=plan) == report
    with pytest.raises(ValueError, match="original VerifyPlan"):
        replace(report, plan=object())
    for change in (
        {"registry_id": "other"},
        {"repository_id": "other"},
        {"source_snapshot_id": "other"},
        {"plan_id": "other"},
        {"obligations": plan.obligations[:-1]},
        {"obligations": tuple(reversed(plan.obligations))},
        {"obligations": (replace(plan.obligations[0], description="altered"),
                         *plan.obligations[1:])},
        {"obligations": (replace(plan.obligations[0], source_status="altered"),
                         *plan.obligations[1:])},
        {"obligations": (replace(plan.obligations[0], evidence_ids=()),
                         *plan.obligations[1:])},
        {"checks": report.checks[:-1]},
        {"checks": (*report.checks,
                    replace(report.checks[-1], check_id="z"))},
        {"checks": (report.checks[0], report.checks[0])},
        {"checks": (replace(report.checks[0], check_id="fake"), report.checks[1])},
        {"checks": (replace(report.checks[0], kind=VerifyCheckKind.TEST),
                    report.checks[1])},
        {"checks": (replace(report.checks[0], argv=(*report.checks[0].argv, "extra")),
                    report.checks[1])},
        {"checks": (replace(report.checks[0], cwd="subdir"), report.checks[1])},
        {"checks": (replace(report.checks[0], covered_refs=()), report.checks[1])},
        {"findings": (VerifyFinding("FAKE_VIOLATION", VerifyFindingSeverity.ERROR,
                                    "Invented failure"),)},
        {"overall_status": VerifyOverallStatus.INCOMPLETE},
        {"evidence": (replace(report.evidence[0], description="altered"),
                      *report.evidence[1:])},
    ):
        with pytest.raises(ValueError):
            replace(report, plan=plan, **change)


def test_report_requires_every_failed_check_and_uncovered_candidate(repo):
    selected = route(repo)
    failed_plan = VerifyPlanBuilder().build(selected, (
        check("a", "raise SystemExit(1)"),
        check("b", "raise SystemExit(2)"),
    ))
    failed = run(repo, failed_plan)
    assert tuple(item.check_id for item in failed.findings if item.code == "CHECK_FAILED") == (
        "a", "b"
    )
    for missing in (0, 1):
        altered = failed.findings[:missing] + failed.findings[missing + 1:]
        with pytest.raises(ValueError, match="findings"):
            replace(failed, plan=failed_plan, findings=altered)
    with pytest.raises(ValueError, match="findings"):
        replace(failed, plan=failed_plan, findings=tuple(
            item for item in failed.findings if item.code != "UNVERIFIED_OBLIGATION"
        ))
    with pytest.raises(ValueError, match="findings"):
        replace(failed, plan=failed_plan, findings=(
            *failed.findings[:-1],
            replace(failed.findings[-1], severity=VerifyFindingSeverity.ERROR),
        ))
    ref = VerifyPlanBuilder().build(selected).obligations[0].target_ref
    covered_plan = VerifyPlanBuilder().build(selected, (
        check("covered", "raise SystemExit(1)", covers=(ref,)),
    ))
    covered = run(repo, covered_plan)
    assert any(item.code == "COVERED_CHECK_FAILED" for item in covered.findings)
    with pytest.raises(ValueError, match="findings"):
        replace(covered, plan=covered_plan, findings=tuple(
            item for item in covered.findings if item.code != "COVERED_CHECK_FAILED"
        ))


def test_check_result_direct_construction_rejects_malformed_payload(repo):
    plan = VerifyPlanBuilder().build(route(repo, False), (check("a", "pass"),))
    result = run(repo, plan).checks[0]
    for change in (
        {"argv": "python -c pass"}, {"argv": ()}, {"argv": ("",)},
        {"argv": (sys.executable, 42)}, {"cwd": ""}, {"cwd": 42},
        {"stdout": None}, {"stderr": None},
        {"covered_refs": []}, {"covered_refs": ("bad",)},
        {"exit_code": 3},
    ):
        with pytest.raises(ValueError):
            replace(result, **change)
