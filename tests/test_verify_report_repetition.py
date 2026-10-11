"""Different observations of one patch remain separate immutable reports."""
from dataclasses import replace

from test_verify_runner import check, repo, route, run  # noqa: F401

from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import load_verify_artifact, save_verify_report


def test_same_patch_new_log_retains_both_reports(repo):  # noqa: F811
    plan = VerifyPlanBuilder().build(route(repo), (check('run', 'print("first")'),))
    first = run(repo, plan)
    first_path = save_verify_report(repo, plan, first)
    original = (repo / first_path).read_bytes()
    second = replace(first, plan=plan,
                     checks=(replace(first.checks[0], stdout='second run\n'),))
    second_path = save_verify_report(repo, plan, second)
    assert first_path != second_path
    assert (repo / first_path).read_bytes() == original
    assert save_verify_report(repo, plan, second) == second_path
    assert load_verify_artifact(repo, first_path)[1] == first
    assert load_verify_artifact(repo, second_path)[1] == second
