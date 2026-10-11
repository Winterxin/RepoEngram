"""Exact early Verify references stay bounded and do not confer checker coverage."""
import json
from dataclasses import replace

from test_verify_plan import complex_route

from skillforge.domain.verification import VerifyObligationKind, VerifyTargetRef
from skillforge.integration.mcp_server import _completion_preview
from skillforge.verify import VerifyPlanBuilder


def previous_bytes(*, escaped=False):
    previous = {'tool': 'verify_patch', 'arguments': {'working_tree': True},
                'instruction': 'After the final code edit, call this tool once. '
                               'Inspect status; missing coverage stays INCOMPLETE. '
                               'Do not substitute only running tests '
                               'or reading an old report.'}
    payload = json.dumps(previous, ensure_ascii=False)
    if escaped:
        payload = json.dumps(payload, ensure_ascii=False)
    return len(payload.encode('utf-8'))


def assert_budget(preview):
    payload = json.dumps(preview, ensure_ascii=False)
    assert len(payload.encode('utf-8')) <= previous_bytes()
    assert len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) <= previous_bytes(
        escaped=True)


def test_exact_preview_references_are_from_the_canonical_plan_without_coverage():
    plan = VerifyPlanBuilder().build(complex_route())
    preview = _completion_preview(plan)
    exposed = {(kind, ref) for kind, refs in preview['targets'].items() for ref in refs}
    expected = {(item.target_ref.kind.value, item.target_ref.ref_id) for item in plan.obligations}
    assert exposed and exposed <= expected
    assert len(exposed) + preview['omitted_targets'] == len(expected)
    assert 'optional' not in {ref for _, ref in exposed}
    assert plan.checks == ()
    assert_budget(preview)
    assert preview['tool'] == 'verify_patch'
    assert preview['arguments'] == {'working_tree': True}


def test_multiple_workflow_targets_share_kind_and_keep_all_references_in_plan():
    full = VerifyPlanBuilder().build(complex_route())
    obligations = tuple(item for item in full.obligations
                        if item.target_ref.kind == VerifyObligationKind.WORKFLOW_STEP)
    plan = VerifyPlanBuilder()._create_plan(
        full.repository_id, full.registry_id, full.source_snapshot_id, full.selected_skills,
        obligations, (), tuple(item for item in full.evidence if item.evidence_id == 'step-ev'))
    preview = _completion_preview(plan)
    refs = preview['targets'].get('WORKFLOW_STEP', [])
    assert refs
    assert len(refs) + preview['omitted_targets'] == 2
    assert plan.obligations == obligations
    assert plan.checks == ()


def test_long_unicode_target_is_omitted_whole_and_not_waived():
    full = VerifyPlanBuilder().build(complex_route())
    original = full.obligations[0]
    long = replace(original, target_ref=VerifyTargetRef(
        original.target_ref.kind, original.target_ref.skill_id, '目标' * 200))
    plan = VerifyPlanBuilder()._create_plan(
        full.repository_id, full.registry_id, full.source_snapshot_id, full.selected_skills,
        (long,), (), tuple(item for item in full.evidence if item.evidence_id in long.evidence_ids))
    preview = _completion_preview(plan)
    assert preview['targets'] == {}
    assert preview['omitted_targets'] == 1
    assert plan.obligations == (long,)
    assert_budget(preview)


def test_empty_plan_preview_does_not_fabricate_targets():
    full = VerifyPlanBuilder().build(complex_route())
    plan = VerifyPlanBuilder()._create_plan(
        full.repository_id, full.registry_id, full.source_snapshot_id,
        full.selected_skills, (), (), ())
    preview = _completion_preview(plan)
    assert preview['targets'] == {} and preview['omitted_targets'] == 0
    assert plan.checks == ()


def test_normal_workflow_reference_fits_without_allocating_checker_coverage():
    full = VerifyPlanBuilder().build(complex_route())
    original = next(o for o in full.obligations
                    if o.target_ref.kind == VerifyObligationKind.WORKFLOW_STEP)
    obligation = replace(original, target_ref=VerifyTargetRef(
        original.target_ref.kind, original.target_ref.skill_id, 'validate-exception-state'))
    plan = VerifyPlanBuilder()._create_plan(
        full.repository_id, full.registry_id, full.source_snapshot_id,
        full.selected_skills, (obligation,), (),
        tuple(e for e in full.evidence if e.evidence_id in obligation.evidence_ids))
    preview = _completion_preview(plan)
    assert preview['targets']['WORKFLOW_STEP'] == ['validate-exception-state']
    assert preview['omitted_targets'] == 0
    assert plan.checks == () and plan.obligations == (obligation,)
    assert 'proven' in preview['instruction'] and 'omissions' in preview['instruction']
    assert_budget(preview)
