"""Explicit declarations cannot invent obligations or turn failed checks into passes."""
import json
import sys
from dataclasses import replace

import pytest
from test_mcp_supplemental_knowledge import rich_repo  # noqa: F401
from test_mcp_supplemental_knowledge import service as project_service  # noqa: F401

from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import load_checks


@pytest.fixture
def binding(project_service):  # noqa: F811
    service = project_service
    root = service.root
    route = service.call('get_related_workflow', {
        'task': 'unrelated', 'scope': {'current_module': 'app.api0.route0'}})
    for skill in route['skills']:
        service.call('get_skill', {'skill_id': skill['skill_id']})
    path = root / '.skillforge/autonomous/checks.json'
    path.write_text(json.dumps({'checks': [{'check_id': 'public', 'kind': 'TEST',
        'argv': [sys.executable, '-c', 'raise SystemExit(1)'], 'timeout_seconds': 10}]}), 'utf-8')
    ref = VerifyPlanBuilder().build(service.route).obligations[0].target_ref
    args = {'check_id': 'public', 'skill_id': ref.skill_id,
            'targets': {ref.kind.value: [ref.ref_id]}}
    return service, path, args, ref


def test_declared_binding_preserves_command_and_still_fails_execution(binding):
    service, path, args, ref = binding
    original, _ = load_checks(service.root, path)
    result = service.call('bind_verify_check', args)
    assert result['status'] == 'DECLARED_NOT_VERIFIED'
    checks, _ = load_checks(service.root, path)
    assert checks[0].argv == original[0].argv
    assert checks[0].timeout_seconds == original[0].timeout_seconds
    assert checks[0].covers == (ref,)
    before = path.read_bytes()
    service.call('bind_verify_check', args)
    assert path.read_bytes() == before
    report = service.call('verify_patch', {})['report']
    assert report['overall_status'] != 'PASSED'
    assert report['checks'][0]['status'] == 'FAILED'


@pytest.mark.parametrize('change', [
    {'check_id': 'unknown'}, {'skill_id': 'foreign'},
    {'targets': {'WORKFLOW_STEP': ['invented']}},
    {'targets': {'UNKNOWN': ['invented']}}, {'targets': {}},
    {'targets': {'WORKFLOW_STEP': ['x'] * 17}},
    {'targets': {'WORKFLOW_STEP': ['x', 'x']}},
    {'targets': {'WORKFLOW_STEP': 'not-an-array'}},
])
def test_invalid_declaration_does_not_write(binding, change):
    service, path, args, _ = binding
    before = path.read_bytes()
    with pytest.raises(ValueError):
        service.call('bind_verify_check', {**args, **change})
    assert path.read_bytes() == before


def test_unread_skill_cannot_bind(binding):
    service, path, args, _ = binding
    before = path.read_bytes()
    service.read_ids.clear()
    with pytest.raises(ValueError, match='read all'):
        service.call('bind_verify_check', args)
    assert path.read_bytes() == before


@pytest.mark.parametrize('replacement', [False, True])
def test_passing_partial_declaration_keeps_other_obligations_unverified(binding, replacement):
    from test_verify_plan import complex_route

    from skillforge.domain.skills import SkillCompilationResult
    from skillforge.repository.skill_registry import SkillRegistryBuilder
    from skillforge.repository.skill_router import SkillRouter

    service, path, args, _ = binding
    # Use the existing multi-obligation unit fixture, not a one-step mined workflow.
    service.route = complex_route(service.registry.repository_id)
    match = service.route.matches[0]
    skill = replace(match.skill, constraints=(), risks=(), verify_rule_ids=(),
                    evidence_ids=('selected', 'step-ev'))
    registry = SkillRegistryBuilder().build(SkillCompilationResult(
        service.registry.repository_id, 'snap', (skill,), (), tuple(
            item for item in service.route.evidence if item.evidence_id in skill.evidence_ids)))
    service.route = SkillRouter().route(registry, service.route.request)
    service.read_ids = {match.skill.skill_id for match in service.route.matches}
    ref = VerifyPlanBuilder().build(service.route).obligations[0].target_ref
    args = {**args, 'skill_id': ref.skill_id, 'targets': {ref.kind.value: [ref.ref_id]}}
    value = json.loads(path.read_text('utf-8'))
    value['checks'][0]['argv'] = [sys.executable, '-c', 'pass']
    if replacement:
        value['checks'][0]['covers'] = [{
            'kind': ref.kind.value, 'skill_id': ref.skill_id, 'ref_id': 'old_step'}]
        args['replace_covers'] = value['checks'][0]['covers']
    path.write_text(json.dumps(value), 'utf-8')
    service.call('bind_verify_check', args)
    report = service.call('verify_patch', {})['report']
    assert len(report['obligations']) > 1
    assert report['checks'][0]['status'] == 'PASSED'
    assert report['overall_status'] != 'PASSED'
    assert any(item['code'] == 'UNVERIFIED_OBLIGATION' for item in report['findings'])


def test_stale_head_cannot_bind(binding, monkeypatch):
    service, path, args, _ = binding
    before = path.read_bytes()
    monkeypatch.setattr('skillforge.integration.mcp_server._head', lambda root: 'changed')
    with pytest.raises(ValueError, match='stale'):
        service.call('bind_verify_check', args)
    assert path.read_bytes() == before


def obsolete_binding(path, ref):
    value = json.loads(path.read_text('utf-8'))
    previous = [{'kind': ref.kind.value, 'skill_id': ref.skill_id,
                 'ref_id': 'previous_version_step'}]
    value['checks'][0]['covers'] = previous
    path.write_text(json.dumps(value), 'utf-8')
    return previous


def test_explicit_replacement_removes_old_refs_without_changing_command(binding):
    service, path, args, ref = binding
    previous = obsolete_binding(path, ref)
    original, _ = load_checks(service.root, path)
    with pytest.raises(ValueError, match='unknown obligation'):
        service.call('verify_patch', {})
    result = service.call('bind_verify_check', {**args, 'replace_covers': previous})
    updated, _ = load_checks(service.root, path)
    assert replace(original[0], covers=(ref,)) == updated[0]
    assert result['status'] == 'DECLARED_NOT_VERIFIED'
    assert result['replaced_refs'] == previous
    report = service.call('verify_patch', {})['report']
    assert report['checks'][0]['status'] == 'FAILED'
    assert report['overall_status'] != 'PASSED'


def test_additive_binding_does_not_silently_remove_old_refs(binding):
    service, path, args, ref = binding
    previous = obsolete_binding(path, ref)
    service.call('bind_verify_check', args)
    value = json.loads(path.read_text('utf-8'))
    assert previous[0] in value['checks'][0]['covers']
    with pytest.raises(ValueError, match='unknown obligation'):
        service.call('verify_patch', {})


@pytest.mark.parametrize('previous', [None, {}, 'not-an-array', [],
    [{'kind': 'UNKNOWN', 'skill_id': 'other', 'ref_id': 'old'}],
    [{'kind': 'WORKFLOW_STEP', 'skill_id': 'other', 'ref_id': 'old'}],
    [{'kind': 'WORKFLOW_STEP', 'skill_id': 'other', 'ref_id': 'old'}] * 2,
    [{'kind': 'WORKFLOW_STEP', 'skill_id': 'other', 'ref_id': str(i)} for i in range(17)],
])
def test_wrong_previous_binding_cannot_overwrite_current_check(binding, previous):
    service, path, args, ref = binding
    obsolete_binding(path, ref)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        service.call('bind_verify_check', {**args, 'replace_covers': previous})
    assert path.read_bytes() == before


def test_replacement_leaves_other_commands_untouched(binding):
    service, path, args, ref = binding
    previous = obsolete_binding(path, ref)
    value = json.loads(path.read_text('utf-8'))
    other = {**value['checks'][0], 'check_id': 'other'}
    value['checks'].append(other)
    path.write_text(json.dumps(value), 'utf-8')
    original, _ = load_checks(service.root, path)
    service.call('bind_verify_check', {**args, 'replace_covers': previous})
    updated, _ = load_checks(service.root, path)
    assert updated[1] == original[1]


def test_same_current_replacement_is_idempotent(binding):
    service, path, args, ref = binding
    service.call('bind_verify_check', args)
    expected = json.loads(path.read_text('utf-8'))['checks'][0]['covers']
    before = path.read_bytes()
    result = service.call('bind_verify_check', {**args, 'replace_covers': expected})
    assert path.read_bytes() == before
    assert result['status'] == 'DECLARED_NOT_VERIFIED'
