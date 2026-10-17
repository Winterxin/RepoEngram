"""Ancillary file mentions and conjunctions must not select unrelated workflows."""
from dataclasses import replace

import pytest
from test_skill_registry import compilation
from test_workflow_intent_routing import workflow

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRoutingRequest, routing_trace
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify import VerifyPlanBuilder


def registry(noise='and'):
    ordinary = workflow('queue', 'Preserve queue order and values',
                        'Preserve queue values while reading and replacing entries.')
    signing = workflow('signing', f'Rotate signing keys {noise} credentials',
                       'Rotate signing keys; update docs/signing.rst and tests.')
    signing = replace(signing, trigger=replace(
        signing.trigger, scope=ProjectScope(ProjectScopeKind.REPOSITORY, ())))
    return SkillRegistryBuilder().build(compilation((ordinary, signing)))


@pytest.mark.parametrize('noise', ['and', 'for', 'with', 'update'])
def test_plain_workflow_requires_more_than_generic_trigger_overlap(noise):
    reg = registry(noise)
    req = SkillRoutingRequest('repo', f'Preserve queue values {noise} docs/tests.',
                              current_module='pkg.renderer')
    route = SkillRouter().route(reg, req)
    assert [item.skill.skill_id for item in route.matches] == ['queue']
    plan = VerifyPlanBuilder().build(route)
    assert {item.target_ref.skill_id for item in plan.obligations} == {'queue'}
    trace = routing_trace(reg, route)
    other = next(item for item in trace['candidates'] if item['skill_id'] == 'signing')
    assert not other['selected']


@pytest.mark.parametrize('task', ['Rotate signing keys and credentials',
                                'Renew signing keys with documentation', 'Protect credentials'])
def test_real_goal_or_exact_trigger_remains_discoverable(task):
    reg = registry()
    route = SkillRouter().route(reg, SkillRoutingRequest('repo', task))
    assert [item.skill.skill_id for item in route.matches] == ['signing']


def test_existing_canonical_registry_is_unchanged_by_routing():
    reg = registry()
    before = reg.to_dict()
    SkillRouter().route(reg, SkillRoutingRequest('repo', 'Preserve queue values and docs.',
                                                current_module='pkg.renderer'))
    assert reg.to_dict() == before
