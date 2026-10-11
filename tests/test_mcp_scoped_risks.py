"""Published risks are readable without a workflow, with no discovery payload growth."""
from dataclasses import replace

import pytest
from test_skill_compiler import anti, dna

from skillforge.domain.project_dna import ProjectKnowledgeStatus as Status
from skillforge.domain.routing import SkillRoutingRequest, routing_trace
from skillforge.domain.semantic_antipattern import SemanticAntiPatternScopeKind as Scope
from skillforge.integration.mcp_server import ProjectTools
from skillforge.integration.scoped_risks import _size, discovery_with_risks
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def setup(source, scope=None, task='Change the exact source safely and preserve existing data'):
    registry = SkillRegistryBuilder().build(SkillCompiler().compile(source))
    route = SkillRouter().route(registry, SkillRoutingRequest('repo', task, **(scope or {})))
    response = {'repository_id': 'repo', 'skills': [],
                'router_trace': routing_trace(registry, route)}
    return registry, route, response


def result(patterns, scope=None, consumed=None):
    source = dna(workflows=(), anti_patterns=tuple(patterns))
    _, route, response = setup(source, scope)
    reads = set() if consumed is None else consumed
    output = discovery_with_risks(response, source, route, scope or {}, reads)
    assert _size(output) <= _size(response)
    assert _size(output, escaped=True) <= _size(response, escaped=True)
    return output, reads


def test_risk_without_workflow_preserves_action_exception_and_status():
    p = anti('risk', Status.CANDIDATE)
    output, reads = result([p], {'current_file': 'src/a.py'})
    assert output['skills'] == []
    entry, = output['source_risks']
    assert entry['avoid_behavior'] == p.draft.avoid_behavior
    assert entry['exceptions'] == [e.description for e in p.draft.exceptions]
    assert entry['source_status'] == Status.CANDIDATE.value
    assert entry['scope']['selectors'] == ['src/a.py']
    assert reads == {'risk'}
    assert 'Not formal verification' in output['risk_policy']
    assert 'workflow_id' not in entry and 'support' not in entry


@pytest.mark.parametrize('scope', [{}, {'current_file': 'src/b.py'},
                                   {'current_module': 'src.a'},
                                   {'current_directory': 'src'}])
def test_explicit_file_scope_never_guesses_location(scope):
    output, reads = result([anti('risk', Status.CANDIDATE)], scope)
    assert 'source_risks' not in output
    assert reads == set()


@pytest.mark.parametrize('status', [Status.DEPRECATED, Status.CONFLICTED])
def test_unusable_status_is_not_delivered(status):
    output, _ = result([anti('risk', status)], {'current_file': 'src/a.py'})
    assert 'source_risks' not in output


def test_repository_risk_and_read_receipt():
    p = anti('risk', Status.CANDIDATE, Scope.REPOSITORY)
    first, reads = result([p])
    assert first['source_risks'][0]['avoid_behavior'] == p.draft.avoid_behavior
    second, _ = result([p], consumed=reads)
    assert second['source_risks'][0]['already_read']
    assert 'avoid_behavior' not in second['source_risks'][0]


def test_oversized_risk_does_not_clip_exception_or_mark_read():
    p = anti('risk', Status.CANDIDATE)
    p = replace(p, draft=replace(p.draft, avoid_behavior='x' * 5000))
    output, reads = result([p], {'current_file': 'src/a.py'})
    assert output['source_risks'] == []
    assert output['omitted_source_risk_count'] == 1
    assert reads == set()


def test_limits_and_determinism():
    patterns = [anti(str(i), Status.CANDIDATE) for i in range(6)]
    first, reads = result(patterns, {'current_file': 'src/a.py'})
    second, _ = result(list(reversed(patterns)), {'current_file': 'src/a.py'})
    assert first == second
    assert len(first['source_risks']) <= 3
    assert len(first['source_risks']) + first['omitted_source_risk_count'] == 6
    assert len(reads) == len(first['source_risks'])


def test_already_hosted_risk_not_repeated():
    source = dna(anti_patterns=(anti('risk', Status.CANDIDATE),))
    _, route, response = setup(source, {'current_file': 'src/a.py'}, 'Do {name}')
    assert route.matches[0].skill.risks
    assert discovery_with_risks(response, source, route,
                                {'current_file': 'src/a.py'}, set()) is response


def test_service_exposes_risk_and_evidence_and_clears_receipt(monkeypatch, tmp_path):
    source = dna(workflows=(), anti_patterns=(anti('risk', Status.CANDIDATE),))
    registry, _, _ = setup(source)
    monkeypatch.setattr('skillforge.integration.mcp_server.repo_root', lambda _: tmp_path)
    monkeypatch.setattr('skillforge.integration.mcp_server.load_snapshot',
                        lambda _: (source, registry, 'base'))
    monkeypatch.setattr('skillforge.integration.mcp_server._head', lambda _: 'base')
    service = ProjectTools(tmp_path)
    args = {'task': 'Change the exact source safely and preserve existing data',
            'scope': {'current_file': 'src/a.py'}}
    first = service.call('get_related_workflow', args)
    assert first['source_risks'][0]['avoid_behavior']
    second = service.call('get_related_workflow', args)
    assert second['source_risks'][0]['already_read']
    evidence = service.call('get_evidence', {'rule_id': 'risk'})
    assert [e['evidence_id'] for e in evidence] == ['risk']
    third = service.call('get_related_workflow', {**args, 'task': args['task'] + ' carefully'})
    assert third['source_risks'][0]['avoid_behavior']
    assert 'already_read' not in third['source_risks'][0]
    monkeypatch.setattr('skillforge.integration.mcp_server._head', lambda _: 'new-head')
    with pytest.raises(ValueError, match='stale'):
        service.call('get_related_workflow', args)


def test_unicode_and_escaped_payload_limits_do_not_lose_exceptions():
    p = anti('risk', Status.CANDIDATE)
    p = replace(p, draft=replace(p.draft, avoid_behavior='保留\\路径与"引用"'))
    output, reads = result([p], {'current_file': 'src/a.py'})
    if output['source_risks']:
        assert output['source_risks'][0]['avoid_behavior'] == p.draft.avoid_behavior
        assert output['source_risks'][0]['exceptions'] == ['except legacy']
        assert reads == {'risk'}
    else:
        assert output['omitted_source_risk_count'] == 1
        assert not reads
