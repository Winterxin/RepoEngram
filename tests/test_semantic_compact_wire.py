"""Lossless identity round trips, real adapter gates and scope isolation offline."""
import json
from copy import deepcopy

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs, SingleFixture, run
from test_semantic_batch_transport import recorded_models, wire_response
from test_prepare import historical_repository as historical_repository, repository as repository
from skillforge.repository.semantic_batch import _adapter, candidate_identity, request_id
from skillforge.repository.semantic_wire import CompactBatch, CandidateAliases, WIRE_VERSION
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.evolution_artifacts import load_baseline


def batch(model,requests):
    entries=[];schemas=[]
    for request in requests:
        _,prompt,schema,_=_adapter(model,request)
        entries.append({'request_id':request_id(request),'repository_id':candidate_identity(request)[0],
                        'candidate_id':candidate_identity(request)[1],'original_prompt':prompt(request)})
        schemas.append(schema)
    return CompactBatch(entries,schemas)


@pytest.mark.parametrize('channel',[0,1,2])
def test_exact_context_and_policy_round_trip_and_shorter_schema(eleven_inputs,channel):
    _,_,models=recorded_models(eleven_inputs)
    fixture=run(eleven_inputs,SingleFixture)
    codec=batch(models[channel],fixture[2][channel].requests)
    assert len(codec.policies)==len(codec.response_schemas)==1
    for key,original in codec.original_prompts.items():
        policy,rest=original.split('\nRESPONSE_SCHEMA:\n',1)
        schema,context=rest.split('\nCANDIDATE_CONTEXT:\n',1)
        candidate=next(c for c in codec.candidates if c['request_id']==key)
        assert codec.policies[candidate['policy']]==policy
        assert codec.response_schemas[candidate['response_schema']]==json.loads(schema)
        assert codec.aliases[key].decode(candidate['CANDIDATE_CONTEXT'])==json.loads(context)
    if len(codec.candidates)>1:
        assert len(codec.prompt.encode()) < sum(len(s.encode()) for s in codec.original_prompts.values())
    from jsonschema import Draft202012Validator, ValidationError
    raw={'results':{}}
    for c in codec.candidates:
        key=c['request_id'];request=fixture[2][channel].requests[int(key[1:])]
        response=wire_response(fixture[2][channel].responder.complete(request))
        raw['results'][key]={k:c[k] for k in ('request_id','repository_id','candidate_id')}
        raw['results'][key].update(response=codec.aliases[key].encode(response),error=None)
    Draft202012Validator(codec.schema).validate(raw)
    incomplete=deepcopy(raw);incomplete['results'].pop('r0')
    with pytest.raises(ValidationError):Draft202012Validator(codec.schema).validate(incomplete)
    decoded,failures=codec.normalize(raw)
    assert not failures
    for item in decoded['results']:
        request=next(r for r in fixture[2][channel].requests if request_id(r)==item['request_id'])
        assert item['response']==wire_response(fixture[2][channel].responder.complete(request))


def test_scope_alias_cannot_be_rebound_and_prose_is_unchanged():
    original={'repository_id':'a'*20,'evidence_ids':['b'*24],
              'path':'b'*24+'.py','message':'commit '+ 'b'*24}
    aliases=CandidateAliases('r0',original)
    assert aliases.decode(aliases.encode(original))==original
    assert aliases.encode(original)['message']==original['message']
    assert aliases.encode(original)['evidence_ids']==original['evidence_ids']
    commit='c'*40
    literal=CandidateAliases('r0',{'commit_id':commit,'message':'Revert '+commit[:7]})
    assert literal.encode_ids=={}
    with pytest.raises(ValueError,match='foreign'):
        aliases.decode({'evidence_ids':['r1_i1']})
    with pytest.raises(ValueError,match='unknown'):
        aliases.decode({'evidence_ids':['r0_i999']})


def test_compact_tendency_keeps_fingerprint_and_cached_binding(tmp_path,monkeypatch):
    import test_semantic_batch_transport as original
    factory=original.complete_identity
    def identity():
        result=factory();result['response_parameters']['semantic_wire']=WIRE_VERSION
        return result
    monkeypatch.setattr(original,'complete_identity',identity)
    original.test_tendency_batch_keeps_exact_request_binding(tmp_path)


def test_generated_role_and_step_labels_are_not_input_identity_aliases():
    aliases = CandidateAliases('r0', {'evidence_ids': ['a' * 24]})
    response = {'role_id': 'r0_i999', 'steps': [
        {'step_id': 'r1_i999', 'evidence_ids': ['r0_i0']}]}
    assert aliases.decode(response) == {'role_id': 'r0_i999', 'steps': [
        {'step_id': 'r1_i999', 'evidence_ids': ['a' * 24]}]}
    assert aliases.encode({'role_id': 'b' * 24}) == {'role_id': 'b' * 24}


def test_shared_and_local_evidence_publish_original_order_without_repairing_responses():
    schema = {'type': 'object', 'properties': {'evidence_ids': {
        'type': 'array', 'items': {'type': 'string'}}}}
    ids = ['1' * 24, '8' * 24, 'f' * 24]
    contexts = [{'evidence': [{'evidence_id': key} for key in ids]},
                {'evidence': [{'evidence_id': key} for key in ['8' * 24, '9' * 24]]}]
    entries = [{'request_id': str(i), 'repository_id': 'repo', 'candidate_id': str(i),
                'original_prompt': 'policy\nRESPONSE_SCHEMA:\n' + json.dumps(schema)
                + '\nCANDIDATE_CONTEXT:\n' + json.dumps(context)}
               for i, context in enumerate(contexts)]
    codec = CompactBatch(entries, [schema, schema])
    payload = json.loads(codec.prompt.split('SEMANTIC_COMPACT:\n', 1)[1])
    listed = payload['REFERENCE_BINDINGS']['r0']['evidence']
    assert codec.aliases['r0'].decode({'evidence_ids': listed})['evidence_ids'] == ids
    # The adapter never sorts, removes duplicates or inserts missing citations.
    wrong = list(reversed(listed)) + [listed[0]]
    assert codec.aliases['r0'].decode({'evidence_ids': wrong})['evidence_ids'] == list(reversed(ids)) + [ids[0]]


def test_structured_and_unstructured_schema_are_not_merged(eleven_inputs,tmp_path):
    from test_structural_evidence_binding import case
    (tmp_path/'structured').mkdir()
    _,_,structured,_=case.__wrapped__(tmp_path/'structured')
    _,_,models=recorded_models(eleven_inputs)
    fixture=run(eleven_inputs,SingleFixture)
    codec=batch(models[0],(fixture[2][0].requests[0],structured))
    assert len(codec.response_schemas)==2
    assert {c['response_schema'] for c in codec.candidates}=={'s0','s1'}
    for c in codec.candidates:
        key=c['request_id'];_,rest=codec.original_prompts[key].split('\nRESPONSE_SCHEMA:\n',1)
        _,context=rest.split('\nCANDIDATE_CONTEXT:\n',1)
        assert codec.aliases[key].decode(c['CANDIDATE_CONTEXT'])==json.loads(context)


@pytest.mark.parametrize('mode,calls',[('success',3),('malformed',4),('gate_failure',4),
    ('partial',4),('duplicate',4),('candidate_error',4)])
def test_compact_native_prepare_preserves_full_knowledge_and_failure_isolation(eleven_inputs,tmp_path,mode,calls):
    expected,runner,models=recorded_models(eleven_inputs,mode)
    for model in models:model.semantic_cache_identity['response_parameters']['semantic_wire']=WIRE_VERSION
    result=SkillForgePreparer(*models,semantic_cache_root=tmp_path/'cache').prepare(
        PrepareRequest(eleven_inputs[0],'update routes',CodeAgentTarget.CODEX,'app.api.route0'))
    assert len(runner.calls)==calls
    assert sum('SEMANTIC_COMPACT:' in prompt for prompt in runner.calls)==3
    reference=run(eleven_inputs,SingleFixture,mode if mode in ('malformed','gate_failure') else 'success')
    baseline=load_baseline(eleven_inputs[0],result.evolution_baseline_path)
    assert baseline.source_dna==reference[1][0]
    assert baseline.source_compilation==reference[1][1]
    assert baseline.semantic_structural==reference[0].structural
