"""Fresh task orchestration must not inherit paid U07 data or arbitrary caps."""
import importlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evals'))


@pytest.fixture(autouse=True)
def isolate_campaign_configuration(monkeypatch):
    # These scripts configure a standalone coordinator process. Keep that
    # process-local configuration from leaking into other harness test modules.
    common=importlib.import_module('u07_cost_control')
    for owner,names in ((common,('ARTIFACTS','ORIGINAL_CONTROL','RESUME_TREATMENT',
            'RECOVER_PAID','PINNED_STARTUP','campaign_budget')),
            (common.recovery,('TASK','RUNTIME','RECOVERY','CAMPAIGN','OUT',
            'AGENT_WALL_SECONDS','ACTOR_PYTEST_GUARD','AGENT_OUTPUT_TOKENS'))):
        for name in names:monkeypatch.setattr(owner,name,getattr(owner,name))


def module():
    if 'u08_observation' in sys.modules:return importlib.reload(sys.modules['u08_observation'])
    return importlib.import_module('u08_observation')


def test_fresh_task_has_independent_bounded_plan_without_paid_recovery(monkeypatch):
    for key in ('SKILLFORGE_U07_RESUME','SKILLFORGE_U07_RECOVER_PAID',
                'SKILLFORGE_U07_PINNED_STARTUP','SKILLFORGE_U07_LOCK_RECOVERY'):
        monkeypatch.setenv(key,'1')
    u=module()
    assert u.r.TASK=='U08' and u.common.PINNED_STARTUP
    assert not u.common.RESUME_TREATMENT and not u.common.RECOVER_PAID
    assert u.r.AGENT_WALL_SECONDS==180
    assert u.budget().max_calls==5 and u.budget().cold_shape==u.SHAPE
    assert u.budget().semantic_limit==164.720042
    assert all('U07' not in str(p) for p in (u.r.CAMPAIGN,u.r.RECOVERY,u.OUT))


def test_model_budget_is_reduced_before_spawn_when_time_already_consumed(tmp_path,monkeypatch):
    u=module();monkeypatch.setattr(u.r,'CAMPAIGN',tmp_path)
    (tmp_path/'prepare-deadline-plan.json').write_text(json.dumps({'prepare_seconds':100}),'utf-8')
    assert u.budget().prepare_limit==100
    assert u.budget().max_calls==5


def test_wrong_candidate_shape_stops_before_a_model_call():
    u=module();b=u.budget()
    from skillforge.cost_control import BudgetExceeded
    with pytest.raises(BudgetExceeded,match='differs'):
        b.validate_shape({**u.SHAPE,'structural':2})
    assert b.calls==0


def test_no_prepare_spawn_without_complete_actor_and_scoring_allowance(tmp_path,monkeypatch):
    u=module()
    monkeypatch.setattr(u,'require_observation',lambda **kwargs:None)
    for name in ('validate_offline','require_completed_control','require_headroom'):
        monkeypatch.setattr(u.common,name,lambda:None)
    monkeypatch.setattr(u.r,'require_preflight',lambda arm:None)
    monkeypatch.setattr(u.r,'CAMPAIGN',tmp_path)
    monkeypatch.setattr(u,'remaining',lambda:100)
    monkeypatch.setattr(u.r,'run_measured',lambda *a,**k:pytest.fail('prepare spawned without full allowance'))
    from skillforge.cost_control import BudgetExceeded
    with pytest.raises(BudgetExceeded,match='BEFORE Semantic'):
        u.run_treatment()


def test_invalid_actual_control_stops_before_treatment_or_prepare(tmp_path,monkeypatch):
    u=module();control=tmp_path/'control';control.mkdir()
    monkeypatch.setattr(u,'require_observation',lambda **kwargs:None)
    (control/'result.json').write_text(json.dumps({'exit_code':0,'timeout':False}),'utf-8')
    (control/'freeze.json').write_text('{}','utf-8')
    (control/'events.jsonl').write_text(json.dumps({'type':'turn.started'})+'\n'+
        json.dumps({'type':'turn.failed','error':{'message':'usage limit'}}),'utf-8')
    monkeypatch.setattr(u.common,'control_folder',lambda:control)
    monkeypatch.setattr(u.common,'validate_offline',lambda:None)
    monkeypatch.setattr(u.r,'run_measured',lambda *a,**k:pytest.fail('invalid Control spent prepare'))
    monkeypatch.setattr(u.r,'freeze_and_run',lambda *a,**k:pytest.fail('invalid Control spent Treatment'))
    from actor_outcome import InvalidActorRun
    with pytest.raises(InvalidActorRun,match='ACCOUNT_USAGE_LIMIT'):
        u.run_treatment()


def test_invalid_control_is_not_reported_as_scored(tmp_path,monkeypatch,capsys):
    u=module();monkeypatch.setattr(u.r,'CAMPAIGN',tmp_path/'campaign')
    monkeypatch.setattr(u,'require_observation',lambda **kwargs:None)
    monkeypatch.setattr(u,'OUT',tmp_path)
    (tmp_path/'observation-clock.json').write_text(json.dumps({'start_ns':1}),'utf-8')
    for owner,name in ((u.common,'validate_offline'),(u,'clean_roles')):
        monkeypatch.setattr(owner,name,lambda:None)
    monkeypatch.setattr(u.r,'require_preflight',lambda arm:None)
    monkeypatch.setattr(u.r,'freeze_and_run',lambda arm:None)
    from actor_outcome import InvalidActorRun
    def invalid():raise InvalidActorRun('INVALID_INFRASTRUCTURE: ACCOUNT_USAGE_LIMIT')
    monkeypatch.setattr(u.common,'require_completed_control',invalid)
    monkeypatch.setattr(u.r,'grade_arm',lambda arm:pytest.fail('invalid Actor was scored'))
    with pytest.raises(InvalidActorRun):u.run_control()
    assert 'SCORED' not in capsys.readouterr().out


@pytest.mark.parametrize('task',('U09','U10'))
def test_fresh_task_is_propagated_to_prepare_worker_without_recovery(task):
    u=module();u.configure(task)
    assert u.r.TASK==task and task.lower()+'-budgeted-observation' in str(u.OUT)
    assert 'U08' not in str(u.common.ORIGINAL_CONTROL)
    assert u.worker_argv('prepare')[-3:]==['prepare','--task',task]
    assert u.budget().max_calls==5 and u.r.AGENT_WALL_SECONDS==180
    assert not u.common.RECOVER_PAID and not u.common.RESUME_TREATMENT


def test_unqualified_task_is_rejected_before_any_launch():
    u=module()
    with pytest.raises(ValueError,match='qualified'):u.configure('U11')


def test_qualification_does_not_age_observation_clock(tmp_path,monkeypatch):
    u=module();u.configure('U10');monkeypatch.setattr(u,'OUT',tmp_path)
    monkeypatch.setattr(u.r,'CAMPAIGN',tmp_path/'campaign')
    monkeypatch.setattr(u.r,'RECOVERY',tmp_path/'preflight')
    monkeypatch.setattr(u.common,'validate_offline',lambda:None)
    monkeypatch.setattr(u,'clean_roles',lambda:None)
    monkeypatch.setattr(u.common,'ARTIFACTS',tmp_path/'offline')
    u.common.ARTIFACTS.mkdir();(u.common.ARTIFACTS/'budget-plan.json').write_text('{}')
    seen=[];monkeypatch.setattr(u.r,'preflight',seen.append)
    import semantic_catalog
    monkeypatch.setattr(semantic_catalog,'catalog_identity',lambda *args:{'sha256':'frozen'})
    u.infrastructure(start_clock=False)
    assert seen==['control','treatment']
    assert not (tmp_path/'observation-clock.json').exists()
    assert u.read(tmp_path/'infrastructure-ready.json')['observation_started'] is False
    u.infrastructure()
    assert (tmp_path/'observation-clock.json').exists()
    assert u.read(tmp_path/'infrastructure-ready.json')['observation_started'] is True
    u.require_observation()
    with pytest.raises(FileExistsError):u.infrastructure()


@pytest.mark.parametrize('launch',('control/launch.json','treatment/launch.json','prepare-worker-launch.json'))
def test_clock_cannot_restart_after_any_paid_phase(tmp_path,monkeypatch,launch):
    u=module();monkeypatch.setattr(u,'OUT',tmp_path)
    monkeypatch.setattr(u.r,'CAMPAIGN',tmp_path/'campaign')
    target=u.r.CAMPAIGN/launch;target.parent.mkdir(parents=True);target.write_text('{}')
    monkeypatch.setattr(u.common,'validate_offline',lambda:None)
    monkeypatch.setattr(u,'clean_roles',lambda:None)
    monkeypatch.setattr(u.r,'preflight',lambda arm:pytest.fail('paid phase caused fresh preflight'))
    from skillforge.cost_control import BudgetExceeded
    with pytest.raises(BudgetExceeded,match='clock reset'):u.infrastructure()
    assert not (tmp_path/'observation-clock.json').exists()


@pytest.mark.parametrize('phase',('run_control','run_treatment'))
def test_qualification_without_clock_cannot_launch_a_paid_phase(tmp_path,monkeypatch,phase):
    u=module();monkeypatch.setattr(u,'OUT',tmp_path)
    monkeypatch.setattr(u.r,'freeze_and_run',lambda *args:pytest.fail('Actor started'))
    monkeypatch.setattr(u.r,'run_measured',lambda *args,**kwargs:pytest.fail('prepare started'))
    from skillforge.cost_control import BudgetExceeded
    with pytest.raises(BudgetExceeded,match='not started BEFORE'):getattr(u,phase)()


def test_observation_clock_rejects_task_budget_drift_and_insufficient_allowance(tmp_path,monkeypatch):
    u=module();monkeypatch.setattr(u,'OUT',tmp_path)
    monkeypatch.setattr(u.common,'ARTIFACTS',tmp_path/'offline');u.common.ARTIFACTS.mkdir()
    plan=u.common.ARTIFACTS/'budget-plan.json';plan.write_text('{}')
    clock=tmp_path/'observation-clock.json'
    from skillforge.cost_control import BudgetExceeded
    clock.write_text(json.dumps({'task':'U09','budget_plan_sha256':u.r.digest(plan)}))
    with pytest.raises(BudgetExceeded,match='binding drift'):u.require_observation()
    clock.write_text(json.dumps({'task':u.r.TASK,'budget_plan_sha256':'wrong'}))
    with pytest.raises(BudgetExceeded,match='binding drift'):u.require_observation()
    clock.write_text(json.dumps({'task':u.r.TASK,'budget_plan_sha256':u.r.digest(plan)}))
    monkeypatch.setattr(u,'remaining',lambda:210)
    with pytest.raises(BudgetExceeded,match='allowance exhausted'):u.require_observation(reserve=210)
