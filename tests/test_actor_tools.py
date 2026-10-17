"""Actual CLI command-path guards must fail before a model launch."""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evals'))
import actor_tools as gate


def installed(root):
    binary=root/'.runtime/bin';binary.mkdir(parents=True)
    for name in ('rg.exe','bounded_command.py','public_output.py','skillforge-rg.exe'):
        (binary/name).write_bytes(b'fixture tool')
    shell=root/'pwsh.exe';shell.write_bytes(b'fixture shell')
    return {'PATH':str(binary)+';system-tools','TEMP':str(root/'temp')},shell


def response(root,env,path=None,complete=True):
    directory=Path(env['TEMP'])/'command-logs';directory.mkdir(parents=True,exist_ok=True)
    log=directory/'1-stdout.txt';log.write_bytes(b'x'*gate.PROBE_BYTES+b'\n' if complete else b'partial')
    rg=path or root/'.runtime/bin/rg.exe'
    output=('POLICY_PATH:'+str(rg)+'\r\n'+'x'*8192+
        '\r\n[bounded output; complete private log: '+str(log)+'; next offset: 8192]\r\n'+
        'Read another page: python "public_output.py" "'+str(log)+'" --offset 8192\r\n')
    return {'id':2,'result':{'exitCode':0,'stdout':output,'stderr':''}}


def test_explicit_path_is_required_before_spawn(tmp_path,monkeypatch):
    env,shell=installed(tmp_path)
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:pytest.fail('unprotected Actor command spawned'))
    with pytest.raises(RuntimeError,match='lacks shell PATH'):
        gate.assert_tools_bounded(['codex','exec'],tmp_path,env,tmp_path/'proof',shell)
    assert not (tmp_path/'.runtime/output-policy-proof.txt').exists()


def test_wrong_path_order_and_missing_wrapper_fail_closed(tmp_path):
    env,shell=installed(tmp_path)
    with pytest.raises(ValueError,match='lead PATH'):
        gate.tool_environment_overrides(tmp_path,{**env,'PATH':'vendor-tools;'+env['PATH']})
    (tmp_path/'.runtime/bin/rg.exe').unlink()
    with pytest.raises(ValueError,match='missing'):
        gate.tool_environment_overrides(tmp_path,env)


def test_wrong_native_path_and_lost_full_log_are_rejected(tmp_path):
    env,shell=installed(tmp_path)
    with pytest.raises(RuntimeError,match='bypasses'):
        gate.require_bounded_result(response(tmp_path,env,path=tmp_path/'vendor/rg.exe'),tmp_path,Path(env['TEMP'])/'command-logs')
    with pytest.raises(RuntimeError,match='lost'):
        gate.require_bounded_result(response(tmp_path,env,complete=False),tmp_path,Path(env['TEMP'])/'command-logs')


@pytest.mark.parametrize('corruption',['unbounded','wrong-offset','extra-data'])
def test_incorrect_paging_cannot_authorize_actor(tmp_path,corruption):
    env,shell=installed(tmp_path);r=response(tmp_path,env)
    if corruption=='unbounded':r['result']['stdout']=r['result']['stdout'].replace('x'*8192,'x'*10000)
    elif corruption=='wrong-offset':r['result']['stdout']=r['result']['stdout'].replace('next offset: 8192','next offset: 1')
    else:r['result']['stdout']+='x'*10000
    with pytest.raises(RuntimeError):gate.require_bounded_result(r,tmp_path,Path(env['TEMP'])/'command-logs')


def test_command_path_proof_has_zero_model_calls_and_complete_private_log(tmp_path,monkeypatch):
    env,shell=installed(tmp_path);reply=response(tmp_path,env)
    class Process:
        stdin=io.StringIO()
        stdout=[json.dumps({'id':1,'result':{}})+'\n',json.dumps(reply)+'\n']
        def wait(self,timeout):return 0
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:Process())
    args=['codex','exec',*gate.tool_environment_overrides(tmp_path,env)]
    detail=gate.assert_tools_bounded(args,tmp_path,env,tmp_path/'proof',shell)
    assert detail['public_data_bytes']==8192 and detail['private_log_bytes']==gate.PROBE_BYTES+1
    proof=json.loads((tmp_path/'proof/proof.json').read_text('utf-8'))
    assert proof['status']=='PASS' and proof['live_models']==proof['Agent_runs']==0
    assert proof['requested_methods']==['initialize','command/exec']
    events=[json.loads(line) for line in (tmp_path/'proof/cost-events.jsonl').read_text('utf-8').splitlines()]
    assert len(events)==2 and all(e['real_model_calls']==0 and e['payload_bytes'] and e['end_ns']>=e['start_ns'] for e in events)
    assert not (tmp_path/'.runtime/output-policy-proof.txt').exists()


def test_server_exit_fails_without_waiting_for_deadline(tmp_path,monkeypatch):
    env,shell=installed(tmp_path)
    class Process:
        stdin=io.StringIO();stdout=[]
        def wait(self,timeout):return 0
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:Process())
    with pytest.raises(RuntimeError,match='exited'):
        gate.assert_tools_bounded(['codex','exec',*gate.tool_environment_overrides(tmp_path,env)],tmp_path,env,tmp_path/'proof',shell)
    assert not (tmp_path/'.runtime/output-policy-proof.txt').exists()


def test_guard_runtime_drift_cannot_authorize_actor(tmp_path,monkeypatch):
    env,shell=installed(tmp_path);reply=response(tmp_path,env)
    class Process:
        stdin=io.StringIO()
        @property
        def stdout(self):
            (tmp_path/'.runtime/bin/rg.exe').write_bytes(b'changed guard')
            return [json.dumps({'id':1,'result':{}})+'\n',json.dumps(reply)+'\n']
        def wait(self,timeout):return 0
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:Process())
    with pytest.raises(RuntimeError,match='runtime drift'):
        gate.assert_tools_bounded(['codex','exec',*gate.tool_environment_overrides(tmp_path,env)],tmp_path,env,tmp_path/'proof',shell)
    proof=json.loads((tmp_path/'proof/proof.json').read_text('utf-8'))
    assert proof['status']=='INVALID_INFRASTRUCTURE' and proof['Agent_runs']==0
