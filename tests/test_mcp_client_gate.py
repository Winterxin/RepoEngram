import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evals'))
from mcp_client_gate import require_inventory,REQUIRED,assert_client_ready


def test_handshake_timeout_cannot_authorize_actor():
    with pytest.raises(RuntimeError,match='INVALID_INFRASTRUCTURE'):
        require_inventory([{'result':{'data':[{'name':'skillforge','tools':{},'toolsError':'timed out handshaking after 30s'}]}}])


def test_partial_inventory_cannot_authorize_actor():
    with pytest.raises(RuntimeError,match='incomplete tools'):
        require_inventory([{'result':{'data':[{'name':'skillforge','tools':{'get_skill':{}}}]}}])


def test_all_production_tools_are_required_without_synthesizing_coverage():
    require_inventory([{'result':{'data':[{'name':'skillforge','tools':dict.fromkeys(REQUIRED,{})}]}}])


def test_actual_gate_records_failure_with_zero_models_and_only_metadata_methods(tmp_path,monkeypatch):
    import io,json
    import mcp_client_gate as gate
    calls=[]
    class Process:
        stdin=io.StringIO()
        stdout=[json.dumps({'id':1,'result':{}})+'\n',json.dumps({'id':2,'result':{'data':[
            {'name':'skillforge','tools':{},'toolsError':'handshake timeout'}]}})+'\n']
        def wait(self,timeout):return 0
    def spawn(argv,**kwargs):calls.append(argv);return Process()
    monkeypatch.setattr(gate.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError,match='INVALID_INFRASTRUCTURE'):
        assert_client_ready(['codex','exec','-c','mcp_servers.skillforge.command="protected-launcher"'],tmp_path,{},tmp_path/'proof')
    proof=json.loads((tmp_path/'proof/proof.json').read_text('utf-8'))
    assert proof['Agent_runs']==proof['live_models']==0
    assert proof['requested_methods']==['initialize','mcpServerStatus/list']
    assert calls[0][:2]==['codex','app-server']
    assert 'mcp_servers.skillforge.command="protected-launcher"' in calls[0]
    events=[json.loads(l) for l in (tmp_path/'proof/cost-events.jsonl').read_text('utf-8').splitlines()]
    assert len(events)==2 and all(e['real_model_calls']==0 and e['payload_sha256'] and e['end_ns']>=e['start_ns'] for e in events)


def test_client_exit_does_not_wait_for_the_full_startup_deadline(tmp_path,monkeypatch):
    import io,time
    import mcp_client_gate as gate
    class Process:
        stdin=io.StringIO()
        stdout=[]
        def wait(self,timeout):return 0
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:Process())
    start=time.monotonic()
    with pytest.raises(RuntimeError,match='exited before replying'):
        assert_client_ready(['codex','exec'],tmp_path,{},tmp_path/'proof')
    assert time.monotonic()-start<3


def test_catalog_drift_rejected_before_starting_client(tmp_path,monkeypatch):
    import json
    import mcp_client_gate as gate
    from semantic_catalog import catalog_identity
    catalog=tmp_path/'models.json'
    catalog.write_text(json.dumps({'models':[{'slug':'gpt-6-luna','base_instructions':'frozen'}]}),'utf-8')
    identity=catalog_identity(catalog,'gpt-6-luna')
    catalog.write_text(json.dumps({'models':[{'slug':'gpt-6-luna','base_instructions':'changed'}]}),'utf-8')
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:pytest.fail('spawn after catalog drift'))
    with pytest.raises(RuntimeError,match='catalog drift'):
        assert_client_ready(['codex','exec'],tmp_path,{},tmp_path/'proof',model_catalog=identity)
