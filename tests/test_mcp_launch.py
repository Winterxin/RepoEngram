import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evals'))
from mcp_launch import protected_mcp_overrides, metadata_overrides, EXECUTION_ENV
from skillforge.cost_control import BudgetExceeded


def environment(root):
    home=root/'.runtime/codex-home';home.mkdir(parents=True)
    (home/'models_cache.json').write_text(json.dumps({'models':[{
        'slug':'gpt-6-luna','base_instructions':'unchanged specification'}]}),'utf-8')
    values={key:'frozen-input' for key in EXECUTION_ENV}
    values.update(CODEX_HOME=str(home),GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',
                  GIT_CONFIG_VALUE_0=str(root),SKILLFORGE_ACTOR_ROOT=str(root),
                  PYTEST_PLUGINS='skillforge.verify.actor_pytest',
                  OPENAI_API_KEY='must-not-forward',UNRELATED_HOST_SECRET='must-not-forward')
    return values


def test_real_launcher_explicitly_preserves_filtered_execution_inputs_and_sandbox(tmp_path):
    env=environment(tmp_path)
    original=['-c','mcp_servers.skillforge.command="python"',
              '-c','mcp_servers.skillforge.args=[]',
              '-c','mcp_servers.skillforge.env.PYTHONPATH="obsolete"',
              '-c','mcp_servers.skillforge.tools.verify_patch.approval_mode="approve"']
    result=protected_mcp_overrides(original,codex='frozen-codex',python='frozen-python',
        workspace=tmp_path,permissions=['-c','permission-proof=true'],env=env)
    config=dict(item.split('=',1) for item in result[1::2])
    assert json.loads(config['mcp_servers.skillforge.command'])=='frozen-codex'
    argv=json.loads(config['mcp_servers.skillforge.args'])
    assert argv[:5]==['sandbox','-P','unseen-sensitive','-C',str(tmp_path)]
    assert 'permission-proof=true' in argv and 'features.remote_plugin=false' in argv
    assert argv[-6:]==['frozen-python','-u','-m','skillforge.integration.mcp_server','--repo',str(tmp_path)]
    assert config['mcp_servers.skillforge.tools.verify_patch.approval_mode']=='"approve"'
    forwarded={key.split('.env.',1)[1]:json.loads(value) for key,value in config.items() if '.env.' in key}
    assert forwarded=={key:env[key] for key in EXECUTION_ENV}
    assert not any('must-not-forward' in item for item in result)


def test_control_metadata_staging_preserves_original_model_spec_without_skill_prepare(tmp_path):
    env=environment(tmp_path)
    env.pop('SKILLFORGE_ACTOR_ROOT');env.pop('PYTEST_PLUGINS')
    before=json.loads((Path(env['CODEX_HOME'])/'models_cache.json').read_text('utf-8'))
    first=metadata_overrides(tmp_path,env)
    catalog=tmp_path/'.runtime/semantic-catalog-v1/models.json'
    assert json.loads(catalog.read_text('utf-8'))['models']==before['models']
    assert not (tmp_path/'.skillforge').exists()
    assert metadata_overrides(tmp_path,env)==first


@pytest.mark.parametrize('missing',['CODEX_HOME','PYTHONPATH','TEMP','PYTEST_PLUGINS'])
def test_incomplete_execution_environment_fails_before_launch(tmp_path,missing):
    env=environment(tmp_path);env.pop(missing)
    with pytest.raises((ValueError,BudgetExceeded)):
        protected_mcp_overrides([],codex='codex',python='python',workspace=tmp_path,permissions=[],env=env)


def test_invalid_catalog_fails_before_cli_launch(tmp_path):
    env=environment(tmp_path)
    (Path(env['CODEX_HOME'])/'models_cache.json').write_text('{"models":[]}','utf-8')
    with pytest.raises(BudgetExceeded,match='unverifiable'):
        metadata_overrides(tmp_path,env)


@pytest.mark.parametrize('registration',[
    ['-c','mcp_servers.skillforge.command="python"'],
    ['-c','mcp_servers.skillforge.command="python"','-c','mcp_servers.skillforge.command="other"',
     '-c','mcp_servers.skillforge.args=[]'],
])
def test_ambiguous_or_missing_real_launch_is_rejected(tmp_path,registration):
    with pytest.raises(ValueError,match='registration'):
        protected_mcp_overrides(registration,codex='codex',python='python',workspace=tmp_path,
                                permissions=[],env=environment(tmp_path))
