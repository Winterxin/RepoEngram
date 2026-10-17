"""Ordinary CLI commits reuse existing audited evolution; campaign stays strict."""
import hashlib
import time
from pathlib import Path

import pytest
from test_incremental_evolution_v2 import rich_repo as rich_repo, models, calls, full, equivalent, _git
from test_repo_snapshot_semantic_cache import complete_identity

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import automatic_prepare, VerifiedPreparer


def ports():
    result = models()
    # These ports construct deterministic responses in-process; no CLI/network.
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for port in result:
        port.semantic_cache_identity = complete_identity()
        port.semantic_cache_identity['transport']['source_sha256'] = source
        port.semantic_cache_identity['response_parameters'] = {'fixture': 'incremental-v2', 'network': False}
        port._runner = None
    return result


@pytest.mark.parametrize('scenario,expected', [('unrelated', 0), ('local', 1), ('multi', 2)])
def test_ordinary_cli_automatically_refreshes_only_changed_knowledge(rich_repo, monkeypatch,
                                                                    scenario, expected, capsys):
    from skillforge import cli
    root = rich_repo
    providers = ports();preparer = SkillForgePreparer(*providers,
                                                    semantic_cache_root=root / '.skillforge/cache')
    request = PrepareRequest(root, 'Add coordinated routes', CodeAgentTarget.CODEX, 'app.api.route0')
    first = automatic_prepare(request, preparer=preparer)
    before = calls(providers)
    prior_lengths = [len(port.requests) for port in providers]
    if scenario == 'unrelated':
        (root / 'one-off.md').write_text('A single documentation addition.')
    else:
        for group in (1,) if scenario == 'local' else (1, 2):
            path = root / f'src/app/api{group}/route0.py'
            path.write_text('# relocated import\n' + path.read_text())
    _git(root, 'add', '.', ':!.skillforge', ':!.codex')
    _git(root, 'commit', '-qm', 'Record local update')
    full_ports = models()
    full_started = time.monotonic_ns()
    reference = full(root, full_ports)
    full_seconds = (time.monotonic_ns() - full_started) / 1e9
    expected_baseline = load_baseline(root, reference.evolution_baseline_path)
    assert calls(full_ports) == 11
    monkeypatch.setattr(cli, 'SkillForgePreparer', lambda: preparer)
    incremental_started = time.monotonic_ns()
    assert cli.main(['prepare', str(root), '--task', request.task, '--target', 'codex',
                     '--current-module', request.current_module, '--json']) == 0
    incremental_seconds = (time.monotonic_ns() - incremental_started) / 1e9
    import json
    result = json.loads(capsys.readouterr().out)
    assert result['prepare_mode'] == 'incremental'
    assert calls(providers) - before == expected
    # Check the whole resulting knowledge/compilation/Registry/Router, not just counters.
    index = json.loads((root / '.skillforge/productization/prepare/verified-baseline.json').read_text('utf-8'))
    actual = load_baseline(root, index['payload']['baseline'])
    equivalent(root, expected_baseline, actual)
    events = (root / '.skillforge/productization/prepare/cost-events.jsonl').read_text('utf-8').splitlines()
    event = json.loads(events[-1])
    assert event['cache'] == 'incremental' and event['semantic_evaluation_count'] == expected
    assert event['real_semantic_calls'] == (0 if expected == 0 else None)
    assert first.evolution_baseline_path != index['payload']['baseline']
    warm_started = time.monotonic_ns()
    again = automatic_prepare(request, preparer=preparer)
    warm_seconds = (time.monotonic_ns() - warm_started) / 1e9
    assert again.prepare_mode == 'evolution_baseline' and calls(providers) - before == expected
    from skillforge.domain.serialization import to_json_value
    def size(requests):
        return sum(len(json.dumps(to_json_value(item), ensure_ascii=False,
                                  sort_keys=True, separators=(',', ':')).encode('utf-8'))
                   for item in requests)
    incremental_requests = [item for port, offset in zip(providers, prior_lengths)
                            for item in port.requests[offset:]]
    print(json.dumps({'scenario':scenario,'full_replay_calls':calls(full_ports),
                      'automatic_incremental_replay_calls':expected,'following_warm_replay_calls':0,
                      'full_replay_seconds':full_seconds,'incremental_replay_seconds':incremental_seconds,
                      'following_warm_seconds':warm_seconds,
                      'full_request_context_bytes':size(item for port in full_ports for item in port.requests),
                      'incremental_request_context_bytes':size(incremental_requests),
                      'canonical_full_knowledge_parity':True,'live_models':0}))


@pytest.mark.parametrize('strict', (False, True))
def test_dirty_sources_and_strict_campaign_stop_before_semantic(rich_repo, strict):
    root = rich_repo;providers = ports();preparer = SkillForgePreparer(*providers,
                                                    semantic_cache_root=root / '.skillforge/cache')
    request = PrepareRequest(root, 'Add coordinated routes', CodeAgentTarget.CODEX)
    automatic_prepare(request, preparer=preparer)
    (root / 'one-off.md').write_text('committed')
    _git(root, 'add', 'one-off.md');_git(root, 'commit', '-qm', 'Advance HEAD')
    before = calls(providers)
    if strict:
        import json
        index = json.loads((root / '.skillforge/productization/prepare/verified-baseline.json').read_text('utf-8'))
        controlled = VerifiedPreparer(preparer, providers,
                                     config=index['payload']['identity']['config'], ledger=CostLedger(),
                                     budget=SemanticBudget(), allow_incremental=True)
        with pytest.raises(BudgetExceeded, match='snapshot/config drift'):controlled.prepare(request)
    else:
        path=root / 'src/app/api1/route0.py';path.write_text('# dirty\n'+path.read_text())
        with pytest.raises(BudgetExceeded, match='committed source'):
            automatic_prepare(request, preparer=preparer)
    assert calls(providers) == before
