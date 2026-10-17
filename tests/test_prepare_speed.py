"""Normal prepare avoids repeated decoding while retaining source/artifact gates."""
import json
from collections import Counter

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_effect_cost_control import actual_models

from skillforge.cost_control import BudgetExceeded, digest
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.errors import SkillForgeError
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache


@pytest.fixture
def prepared(eleven_inputs, tmp_path):
    models, transport, budget, ledger = actual_models(eleven_inputs, tmp_path)
    tendency = transport.bind(CodexCliTendencySemanticModel(runner=transport))
    preparer = SkillForgePreparer(*models, tendency, semantic_cache_root=tmp_path / '.skillforge/cache')
    request = PrepareRequest(eleven_inputs[0], 'update routes', CodeAgentTarget.CODEX, 'app.api.route0')
    cold = automatic_prepare(request, preparer=preparer, ledger=ledger)
    return preparer, request, cold, budget, ledger


def test_automatic_warm_loads_once_and_still_validates_static_source(prepared, monkeypatch):
    preparer, request, cold, budget, ledger = prepared
    from skillforge.productization import prepare as inner, verified_prepare as outer
    loads = []
    original = outer.load_baseline
    def load(*args, **kwargs):
        loads.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(outer, 'load_baseline', load)
    monkeypatch.setattr(inner, 'load_baseline', load)
    scans = []
    analyze = inner.PythonStaticAnalyzer.analyze
    def scan(*args, **kwargs):
        scans.append(args)
        return analyze(*args, **kwargs)
    monkeypatch.setattr(inner.PythonStaticAnalyzer, 'analyze', scan)
    calls = budget.calls
    warm = automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert len(loads) == 1 and len(scans) == 1
    assert warm.dna_snapshot_id == cold.dna_snapshot_id
    assert warm.selected_skill_ids == cold.selected_skill_ids
    assert warm.to_dict()['semantic_channels'] == cold.to_dict()['semantic_channels']
    assert budget.calls == calls and ledger.events[-1]['real_semantic_calls'] == 0


def test_baseline_schema_resolution_is_scoped_to_each_load(prepared, monkeypatch):
    _, request, cold, _, _ = prepared
    from skillforge.productization import evolution_artifacts as artifacts
    original = artifacts.get_type_hints
    counts = Counter()
    def hints(kind, **kwargs):
        counts[kind] += 1
        return original(kind, **kwargs)
    monkeypatch.setattr(artifacts, 'get_type_hints', hints)
    one = artifacts.load_baseline(request.path, cold.evolution_baseline_path)
    assert counts and all(count == 1 for count in counts.values())
    before = dict(counts)
    two = artifacts.load_baseline(request.path, cold.evolution_baseline_path)
    assert one == two and all(counts[kind] == count * 2 for kind, count in before.items())


def test_forged_snapshot_manifest_cannot_bypass_static_gate(prepared):
    preparer, request, _, budget, ledger = prepared
    file = request.path / 'src/app/api/route0.py'
    file.write_text(file.read_text('utf-8') + '\nimport app.service.extra\n', encoding='utf-8')
    index = request.path / '.skillforge/productization/prepare/verified-baseline.json'
    entry = json.loads(index.read_text('utf-8'))
    entry['payload']['identity']['context'] = RepoSnapshotSemanticCache(request.path).context
    entry['checksum'] = digest(entry['payload'])
    index.write_text(json.dumps(entry), encoding='utf-8')
    calls = budget.calls
    with pytest.raises(SkillForgeError, match='static facts differ'):
        automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert budget.calls == calls


def test_baseline_change_during_routing_is_rejected(prepared, monkeypatch):
    preparer, request, cold, budget, ledger = prepared
    from skillforge.productization import prepare as inner
    original = inner.SkillRouter.route
    path = request.path / cold.evolution_baseline_path
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text(path.read_text('utf-8') + ' ', encoding='utf-8')
        return result
    monkeypatch.setattr(inner.SkillRouter, 'route', mutate)
    calls = budget.calls
    with pytest.raises(BudgetExceeded, match='changed during task preparation'):
        automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert budget.calls == calls
