"""Offline integration: paid decisions seed only validated responses, then resume."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_cold_semantic_batching import SingleFixture, run
from test_cold_semantic_batching import eleven_inputs as eleven_inputs_fixture
from test_cold_semantic_batching import twelve_inputs as twelve_inputs_fixture
from test_effect_cost_control import actual_models
from test_prepare import historical_repository as historical_repository_fixture
from test_prepare import repository as repository_fixture
from test_semantic_batch_transport import recorded_models

EVALS = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from recovered_semantics import PaidResponseRecovery

from skillforge.cost_control import BudgetExceeded, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.verified_prepare import _INDEX, VerifiedPreparer
from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_batch import _adapter, cli_batch_payload, request_id

CHANNELS = (CodexCliStructuralSemanticModel, CodexCliSemanticModel,
            CodexCliAntiPatternSemanticModel)


def paid_fixture(saved, workspace, out):
    fixture, runner, saved_adapters = saved
    paid = PaidResponseRecovery.__new__(PaidResponseRecovery)
    paid.workspace = workspace
    paid.out = out
    paid.context = RepoSnapshotSemanticCache(workspace).context
    paid.freeze = {"model_identities": [copy.deepcopy(model.semantic_cache_identity)
                                         for model in saved_adapters]}
    paid.seal = {"mode": "FIXTURE_PAID_RESPONSE_RECOVERY", "original_wire_attestation": False}
    paid.raw = {}
    paid.raw_hashes = {}
    paid.requests = {}
    paid.recovered = set()
    paid.seen = set()
    paid._sealed_raw = None

    # Models in the recording fixture have the original 2/5/5 requests. Build
    # exact, request-keyed wire envelopes and omit exactly one workflow result.
    for index, (model_type, single_model, adapter) in enumerate(
            zip(CHANNELS, fixture[2], saved_adapters, strict=True)):
        name = model_type.__name__
        _, prompt_builder, _, _ = _adapter(adapter, single_model.requests[0])
        envelopes = []
        for request in single_model.requests:
            prompt = prompt_builder(request)
            recorded_request, raw_response = runner.records[prompt]
            assert recorded_request == request
            key = request_id(request)
            paid.requests[key] = {"channel": name, "original_prompt": prompt}
            envelopes.append({
                "request_id": key,
                "repository_id": request.repository_id,
                "candidate_id": request.candidate_id,
                "response": raw_response,
                "error": None,
            })
        if index == 1:
            paid.missing_request_id = envelopes.pop(0)["request_id"]
        paid.raw[name] = {"results": envelopes}
        payload = json.dumps(paid.raw[name], sort_keys=True).encode("utf-8")
        paid.raw_hashes[name] = hashlib.sha256(payload).hexdigest()
    assert sum(len(raw["results"]) for raw in paid.raw.values()) == 11
    paid._sealed_raw = copy.deepcopy(paid.raw)
    return paid


def watch_transport(transport):
    calls = []
    original = transport.runner

    def runner(command, **kwargs):
        calls.append(kwargs["input"])
        return original(command, **kwargs)

    transport.runner = runner
    return calls


@pytest.fixture
def twelve_inputs(tmp_path):
    root = repository_fixture.__wrapped__(tmp_path)
    historical = historical_repository_fixture.__wrapped__(root)
    eleven = eleven_inputs_fixture.__wrapped__(historical)
    return twelve_inputs_fixture.__wrapped__(eleven)


def test_paid_checkpoint_then_one_candidate_prepare_and_warm_reuse(
        twelve_inputs, tmp_path):
    root = twelve_inputs[0]
    cache_root = tmp_path.parent / (tmp_path.name + "-paid-semantic-cache")
    output = tmp_path.parent / (tmp_path.name + "-paid-recovery-proof")

    saved = recorded_models(twelve_inputs)
    paid = paid_fixture(saved, root, output)

    seed_models, seed_transport, seed_budget, seed_ledger = actual_models(
        twelve_inputs, tmp_path, cold_shape={"workflow": 5, "tendency": 0,
                                            "anti-pattern": 5, "structural": 2},
    )
    seed_transport_calls = watch_transport(seed_transport)
    paid.freeze = {"model_identities": [copy.deepcopy(model.semantic_cache_identity)
                                         for model in seed_models]}
    for model in seed_models:
        paid.attach(model, live=False, ledger=seed_ledger)
    seed = paid.seed_validated_cache(
        seed_models, seed_budget, seed_ledger, cache_root=cache_root,
    )
    assert seed["status"] == "PASS"
    assert seed["known_candidates"] == 11
    assert seed["real_semantic_calls"] == seed_budget.calls == 0
    assert seed["baseline_publications"] == 0
    assert not (root / _INDEX).exists()
    assert len(list(cache_root.glob("*.json"))) == 11
    assert not [event for event in seed_ledger.events if event["kind"] == "semantic"]
    assert len([event for event in seed_ledger.events
                if event["kind"] == "semantic-recovery"]) == 11
    assert seed_transport_calls == []

    models, transport, budget, ledger = actual_models(
        twelve_inputs, tmp_path,
        cold_shape={"workflow": 5, "tendency": 0, "anti-pattern": 5, "structural": 2},
    )
    live_prompts = watch_transport(transport)
    for model in models:
        paid.attach(model, live=True, ledger=ledger)
    request = PrepareRequest(root, "update routes", CodeAgentTarget.CODEX, "app.api.route0")
    preparer = SkillForgePreparer(*models, semantic_cache_root=cache_root)
    controlled = VerifiedPreparer(preparer, models, config={"recovery_fixture": True},
                                  budget=budget, ledger=ledger)
    cold = controlled.prepare(request)
    baseline = load_baseline(root, cold.evolution_baseline_path)
    expected, artifacts, _, _ = run(twelve_inputs, SingleFixture)
    assert baseline.source_dna == artifacts[0]
    assert baseline.source_compilation == artifacts[1]
    assert baseline.semantic_structural == expected.structural
    assert cold.semantic_failure_count == 0
    assert paid.missing_request_id in paid.requests
    assert budget.calls == 1
    semantic_calls = [event for event in ledger.events if event["kind"] == "semantic"]
    assert len(semantic_calls) == 1
    assert semantic_calls[0]["real_call"] == 1
    assert len(live_prompts) == 1
    missing_request = next(request for port in saved[0][2] for request in port.requests
                           if request_id(request) == paid.missing_request_id)
    missing_adapter = next(model for model in models if type(model).__name__ ==
                           paid.requests[paid.missing_request_id]["channel"])
    _, expected_batch, _, _, _ = cli_batch_payload(missing_adapter, (missing_request,))
    # Both valid wire forms carry only the exact missing original request. Do
    # not assume legacy SEMANTIC_BATCH when compact input is cheaper.
    assert live_prompts[0] in {
        expected_batch, paid.requests[paid.missing_request_id]["original_prompt"]}
    assert len([event for event in ledger.events if event["kind"] == "semantic-cache"
                and event["cache"] == "hit"]) == 11

    warm_budget = SemanticBudget(max_calls=0,
                                 cold_shape={"workflow": 5, "tendency": 0,
                                             "anti-pattern": 5, "structural": 2})
    transport.budget = warm_budget
    for model in models:
        model.effect_budget = warm_budget
        model._runner.budget = warm_budget
    warm = VerifiedPreparer(preparer, models, config=controlled.config,
                            budget=warm_budget, ledger=ledger).prepare(request)
    assert warm_budget.calls == 0
    assert warm.semantic_failures == cold.semantic_failures
    assert warm.semantic_failure_count == cold.semantic_failure_count
    assert warm.to_dict()["semantic_channels"] == cold.to_dict()["semantic_channels"]
    assert load_baseline(root, warm.evolution_baseline_path) == baseline


def test_paid_checkpoint_persistence_failure_aborts_without_transport(
        twelve_inputs, tmp_path, monkeypatch):
    from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache

    root = twelve_inputs[0]
    cache_root = tmp_path.parent / (tmp_path.name + "-failed-semantic-cache")
    saved = recorded_models(twelve_inputs)
    paid = paid_fixture(saved, root,
                        tmp_path.parent / (tmp_path.name + "-failed-recovery-proof"))
    models, transport, budget, ledger = actual_models(
        twelve_inputs, tmp_path,
        cold_shape={"workflow": 5, "tendency": 0, "anti-pattern": 5, "structural": 2},
    )
    paid.freeze = {"model_identities": [copy.deepcopy(model.semantic_cache_identity)
                                         for model in models]}
    live_prompts = watch_transport(transport)
    for model in models:
        paid.attach(model, live=False, ledger=ledger)
    monkeypatch.setattr(RepoSnapshotSemanticCache, "_persist", lambda self, entries: None)
    with pytest.raises(BudgetExceeded, match="durably saved"):
        paid.seed_validated_cache(models, budget, ledger, cache_root=cache_root)
    assert budget.calls == 0
    assert live_prompts == []
    assert not [event for event in ledger.events if event["kind"] == "semantic"]
    assert not list(cache_root.glob("*.json"))
