"""Recovered paid responses remain exact and cannot silently spend on drift."""
import copy
import sys
from pathlib import Path

import pytest
from test_cold_semantic_batching import twelve_inputs, eleven_inputs
from test_prepare import historical_repository, repository, FakeModel
from test_repo_snapshot_semantic_cache import complete_identity
from test_semantic_batch_transport import wire_response

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from recovered_semantics import PaidResponseRecovery
from skillforge.cost_control import CostLedger, BudgetExceeded
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_structural_rules import build_semantic_structural_request
from skillforge.repository.semantic_batch import _adapter, request_id


def fixture_recovery(inputs):
    root, *_, mined = inputs
    requests = tuple(build_semantic_structural_request(mined, c) for c in mined.candidates)
    assert len(requests) == 2
    recovery = PaidResponseRecovery.__new__(PaidResponseRecovery)
    recovery.workspace, recovery.context = root, RepoSnapshotSemanticCache(root).context
    identity = complete_identity()
    recovery.freeze = {"model_identities": [identity]}
    recovery.seal, recovery.recovered, recovery.seen = {"mode": "FIXTURE_NO_MODEL"}, set(), set()
    model = CodexCliStructuralSemanticModel(runner=lambda *a, **k: pytest.fail("unexpected subprocess"))
    model.semantic_cache_identity = copy.deepcopy(identity)
    _, prompt, _, _ = _adapter(model, requests[0])
    recovery.requests = {request_id(r): {"channel": type(model).__name__, "original_prompt": prompt(r)} for r in requests}
    first = requests[0]
    recovery.raw = {type(model).__name__: {"results": [{"request_id": request_id(first),
        "repository_id": first.repository_id, "candidate_id": first.candidate_id,
        "response": wire_response(FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE).complete(first)), "error": None}]}}
    recovery._sealed_raw = copy.deepcopy(recovery.raw)
    recovery.raw_hashes = {type(model).__name__: "a" * 64}
    return recovery, model, requests


def test_recovery_calls_only_exact_missing_candidate(twelve_inputs):
    recovered, model, requests = fixture_recovery(twelve_inputs)
    dispatched = []
    model.complete_batch = lambda rs: dispatched.extend(rs) or {request_id(r): FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE).complete(r) for r in rs}
    ledger = CostLedger()
    recovered.attach(model, live=True, ledger=ledger)
    results = model.complete_batch(requests)
    assert dispatched == [requests[1]]
    assert len(results) == 2 and len(recovered.recovered) == 1
    assert ledger.events[0]["cache"] == "PAID_RESPONSE_RECOVERY_NOT_CACHE_HIT"
    assert model.semantic_cache_identity["response_parameters"]["paid_response_recovery"] == recovered.seal


@pytest.mark.parametrize("drift", ["prompt", "response", "snapshot", "identity"])
def test_recovery_drift_aborts_before_transport(twelve_inputs, drift):
    recovered, model, requests = fixture_recovery(twelve_inputs)
    model.complete_batch = lambda rs: pytest.fail("drift reached transport")
    if drift == "identity":
        model.semantic_cache_identity["model_selector"] = "different"
        with pytest.raises(BudgetExceeded):
            recovered.attach(model, live=True, ledger=CostLedger())
        return
    recovered.attach(model, live=True, ledger=CostLedger())
    if drift == "prompt":
        recovered.requests[request_id(requests[0])]["original_prompt"] += "hint"
    elif drift == "response":
        recovered.raw[type(model).__name__]["results"][0]["error"] = "tampered"
    else:
        (recovered.workspace / "new-source.py").write_text("changed = 1")
    with pytest.raises(BudgetExceeded):
        model.complete_batch(requests)


def test_offline_recovery_has_no_missing_fallback(twelve_inputs):
    recovered, model, requests = fixture_recovery(twelve_inputs)
    recovered.attach(model, live=False, ledger=CostLedger())
    assert len(model.complete_batch(requests)) == 1
    with pytest.raises(RuntimeError, match="OFFLINE_RECOVERY_MISSING_RESPONSE"):
        model.complete(requests[1])
