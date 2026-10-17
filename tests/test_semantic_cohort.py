"""Offline original-gate/artifact equality, cold/warm and fatal cohort budgets."""
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_semantic_batch_transport import recorded_models

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
from skillforge.productization.verified_prepare import automatic_prepare
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.semantic_cohort import COHORT_VERSION, prefetch_semantic_cohort
from skillforge.repository.semantic_wire import WIRE_VERSION
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def enable(models):
    for model in models:
        model.semantic_cache_identity["response_parameters"].update(
            semantic_wire=WIRE_VERSION, semantic_cohort=COHORT_VERSION)


def assemble(inputs, models, directory=None):
    root, facts, history, candidates, negative, selected, anti, mined = inputs
    cache = RepoSnapshotSemanticCache(root, directory) if directory else None
    structural, workflow, anti_pattern = tuple(cache.wrap(m) for m in models) if cache else models
    result = evaluate_semantic_knowledge(facts.repository_id, facts, mined, structural,
        history=history, candidates=candidates, selected_workflows=selected,
        negative=negative, selected_anti_patterns=anti,
        workflow_model=workflow, anti_pattern_model=anti_pattern)
    return result, cache


def artifacts(inputs, result):
    dna = ProjectDNABuilder().build(inputs[1].repository_id, workflows=result.workflows,
        anti_patterns=result.anti_patterns, semantic_structural_rules=result.structural,
        semantic_tendencies=result.tendencies)
    compiled = SkillCompiler().compile(dna, result.structural, result.tendencies)
    registry = SkillRegistryBuilder().build(compiled)
    routed = SkillRouter().route(registry, SkillRoutingRequest(inputs[1].repository_id,
        "update routes", current_module="app.api.route0"))
    return dna, compiled, registry, routed


def test_cold_one_transport_and_exact_zero_transport_cache(eleven_inputs, tmp_path):
    expected, runner, models = recorded_models(eleven_inputs)
    enable(models)
    directory = tmp_path.parent / (tmp_path.name + "-cohort-cache")
    cold, cache = assemble(eleven_inputs, models, directory)
    assert len(runner.calls) == 1
    assert not runner.fallback_ids
    assert (cold.workflows, cold.anti_patterns, cold.structural) == (
        expected[0].workflows, expected[0].anti_patterns, expected[0].structural)
    assert not cold.failures
    assert cache.misses == 11
    assert len(list(directory.glob("*.json"))) == 11
    warm, warm_cache = assemble(eleven_inputs, models, directory)
    assert len(runner.calls) == 1
    assert warm_cache.hits == 11
    assert artifacts(eleven_inputs, cold) == artifacts(eleven_inputs, warm)
    assert artifacts(eleven_inputs, cold)[:3] == expected[1][:3]


def test_real_prepare_entry_cold_baseline_and_warm_provenance(eleven_inputs, tmp_path):
    _, runner, models = recorded_models(eleven_inputs)
    enable(models)
    root = eleven_inputs[0]
    request = PrepareRequest(root, "update routes", CodeAgentTarget.CODEX)
    preparer = SkillForgePreparer(*models,
        semantic_cache_root=tmp_path.parent / (tmp_path.name + "-prepare-cache"))
    ledger = CostLedger()
    cold = automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert len(runner.calls) == 1 and cold.semantic_failure_count == 0
    baseline = root / cold.evolution_baseline_path
    original_bytes = baseline.read_bytes()
    warm = automatic_prepare(request, preparer=preparer, ledger=ledger)
    assert len(runner.calls) == 1
    assert baseline.read_bytes() == original_bytes
    for key in ("selected_skill_ids", "workflow_accepted_count", "workflow_rejection_count",
                "anti_pattern_accepted_count", "anti_pattern_rejection_count",
                "semantic_failure_count", "semantic_failures"):
        assert getattr(cold, key) == getattr(warm, key)
    assert ledger.events[-1]["cache"] == "hit"


def test_normal_native_adapters_enable_cohort_without_eval_opt_in(eleven_inputs, monkeypatch):
    expected, runner, prototypes = recorded_models(eleven_inputs)
    monkeypatch.setattr(subprocess, "run", runner)
    models = tuple(type(model)(runner=subprocess.run, sandbox_mode="read-only")
                   for model in prototypes)
    assert all(not hasattr(model, "semantic_cache_identity") for model in models)
    actual, _ = assemble(eleven_inputs, models)
    assert len(runner.calls) == 1
    assert artifacts(eleven_inputs, actual)[:3] == expected[1][:3]


@pytest.mark.parametrize("mismatch", ["opt_in", "identity", "runner", "timeout", "permission"])
def test_incompatible_transports_keep_original_channel_batches(eleven_inputs, mismatch):
    expected, runner, models = recorded_models(eleven_inputs)
    enable(models)
    changed = models[0]
    if mismatch == "opt_in":
        changed.semantic_cache_identity["response_parameters"].pop("semantic_cohort")
    elif mismatch == "identity":
        changed.semantic_cache_identity = deepcopy(changed.semantic_cache_identity)
        changed.semantic_cache_identity["response_parameters"]["different"] = True
    elif mismatch == "runner":
        changed._runner = lambda *args, **kwargs: runner(*args, **kwargs)
    elif mismatch == "timeout":
        changed._timeout = 179
        changed.semantic_cache_identity["timeout_seconds"] = 179
    else:
        changed._sandbox_mode = "workspace-write"
        assert prefetch_semantic_cohort({
            "structural": (models[0], expected[2][0].requests),
            "workflow": (models[1], expected[2][1].requests),
            "anti-pattern": (models[2], expected[2][2].requests),
        }) is None
        assert not runner.calls
        return
    actual, _ = assemble(eleven_inputs, models)
    assert len(runner.calls) == 3
    assert artifacts(eleven_inputs, actual)[:3] == expected[1][:3]


@pytest.mark.parametrize("mode", [
    "partial", "duplicate", "candidate_error", "malformed", "gate_failure",
])
def test_bad_candidate_isolated_and_other_channels_preserved(eleven_inputs, mode):
    expected, runner, models = recorded_models(eleven_inputs, mode)
    enable(models)
    actual, _ = assemble(eleven_inputs, models)
    assert len(runner.calls) == 2  # One cohort + the same original one-candidate repair.
    assert len(runner.fallback_ids) == 1
    assert actual.structural == expected[0].structural
    assert actual.anti_patterns == expected[0].anti_patterns
    assert len(actual.workflows.workflows) == (4 if mode in {"gate_failure", "malformed"} else 5)


@pytest.mark.parametrize("mode,max_calls", [("gate_failure", 1), ("batch_failure", 5)])
def test_fatal_budget_before_fallback_spawn(eleven_inputs, tmp_path, mode, max_calls):
    _, runner, models = recorded_models(eleven_inputs, mode)
    enable(models)
    budget = SemanticBudget(max_calls=max_calls)
    def governed(command, **kwargs):
        budget.reserve()
        return runner(command, **kwargs)
    for model in models:
        model._runner = governed
        model.effect_budget = budget
    directory = tmp_path.parent / (tmp_path.name + "-budget-cache")
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        assemble(eleven_inputs, models, directory)
    assert budget.calls == len(runner.calls) == max_calls
    if mode == "gate_failure":
        assert len(list(directory.glob("*.json"))) == 10


@pytest.mark.parametrize("repair_valid", [True, False])
@pytest.mark.parametrize("channels", ["cross", "anti-only"])
def test_multiple_invalid_candidates_share_one_repair_with_original_gates(
    eleven_inputs, tmp_path, repair_valid, channels,
):
    from skillforge.repository.semantic_batch import candidate_identity
    from skillforge.repository.semantic_context_catalog import ExactContextCatalog

    expected, original, models = recorded_models(eleven_inputs)
    enable(models)
    records = {candidate_identity(request)[1]: (request, raw)
               for request, raw in original.records.values()}
    bad = [candidate_identity(request)[1]
           for request, raw in original.records.values() if "workflow" in raw][:2]
    bad += [candidate_identity(request)[1]
            for request, raw in original.records.values() if "anti_pattern" in raw][:2]
    assert len(bad) == 4
    if channels == "anti-only":
        bad = bad[2:]
    calls = []
    budget = SemanticBudget(max_calls=2)

    def runner(command, **kwargs):
        from skillforge.repository.semantic_wire import CandidateAliases

        budget.reserve()
        data = json.loads(kwargs["input"].split("SEMANTIC_COMPACT:\n", 1)[1])
        entries = data["candidates"]
        concrete = {}
        for entry in entries:
            context = ExactContextCatalog.expand(
                entry["CANDIDATE_CONTEXT"], data.get("EXACT_SHARED_CONTEXT", {}))
            without_feedback = {k: v for k, v in context.items() if k != "validation_feedback"}
            matches = [key for key, (request, _) in records.items()
                       if key == entry["candidate_id"]
                       or CandidateAliases(entry["request_id"], json.loads(json.dumps(
                           request.to_dict(), sort_keys=True))).encode(
                               request.to_dict()) == without_feedback]
            assert len(matches) == 1
            concrete[entry["request_id"]] = matches[0]
        calls.append(list(concrete.values()))
        # Teach the sealed fixture runner the bounded diagnostic field only;
        # preserve every original source fact and the expected raw response.
        updated = {}
        for entry in entries:
            request, raw = records[concrete[entry["request_id"]]]
            context = ExactContextCatalog.expand(
                entry["CANDIDATE_CONTEXT"], data.get("EXACT_SHARED_CONTEXT", {}))
            if context.get("validation_feedback"):
                from dataclasses import replace
                request = replace(request, validation_feedback=context["validation_feedback"])
            from skillforge.repository.semantic_batch import _adapter
            adapter = next(model for model in models
                           if ("anti_pattern" in raw and "AntiPattern" in type(model).__name__)
                           or ("workflow" in raw
                               and type(model).__name__ == "CodexCliSemanticModel")
                           or ("scope_modules" in raw and "Structural" in type(model).__name__))
            updated[_adapter(adapter, request)[1](request)] = (request, raw)
        original.records = updated
        result = original(command, **kwargs)
        path = Path(command[command.index("-o") + 1])
        output = json.loads(path.read_text("utf-8"))
        for response in output["results"]:
            candidate_id = concrete[response["request_id"]]
            if candidate_id in bad and (len(calls) == 1 or not repair_valid):
                response["response"]["reason"] = (
                    "The absent.module relation establishes this decision.")
        path.write_text(json.dumps(output), encoding="utf-8")
        return result

    for model in models:
        model._runner = runner
        model.effect_budget = budget
    directory = tmp_path.parent / (tmp_path.name + "-shared-repair-cache")
    actual, cache = assemble(eleven_inputs, models, directory)
    assert budget.calls == len(calls) == 2
    assert set(calls[1]) == set(bad)
    assert actual.structural == expected[0].structural
    assert actual.workflow_retry_count == (2 if channels == "cross" else 0)
    assert actual.anti_pattern_retry_count == 2
    if repair_valid:
        assert not actual.failures
        assert (actual.workflows, actual.anti_patterns) == (
            expected[0].workflows, expected[0].anti_patterns)
        warm, warm_cache = assemble(eleven_inputs, models, directory)
        assert len(calls) == 2 and warm_cache.hits == 11
        assert not warm.failures
    else:
        assert len(actual.failures) == len(bad)
        assert all(len(item["attempts"]) == 2 for item in actual.failures)
        assert len(list(directory.glob("*.json"))) == 11 - len(bad)
        assert cache.pending
