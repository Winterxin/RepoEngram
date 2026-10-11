"""Exact reuse and conservative invalidation; no live Semantic or Agent calls."""

import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_prepare import AntiPatternModel, FakeModel, WorkflowModel
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.repo_snapshot_semantic_cache import (
    CUSTOM_TRANSPORT_IDENTITY_SCHEMA,
    RepoSnapshotSemanticCache,
)
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


def complete_identity(model="offline-fixture-v1", decision="accept"):
    def digest(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return {
        "schema": CUSTOM_TRANSPORT_IDENTITY_SCHEMA,
        "model_selector": model,
        "executable": {"path": "offline-fixture", "sha256": digest("fixture executable")},
        "transport": {"name": "offline-fixture", "source_sha256": digest(
            inspect.getsource(FakeModel.complete) + inspect.getsource(RecordingModel.complete),
        )},
        "cli_config": {"argv_sha256": digest("fixture argv"),
                       "environment_sha256": digest("fixture environment"),
                       "config_policy": "isolated fixture: no external config"},
        "timeout_seconds": 180,
        "permission_wrapper": {"version": "offline-fixture-v1",
                               "config_sha256": digest("no subprocess or network")},
        "response_parameters": {"decision": decision, "invalid": False, "error": False},
    }


class RecordingModel(FakeModel):
    def __init__(self, invalid=False, error=False):
        super().__init__(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
        self.calls = 0
        self.invalid, self.error = invalid, error
        self.semantic_cache_identity = complete_identity()
        self.semantic_cache_identity["response_parameters"].update(invalid=invalid, error=error)

    def complete(self, request):
        self.calls += 1
        if self.error:
            raise RuntimeError("timeout")
        response = super().complete(request)
        return replace(response, decision_evidence_ids=("unknown",)) if self.invalid else response


def evaluate(root, directory, model):
    cache = RepoSnapshotSemanticCache(root, directory)
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(root))
    result = SemanticStructuralRuleEvaluator().evaluate(mined, cache.wrap(model))
    cache.publish(result)
    return cache, result


def test_persistent_exact_reuse(historical_repository, tmp_path):
    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    cold_model, warm_model = RecordingModel(), RecordingModel()
    cold, result = evaluate(historical_repository, directory, cold_model)
    warm, reused = evaluate(historical_repository, directory, warm_model)
    assert cold_model.calls == cold.misses > 0
    assert warm_model.calls == 0 and warm.hits == cold.misses
    assert reused == result
    assert len(list(directory.glob("*.json"))) == cold.misses


@pytest.mark.parametrize("field", (
    "model_selector", "executable.path", "executable.sha256", "transport.name",
    "transport.source_sha256", "cli_config.argv_sha256", "cli_config.environment_sha256",
    "cli_config.config_policy", "timeout_seconds", "permission_wrapper.version",
    "permission_wrapper.config_sha256", "response_parameters.decision",
))
def test_each_custom_transport_field_change_misses(historical_repository, tmp_path, field):
    directory = tmp_path.parent / (tmp_path.name + "-cache")
    evaluate(historical_repository, directory, RecordingModel())
    model = RecordingModel()
    model.semantic_cache_identity = deepcopy(complete_identity())
    parent = model.semantic_cache_identity
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    name = parts[-1]
    old = parent[name]
    parent[name] = old + 1 if isinstance(old, int) else (
        "0" * 64 if name.endswith("sha256") else old + "-changed"
    )
    cache, result = evaluate(historical_repository, directory, model)
    assert model.calls > 0 and cache.hits == 0 and not result.failures


@pytest.mark.parametrize("field", tuple(complete_identity()))
def test_missing_custom_identity_field_bypasses(historical_repository, tmp_path, field):
    cache = RepoSnapshotSemanticCache(historical_repository, tmp_path.parent / "unused-cache")
    model = RecordingModel()
    model.semantic_cache_identity = complete_identity()
    del model.semantic_cache_identity[field]
    assert cache.wrap(model) is model
    assert cache.hits == cache.misses == 0


@pytest.mark.parametrize("identity", (None, {}, {"model": "label-only"}))
def test_absent_or_label_only_identity_bypasses(historical_repository, tmp_path, identity):
    cache = RepoSnapshotSemanticCache(historical_repository, tmp_path.parent / "unused-cache")
    model = RecordingModel()
    model.semantic_cache_identity = identity
    assert cache.wrap(model) is model


def test_actual_cli_adapter_custom_runner_reuses(historical_repository, tmp_path, monkeypatch):
    import subprocess
    import tempfile
    from pathlib import Path

    from skillforge.domain.serialization import to_json_value
    from skillforge.repository.codex_cli_structural import (
        CODEX_STRUCTURAL_MODEL,
        CodexCliStructuralSemanticModel,
    )

    temporary = tmp_path.parent / (tmp_path.name + "-transport")
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(historical_repository))
    request = build_semantic_structural_request(mined, mined.candidates[0])
    raw = to_json_value(
        FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE).complete(request),
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_text(json.dumps(raw), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    directory = tmp_path.parent / (tmp_path.name + "-cache")
    cold_model = CodexCliStructuralSemanticModel(runner=runner)
    cold_model.semantic_cache_identity = complete_identity(model=CODEX_STRUCTURAL_MODEL)
    cold_model.semantic_cache_identity["transport"]["source_sha256"] = hashlib.sha256(
        inspect.getsource(runner).encode("utf-8"),
    ).hexdigest()
    response_hash = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode("utf-8"),
    ).hexdigest()
    cold_model.semantic_cache_identity["response_parameters"]["raw_response_sha256"] = response_hash
    cold_cache, cold = evaluate(historical_repository, directory, cold_model)
    count = len(calls)
    warm_model = CodexCliStructuralSemanticModel(runner=runner)
    warm_model.semantic_cache_identity = deepcopy(cold_model.semantic_cache_identity)
    warm_cache, warm = evaluate(historical_repository, directory, warm_model)
    assert count == cold_cache.misses > 0 and len(calls) == count
    assert warm_cache.hits == count and warm_cache.misses == 0 and warm == cold
    mismatch = CodexCliStructuralSemanticModel(runner=runner, timeout=181)
    mismatch.semantic_cache_identity = cold_model.semantic_cache_identity
    assert warm_cache.wrap(mismatch) is mismatch


@pytest.mark.parametrize("failure", ("invalid", "timeout"))
def test_failures_never_published(historical_repository, tmp_path, failure):
    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    model = RecordingModel(invalid=failure == "invalid", error=failure == "timeout")
    _, result = evaluate(historical_repository, directory, model)
    assert result.failures and not list(directory.glob("*.json"))
    first = model.calls
    evaluate(historical_repository, directory, model)
    assert model.calls == first * 2


@pytest.mark.parametrize("change", ("snapshot", "config", "production", "schema",
                                    "candidate", "evaluator", "repository"))
def test_every_identity_dimension_invalidates(historical_repository, tmp_path, change,
                                             monkeypatch):
    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    evaluate(historical_repository, directory, RecordingModel())
    cache = RepoSnapshotSemanticCache(historical_repository, directory)
    model = RecordingModel()
    if change == "snapshot":
        path = historical_repository / "new-source.txt"
        path.write_text("new snapshot", encoding="utf-8")
        cache = RepoSnapshotSemanticCache(historical_repository, directory)
    elif change == "config":
        model.semantic_cache_identity = complete_identity(model="different")
    elif change == "production":
        cache.context["production_semantic_version"] = "different"
    elif change == "schema":
        monkeypatch.setattr("skillforge.repository.repo_snapshot_semantic_cache.CACHE_VERSION",
                            "different")
    elif change == "repository":
        cache.context["repository_id"] = "different"
    elif change == "evaluator":
        class OtherModel(RecordingModel):
            pass
        model = OtherModel()
    wrapped = cache.wrap(model)
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(historical_repository))
    if change == "candidate":
        request = build_semantic_structural_request(mined, mined.candidates[0])
        wrapped.complete(replace(
            request, canonical_observation=request.canonical_observation + "x",
        ))
        assert model.calls > 0 and cache.hits == 0
        return
    result = SemanticStructuralRuleEvaluator().evaluate(mined, wrapped)
    assert not result.failures and model.calls > 0 and cache.hits == 0


@pytest.mark.parametrize("corruption", ("truncated", "hash", "identity", "fields"))
def test_corrupt_cache_uses_original_path(historical_repository, tmp_path, corruption):
    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    evaluate(historical_repository, directory, RecordingModel())
    for path in directory.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "hash":
            entry["response_hash"] = "invalid"
        elif corruption == "identity":
            entry["identity"]["repository_id"] = "invalid"
        elif corruption == "fields":
            entry["unexpected"] = True
        path.write_text("{" if corruption == "truncated" else json.dumps(entry), encoding="utf-8")
    model = RecordingModel()
    cache, result = evaluate(historical_repository, directory, model)
    assert model.calls > 0 and cache.hits == 0 and not result.failures


def test_opaque_models_and_unvalidated_responses_are_not_cached(historical_repository, tmp_path):
    cache = RepoSnapshotSemanticCache(historical_repository, tmp_path / "cache")
    model = FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    assert cache.wrap(model) is model
    tracked = RecordingModel()
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(historical_repository))
    SemanticStructuralRuleEvaluator().evaluate(mined, cache.wrap(tracked))
    assert cache.pending and not list(cache.directory.glob("*.json"))


def test_snapshot_drift_during_evaluation_is_not_published(historical_repository, tmp_path):
    cache = RepoSnapshotSemanticCache(
        historical_repository, tmp_path.parent / (tmp_path.name + "-cache"),
    )
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(historical_repository))
    result = SemanticStructuralRuleEvaluator().evaluate(mined, cache.wrap(RecordingModel()))
    (historical_repository / "changed.txt").write_text("changed", encoding="utf-8")
    cache.publish(result)
    assert not cache.pending and not list(cache.directory.glob("*.json"))


def test_gate_valid_rejections_are_reused(historical_repository, tmp_path):
    directory = tmp_path.parent / (tmp_path.name + "-cache")
    models = [RecordingModel(), RecordingModel()]
    for model in models:
        model.decision = SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE
        model.semantic_cache_identity = complete_identity(decision="reject")
    _, cold = evaluate(historical_repository, directory, models[0])
    cache, warm = evaluate(historical_repository, directory, models[1])
    assert cold.rejections and cold == warm and cache.hits > 0 and models[1].calls == 0


def test_optional_tendency_scoped_response_roundtrips_exactly():
    tendency = pytest.importorskip("skillforge.domain.semantic_tendency")
    from skillforge.domain.serialization import to_json_value
    from skillforge.repository.repo_snapshot_semantic_cache import _restore_response

    original = tendency.ScopedTendencyDecision(
        "exact-request-fingerprint",
        tendency.TendencyTransportDecision(
            tendency.TendencyDecisionKind.INSUFFICIENT_EVIDENCE,
            "The supplied observations are insufficient.", ("evidence",),
        ),
    )
    assert _restore_response(type(original), to_json_value(original)) == original


def test_preparation_reuses_after_task_output_removal(historical_repository, tmp_path):
    import shutil

    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    model = RecordingModel()
    first = SkillForgePreparer(model, WorkflowModel(), AntiPatternModel(),
                              semantic_cache_root=directory).prepare(PrepareRequest(
        historical_repository, "update app.api", CodeAgentTarget.CODEX,
    ))
    baseline = load_baseline(historical_repository, first.evolution_baseline_path)
    # Delete only this test's generated local outputs, preserving the legal cache.
    for name in (".skillforge", ".codex"):
        shutil.rmtree(historical_repository / name)
    second_model = RecordingModel()
    second = SkillForgePreparer(second_model, WorkflowModel(), AntiPatternModel(),
                               semantic_cache_root=directory).prepare(PrepareRequest(
        historical_repository, "another app.api task", CodeAgentTarget.CODEX,
    ))
    warm_baseline = load_baseline(historical_repository, second.evolution_baseline_path)
    assert model.calls > 0
    assert warm_baseline.source_dna == baseline.source_dna
    assert warm_baseline.source_compilation == baseline.source_compilation
    assert second.selected_skill_ids == first.selected_skill_ids
    assert second_model.calls == 0
