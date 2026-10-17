"""Exercise the real CLI adapters with a runner that can only return fixture JSON."""

import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_cold_semantic_batching import SingleFixture, run
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import FakeModel
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import complete_identity

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.domain.serialization import to_json_value
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.repository.semantic_batch import (
    _adapter,
    candidate_identity,
    parse_cli_batch,
    prefetch_semantic_batch,
    request_id,
    supports_semantic_batch,
)
from skillforge.repository.semantic_tendencies import build_semantic_tendency_request


def wire_response(response):
    raw = to_json_value(response)
    if "workflow" in raw and raw["workflow"]:
        for key in ("repository_id", "candidate_id", "confidence_score"):
            raw["workflow"].pop(key)
    if raw.get("supporting_fact_ids", False) is None:
        raw.pop("supporting_fact_ids")
    raw.pop("audit_warnings", None)
    return raw


class RecordedFixtureRunner:
    """No subprocess/network fallback exists, including on unmatched prompts."""

    def __init__(self, records, mode="success", target_channel="workflow"):
        self.records, self.mode, self.target_channel = records, mode, target_channel
        self.calls = []
        self.fallback_ids = []
        self.target = None

    def corrupt(self, response):
        raw = deepcopy(response)
        if self.mode == "malformed":
            return {"bad": "response"}
        if self.mode == "gate_failure":
            raw["decision_evidence_ids"] = ["unknown"]
        return raw

    def __call__(self, command, **kwargs):
        prompt = kwargs["input"]
        assert "--ignore-user-config" in command and "--ignore-rules" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        self.calls.append(prompt)
        compact = None
        if "SEMANTIC_COMPACT:\n" in prompt:
            from skillforge.repository.semantic_wire import CandidateAliases, CompactBatch

            data = json.loads(prompt.split("SEMANTIC_COMPACT:\n", 1)[1])
            from skillforge.repository.semantic_context_catalog import ExactContextCatalog

            expanded_contexts = {
                wire["request_id"]: ExactContextCatalog.expand(
                    wire["CANDIDATE_CONTEXT"], data.get("EXACT_SHARED_CONTEXT", {})
                )
                for wire in data["candidates"]
            }
            entries = []
            schemas = []
            for wire in data["candidates"]:
                matches = []
                for original, (request, _) in self.records.items():
                    policy, rest = original.split("\nRESPONSE_SCHEMA:\n", 1)
                    schema, context = rest.split("\nCANDIDATE_CONTEXT:\n", 1)
                    aliases = CandidateAliases(wire["request_id"], json.loads(context))
                    if (
                        policy == data["POLICIES"][wire["policy"]]
                        and json.loads(schema) == data["RESPONSE_SCHEMAS"][wire["response_schema"]]
                        and (
                            wire["candidate_id"] == candidate_identity(request)[1]
                            or aliases.encode(json.loads(context))
                            == expanded_contexts[wire["request_id"]]
                        )
                    ):
                        matches.append((original, request, json.loads(schema)))
                assert len(matches) == 1
                original, request, schema = matches[0]
                entries.append(
                    {
                        "request_id": request_id(request),
                        "repository_id": candidate_identity(request)[0],
                        "candidate_id": candidate_identity(request)[1],
                        "original_prompt": original,
                    }
                )
                schemas.append(schema)
            compact = CompactBatch(entries, schemas)
            assert compact.prompt == prompt
            assert all(
                c["CANDIDATE_CONTEXT"] == expanded_contexts[c["request_id"]]
                for c in compact.candidates
            )
            prompt = "SEMANTIC_BATCH:\n" + json.dumps(entries)
        if "SEMANTIC_BATCH:\n" in prompt:
            entries = json.loads(prompt.split("SEMANTIC_BATCH:\n", 1)[1])
            results = []
            for entry in entries:
                request, response = self.records[entry["original_prompt"]]
                assert entry["request_id"] == request_id(request)
                assert (entry["repository_id"], entry["candidate_id"]) == (
                    candidate_identity(request)
                )
                if self.target_channel in response:
                    self.target = self.target or request_id(request)
                raw = self.corrupt(response) if entry["request_id"] == self.target else response
                if self.mode == "partial" and entry["request_id"] == self.target:
                    continue
                item = {key: value for key, value in entry.items() if key != "original_prompt"}
                item.update(response=raw, error=None)
                if self.mode == "candidate_error" and entry["request_id"] == self.target:
                    item.update(response=None, error="fixture candidate temporarily unavailable")
                results.append(item)
                if self.mode == "duplicate" and entry["request_id"] == self.target:
                    results.append(deepcopy(item))
            if self.mode == "unparseable":
                output = "broken JSON"
            elif self.mode == "batch_failure":
                return subprocess.CompletedProcess(command, 1, "", "fixture batch unsupported")
            else:
                if compact:
                    for item in results:
                        key = next(
                            k
                            for k, v in compact.originals.items()
                            if v["request_id"] == item["request_id"]
                        )
                        wire = next(c for c in compact.candidates if c["request_id"] == key)
                        item.update(
                            {k: wire[k] for k in ("request_id", "repository_id", "candidate_id")}
                        )
                        item["response"] = compact.aliases[key].encode(item["response"])
                output = json.dumps({"results": list(reversed(results))})
        else:
            if prompt not in self.records:
                prefix, context = prompt.split("\nCANDIDATE_CONTEXT:\n", 1)
                data = json.loads(context)
                assert isinstance(data.get("validation_feedback"), str)
                assert 0 < len(data["validation_feedback"].encode("utf-8")) <= 512
                data.pop("validation_feedback")
                # Preserve exact policy/schema and every source fact in the fixture.
                matches = [
                    record
                    for original, record in self.records.items()
                    if original.split("\nCANDIDATE_CONTEXT:\n", 1)[0] == prefix
                    and json.loads(original.split("\nCANDIDATE_CONTEXT:\n", 1)[1]) == data
                ]
                assert len(matches) == 1
                request, response = matches[0]
            else:
                request, response = self.records[prompt]
            self.fallback_ids.append(request_id(request))
            if self.target_channel in response:
                self.target = self.target or request_id(request)
            output = json.dumps(
                self.corrupt(response) if request_id(request) == self.target else response
            )
        Path(command[command.index("-o") + 1]).write_text(output, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")


def recorded_models(inputs, mode="success", target_channel="workflow"):
    fixture = run(inputs, SingleFixture)
    kinds = (
        CodexCliStructuralSemanticModel,
        CodexCliSemanticModel,
        CodexCliAntiPatternSemanticModel,
    )
    records = {}
    for kind, model in zip(kinds, fixture[2], strict=True):
        adapter = kind(runner=lambda *args, **kwargs: None)
        for request in model.requests:
            _, prompt, _, _ = _adapter(adapter, request)
            records[prompt(request)] = (request, wire_response(model.responder.complete(request)))
    runner = RecordedFixtureRunner(records, mode, target_channel)
    models = tuple(kind(runner=runner) for kind in kinds)
    for model in models:
        model.semantic_cache_identity = complete_identity()
        model.semantic_cache_identity["response_parameters"]["mode"] = mode
    return fixture, runner, models


def test_keyed_batch_schema_requires_every_original_candidate(eleven_inputs):
    from jsonschema import Draft202012Validator, ValidationError

    fixture, runner, models = recorded_models(eleven_inputs)
    model = models[1]
    seen = []

    def keyed(command, **kwargs):
        result = runner(command, **kwargs)
        path = Path(command[command.index("-o") + 1])
        original = json.loads(path.read_text("utf-8"))
        raw = {"results": {entry["request_id"]: entry for entry in original["results"]}}
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text("utf-8"))
        validator = Draft202012Validator(schema)
        validator.validate(raw)
        missing = deepcopy(raw)
        missing["results"].pop(next(iter(missing["results"])))
        with pytest.raises(ValidationError):
            validator.validate(missing)
        mismatch = deepcopy(raw)
        mismatch["results"][next(iter(mismatch["results"]))]["candidate_id"] = "wrong-candidate"
        with pytest.raises(ValidationError):
            validator.validate(mismatch)
        seen.extend(json.loads(kwargs["input"].split("SEMANTIC_BATCH:\n", 1)[1]))
        path.write_text(json.dumps(raw), "utf-8")
        return result

    model._runner = keyed
    requests = fixture[2][1].requests
    responses = model.complete_batch(requests)
    assert set(responses) == {request_id(request) for request in requests}
    assert all(
        responses[request_id(request)] == fixture[2][1].responder.complete(request)
        for request in requests
    )
    assert {e["original_prompt"] for e in seen} == {
        _adapter(model, request)[1](request) for request in requests
    }


@pytest.mark.parametrize(
    "mode,calls,fallback",
    [
        ("success", 3, 0),
        ("malformed", 4, 1),
        ("gate_failure", 4, 1),
        ("partial", 4, 1),
        ("duplicate", 4, 1),
        ("candidate_error", 4, 1),
        ("unparseable", 14, 11),
        ("batch_failure", 14, 11),
    ],
)
def test_production_prepare_native_batch_and_single_fallback(
    eleven_inputs,
    tmp_path,
    mode,
    calls,
    fallback,
):
    _, runner, models = recorded_models(eleven_inputs, mode)
    directory = tmp_path.parent / (tmp_path.name + "-cli-cache")
    result = SkillForgePreparer(*models, semantic_cache_root=directory).prepare(
        PrepareRequest(
            eleven_inputs[0],
            "update routes",
            CodeAgentTarget.CODEX,
            "app.api.route0",
        )
    )
    assert len(runner.calls) == calls
    assert len(runner.fallback_ids) == fallback
    assert (
        result.workflow_semantic_selected_count,
        result.anti_pattern_semantic_selected_count,
    ) == (5, 5)
    assert result.structural_candidate_count == 1
    assert result.semantic_failure_count == (1 if mode in {"malformed", "gate_failure"} else 0)
    assert result.workflow_accepted_count == (4 if result.semantic_failure_count else 5)
    if fallback and mode not in {"unparseable", "batch_failure"}:
        assert set(runner.fallback_ids) == {runner.target}
    expected = run(
        eleven_inputs, SingleFixture, mode if mode in {"malformed", "gate_failure"} else "success"
    )
    baseline = load_baseline(eleven_inputs[0], result.evolution_baseline_path)
    assert baseline.source_dna == expected[1][0]
    assert baseline.source_compilation == expected[1][1]
    assert baseline.semantic_structural == expected[0].structural


def test_native_batch_cold_to_warm(eleven_inputs, tmp_path):
    expected, runner, models = recorded_models(eleven_inputs)
    directory = tmp_path.parent / (tmp_path.name + "-cli-cache")
    preparer = SkillForgePreparer(*models, semantic_cache_root=directory)
    first = preparer.prepare(
        PrepareRequest(
            eleven_inputs[0],
            "update routes",
            CodeAgentTarget.CODEX,
        )
    )
    second = preparer.prepare(
        PrepareRequest(
            eleven_inputs[0],
            "different task",
            CodeAgentTarget.CODEX,
        )
    )
    assert len(runner.calls) == 3 and not runner.fallback_ids
    assert first.dna_snapshot_id == second.dna_snapshot_id == expected[1][0].snapshot_id
    assert len(list(directory.glob("*.json"))) == 11
    from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache

    cache = RepoSnapshotSemanticCache(eleven_inputs[0], directory)
    for model, fixture in zip(models, expected[2], strict=True):
        cached = cache.wrap(model)
        for request in fixture.requests:
            assert cached.complete(request) == fixture.responder.complete(request)
    assert cache.hits == 11 and len(runner.calls) == 3
    preparer.prepare(
        PrepareRequest(
            eleven_inputs[0],
            "baseline task",
            CodeAgentTarget.CODEX,
            baseline=first.evolution_baseline_path,
        )
    )
    assert len(runner.calls) == 3  # Explicit Evolution baseline bypasses full-scan semantics.


@pytest.mark.parametrize("channel,retries", [("anti_pattern", 1), ("source_package", 1)])
def test_other_channel_gate_failure_is_candidate_local(eleven_inputs, tmp_path, channel, retries):
    _, runner, models = recorded_models(eleven_inputs, "gate_failure", channel)
    result = SkillForgePreparer(
        *models,
        semantic_cache_root=tmp_path.parent / (tmp_path.name + "-cache"),
    ).prepare(PrepareRequest(eleven_inputs[0], "task", CodeAgentTarget.CODEX))
    assert result.semantic_failure_count == 1
    assert len(runner.calls) == 3 + retries
    assert runner.fallback_ids == [runner.target] * retries
    assert result.workflow_accepted_count == 5


@pytest.mark.parametrize("identity", [None, {"model": "fixture"}, complete_identity()])
def test_native_custom_runner_requires_complete_matching_identity(identity):
    model = CodexCliSemanticModel(
        runner=lambda *args, **kwargs: pytest.fail("runner invoked"), timeout=181
    )
    if identity is not None:
        model.semantic_cache_identity = identity
    assert not supports_semantic_batch(model)


def test_unknown_cli_subclass_conservatively_keeps_single_path(eleven_inputs):
    class CustomWorkflow(CodexCliSemanticModel):
        pass

    fixture, runner, _ = recorded_models(eleven_inputs)
    model = CustomWorkflow(runner=runner)
    model.semantic_cache_identity = complete_identity()
    request = fixture[2][1].requests[0]
    assert prefetch_semantic_batch(model, (request,)) is model
    assert not runner.calls
    assert model.complete(request) == fixture[2][1].responder.complete(request)


def test_structural_batch_failure_keeps_successful_sibling(repository, tmp_path):
    from skillforge.repository.semantic_structural_rules import build_semantic_structural_request
    from skillforge.repository.static_analysis import PythonStaticAnalyzer
    from skillforge.repository.structural_rules import StructuralRuleMiner

    for package in ("worker", "store"):
        folder = repository / "src/app" / package
        folder.mkdir()
        (folder / "__init__.py").write_text("", encoding="utf-8")
    for index in range(2):
        (repository / f"src/app/store/unit{index}.py").write_text("", encoding="utf-8")
    for index in range(3):
        (repository / f"src/app/worker/unit{index}.py").write_text(
            f"import app.store.unit{index % 2}\n",
            encoding="utf-8",
        )
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(repository))
    assert len(mined.candidates) == 2
    adapter = CodexCliStructuralSemanticModel()
    responder = FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE)
    requests = tuple(build_semantic_structural_request(mined, c) for c in mined.candidates)
    records = {
        _adapter(adapter, r)[1](r): (r, wire_response(responder.complete(r))) for r in requests
    }
    runner = RecordedFixtureRunner(records, "gate_failure", "source_package")
    model = CodexCliStructuralSemanticModel(runner=runner)
    model.semantic_cache_identity = complete_identity()
    result = SkillForgePreparer(
        model,
        semantic_cache_root=tmp_path.parent / (tmp_path.name + "-cache"),
    ).prepare(PrepareRequest(repository, "task", CodeAgentTarget.CODEX))
    assert len(runner.calls) == 2 and runner.fallback_ids == [runner.target]
    assert result.semantic_accepted_count == result.semantic_failure_count == 1
    assert result.dna_rule_count == result.compiled_skill_count == 1


@pytest.mark.parametrize("corruption", ["swap", "wrong_id", "missing_id", "duplicate"])
def test_unproven_correspondence_salvages_only_proven_candidates(eleven_inputs, corruption):
    fixture, _, models = recorded_models(eleven_inputs)
    requests = fixture[2][1].requests[:3]
    results = []
    for request in requests:
        repo, candidate = candidate_identity(request)
        results.append(
            {
                "request_id": request_id(request),
                "repository_id": repo,
                "candidate_id": candidate,
                "response": wire_response(fixture[2][1].responder.complete(request)),
                "error": None,
            }
        )
    if corruption == "swap":
        results[0]["candidate_id"], results[1]["candidate_id"] = (
            results[1]["candidate_id"],
            results[0]["candidate_id"],
        )
    elif corruption == "wrong_id":
        results[0]["request_id"] = "unknown"
    elif corruption == "missing_id":
        results[0].pop("request_id")
    else:
        results.append(deepcopy(results[0]))
    parsed = parse_cli_batch({"results": results}, requests, models[1].parse)
    assert request_id(requests[0]) not in parsed
    assert request_id(requests[2]) in parsed
    assert (request_id(requests[1]) in parsed) == (corruption != "swap")


def test_tendency_batch_keeps_exact_request_binding(tmp_path):
    from test_semantic_tendencies import response_for
    from test_structural_roles import _repo

    from skillforge.repository.semantic_tendencies import (
        bind_tendency_decision,
        validate_semantic_tendency_response,
    )

    root, _, history, facts, _, candidates = _repo(
        tmp_path,
        [(f"nebula/plugins/unit_{i}.py",) for i in range(3)],
        initial=("nebula/__init__.py", "nebula/plugins/__init__.py"),
    )
    (seed,) = candidates.candidates
    candidates = replace(
        candidates,
        candidates=tuple(replace(seed, candidate_id=key) for key in ("first", "second")),
    )
    requests = tuple(
        build_semantic_tendency_request(history, facts, candidates, c)
        for c in candidates.candidates
    )
    adapter = CodexCliTendencySemanticModel()
    records = {adapter.prompt(r): (r, wire_response(response_for(r))) for r in requests}
    runner = RecordedFixtureRunner(records)
    model = CodexCliTendencySemanticModel(runner=runner)
    model.semantic_cache_identity = complete_identity()
    responses = model.complete_batch(requests)
    assert len(runner.calls) == 1 and len(responses) == 2
    for request in requests:
        assert bind_tendency_decision(request, responses[request_id(request)]) == (
            response_for(request)
        )
        validate_semantic_tendency_response(request, response_for(request))
    with pytest.raises(ValueError, match="scope|request"):
        bind_tendency_decision(requests[0], responses[request_id(requests[1])])
    from skillforge.domain.structural_rules import StructuralRuleMiningResult
    from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
    from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache

    def cached_evaluate(cache):
        result = evaluate_semantic_knowledge(
            facts.repository_id,
            facts,
            StructuralRuleMiningResult(facts.repository_id, (), ()),
            None,
            history=history,
            candidates=candidates,
            selected_workflows=candidates.candidates,
            tendency_model=cache.wrap(model),
        )
        cache.publish(result.tendencies)
        return result

    directory = tmp_path.parent / (tmp_path.name + "-tendency-cache")
    cold = cached_evaluate(RepoSnapshotSemanticCache(root, directory))
    warm_cache = RepoSnapshotSemanticCache(root, directory)
    assert cached_evaluate(warm_cache) == cold
    assert warm_cache.hits == 2 and warm_cache.misses == 0
    assert len(runner.calls) == 2
