"""Candidate reference domains constrain transport without inventing Evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_cold_semantic_batching import BatchFixture, SingleFixture, run
from test_cold_semantic_batching import eleven_inputs as eleven_inputs
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_semantic_batch_transport import recorded_models, wire_response

from skillforge.repository.semantic_reference_schema import bind_response_schema
from skillforge.repository.semantic_workflow_prompt import semantic_workflow_response_schema


def _current_protocol_response(raw):
    raw = deepcopy(raw)
    raw["classification_protocol"] = "ordered-step-classification-v2"
    workflow = raw.get("workflow")
    if workflow is None:
        return raw
    classifications = {}
    for kind in ("required", "optional", "conditional"):
        for item in workflow.pop(f"{kind}_steps"):
            classifications[item["step_id"]] = (kind, item)
    for step in workflow["ordered_steps"]:
        kind, item = classifications[step["step_id"]]
        step.update({
            "classification_kind": kind,
            "classification_reason": item["reason"],
            "classification_evidence_ids": item["evidence_ids"],
            "classification_condition": item["condition"],
        })
    return raw


def test_bound_schema_preserves_valid_response_and_rejects_wrong_reference_kinds(eleven_inputs):
    fixture = run(eleven_inputs, SingleFixture)
    request = fixture[2][1].requests[0]
    base = semantic_workflow_response_schema()
    original = deepcopy(base)
    bound = bind_response_schema(base, request.to_dict())
    assert base == original
    Draft202012Validator.check_schema(bound)
    valid = _current_protocol_response(wire_response(
        fixture[2][1].responder.complete(request)
    ))
    validator = Draft202012Validator(bound)
    validator.validate(valid)
    for wrong in (
        request.candidate_id,
        request.source_patterns[0].pattern_id,
        request.supporting_commits[0].commit_id,
        "unknown",
    ):
        bad = deepcopy(valid)
        bad["decision_evidence_ids"] = [wrong]
        with pytest.raises(ValidationError):
            validator.validate(bad)
    bad = deepcopy(valid)
    bad["workflow"]["ordered_steps"][0]["evidence_ids"] = ["unknown"]
    with pytest.raises(ValidationError):
        validator.validate(bad)
    bad = deepcopy(valid)
    bad["workflow"]["ordered_steps"][0]["classification_evidence_ids"] = ["unknown"]
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_fact_and_evidence_domains_are_separate_and_empty_domain_fails_closed():
    schema = {
        "type": "object",
        "properties": {
            "supporting_fact_ids": {"type": "array", "items": {"type": "string"}},
            "decision_evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    context = {"evidence": [{"evidence_id": "e"}], "evidence_groups": [{"fact_id": "f"}]}
    bound = bind_response_schema(schema, context)
    validator = Draft202012Validator(bound)
    validator.validate({"supporting_fact_ids": ["f"], "decision_evidence_ids": ["e"]})
    with pytest.raises(ValidationError):
        validator.validate({"supporting_fact_ids": ["e"]})
    with pytest.raises(ValueError, match="empty"):
        bind_response_schema(schema, {"evidence": [], "evidence_groups": []})


def test_all_valid_batch_siblings_checkpoint_before_first_budgeted_repair(eleven_inputs, tmp_path):
    import json
    from dataclasses import replace

    from test_prepare import WorkflowModel

    from skillforge.cost_control import BudgetExceeded

    directory = tmp_path.parent / (tmp_path.name + "-semantic-cache")
    expected = (
        {c.candidate_id for c in eleven_inputs[5]}
        | {c.candidate_id for c in eleven_inputs[6]}
        | {c.candidate_id for c in eleven_inputs[7].candidates}
    )
    invalid_id = min(c.candidate_id for c in eleven_inputs[5])

    class AbortRepair(BatchFixture):
        def response(self, request):
            response = super().response(request)
            if isinstance(self.responder, WorkflowModel) and request.candidate_id == invalid_id:
                return replace(response, decision_evidence_ids=(request.candidate_id,))
            return response

        def complete(self, request):
            saved = {
                json.loads(p.read_text("utf-8"))["response"]["candidate_id"]
                for p in directory.glob("*.json")
            }
            assert saved == expected - {invalid_id}
            raise BudgetExceeded("stop at first repair; every valid sibling already saved")

    with pytest.raises(BudgetExceeded, match="every valid sibling"):
        run(eleven_inputs, AbortRepair, cache_root=directory)
    assert len(list(directory.glob("*.json"))) == 10


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_native_single_adapters_deliver_the_bound_schema(eleven_inputs, channel):
    fixture, runner, models = recorded_models(eleven_inputs)
    request = fixture[2][channel].requests[0]
    model = models[channel]
    seen = []

    def checked(command, **kwargs):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text("utf-8"))
        assert set(schema["$defs"]["evidence"]["enum"]) == {e.evidence_id for e in request.evidence}
        assert schema["properties"]["candidate_id"]["enum"] == [request.candidate_id]
        expected = wire_response(fixture[2][channel].responder.complete(request))
        if channel == 1:
            expected = _current_protocol_response(expected)
        Draft202012Validator(schema).validate(expected)
        seen.append(schema)
        return runner(command, **kwargs)

    model._runner = checked
    result = model.complete(request)
    assert result == fixture[2][channel].responder.complete(request)
    assert len(seen) == 1


def test_second_channel_transport_abort_keeps_first_valid_batch(eleven_inputs, tmp_path):
    from test_prepare import AntiPatternModel

    from skillforge.cost_control import BudgetExceeded

    directory = tmp_path.parent / (tmp_path.name + "-channel-cache")
    workflow_ids = {c.candidate_id for c in eleven_inputs[5]}

    class AbortSecondChannel(BatchFixture):
        def complete_batch(self, requests):
            if isinstance(self.responder, AntiPatternModel):
                saved = {
                    json.loads(p.read_text("utf-8"))["response"]["candidate_id"]
                    for p in directory.glob("*.json")
                }
                assert saved == workflow_ids
                raise BudgetExceeded("fixture second-channel deadline")
            return super().complete_batch(requests)

    with pytest.raises(BudgetExceeded, match="second-channel"):
        run(eleven_inputs, AbortSecondChannel, cache_root=directory)
    assert len(list(directory.glob("*.json"))) == 5
