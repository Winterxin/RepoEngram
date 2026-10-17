"""Explicit per-event choices preserve the original acceptance and scope gates."""
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_semantic_tendencies import inputs as inputs, request_for, response_for

from skillforge.domain.semantic_tendency import ScopedTendencyDecision, TendencyDecisionKind
from skillforge.repository.codex_cli_tendency import parse_semantic_tendency_response
from skillforge.repository.semantic_reference_schema import bind_response_schema
from skillforge.repository.semantic_tendencies import (
    bind_tendency_decision, tendency_request_fingerprint, validate_semantic_tendency_response,
)
from skillforge.repository.semantic_tendency_prompt import tendency_citation_schema


def choice(request):
    sources = {e.evidence_id: e for e in request.evidence}
    citations = {f"e{i}_id": next(k for k in event.evidence_ids if sources[k].provenance == "git_diff"
        and sources[k].path == event.path and sources[k].commit_id == event.commit_id)
        for i, event in enumerate(request.structural_events)}
    return {"decision": "SUPPORTED_TENDENCY", "reason": "Repeated local structural choices.",
            "decision_evidence_id": request.evidence[0].evidence_id, "citations": citations}


def bind(request, raw):
    return bind_tendency_decision(request, ScopedTendencyDecision(
        tendency_request_fingerprint(request), parse_semantic_tendency_response(raw)))


def test_explicit_model_choices_and_original_gate_acceptance(inputs):
    request = request_for(inputs)
    raw = choice(request)
    raw["citations"] = dict(reversed(tuple(raw["citations"].items())))
    result = bind(request, raw)
    expected = set(raw["citations"].values()) | {raw["decision_evidence_id"]}
    assert result.evidence_ids == tuple(sorted(expected))
    accepted = validate_semantic_tendency_response(request, result)
    assert accepted.response == result
    assert accepted.rule == validate_semantic_tendency_response(request, response_for(request)).rule


@pytest.mark.parametrize("fault", ["missing", "null", "foreign", "wrong_event", "extra"])
def test_no_missing_or_foreign_choice_is_filled_or_repaired(inputs, fault):
    request = request_for(inputs)
    raw = choice(request)
    if fault == "missing": del raw["citations"]["e0_id"]
    elif fault == "null": raw["citations"]["e0_id"] = None
    elif fault == "foreign": raw["citations"]["e0_id"] = "unknown"
    elif fault == "wrong_event": raw["citations"]["e0_id"] = raw["citations"]["e1_id"]
    else: raw["citations"]["e999_id"] = raw["citations"]["e0_id"]
    with pytest.raises(ValueError): bind(request, raw)


def test_schema_and_gate_independently_close_each_event(inputs):
    request = request_for(inputs)
    raw = choice(request)
    schema = bind_response_schema(tendency_citation_schema(request), request.to_dict())
    validator = Draft202012Validator(schema)
    validator.validate(raw)
    other_event = raw["citations"]["e1_id"]
    raw["citations"]["e0_id"] = other_event
    with pytest.raises(ValidationError): validator.validate(raw)
    with pytest.raises(ValueError): bind(request, raw)


def test_rejection_and_legacy_gate_are_not_reclassified(inputs):
    request = request_for(inputs)
    raw = choice(request)
    raw["decision"] = "INSUFFICIENT_EVIDENCE"
    raw["citations"] = {k: None for k in raw["citations"]}
    rejected = bind(request, raw)
    assert validate_semantic_tendency_response(request, rejected).decision == TendencyDecisionKind.INSUFFICIENT_EVIDENCE
    legacy = replace(response_for(request), evidence_ids=tuple(reversed(response_for(request).evidence_ids)))
    assert bind_tendency_decision(request, legacy) is legacy
    with pytest.raises(ValueError, match="sorted"): validate_semantic_tendency_response(request, legacy)
