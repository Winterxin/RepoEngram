"""Identity is tied to one actual transport request, independent of model echoes."""

from dataclasses import replace

import pytest
from test_semantic_tendencies import inputs as inputs
from test_semantic_tendencies import request_for

from skillforge.domain.semantic_tendency import ScopedTendencyDecision
from skillforge.repository.codex_cli_tendency import parse_semantic_tendency_response
from skillforge.repository.semantic_tendencies import (
    SemanticTendencyEvaluator,
    bind_tendency_decision,
    tendency_request_fingerprint,
    validate_semantic_tendency_response,
)


def raw(request, **extras):
    return {
        "decision": "SUPPORTED_TENDENCY", "reason": "Observed independent local additions.",
        "evidence_ids": [item.evidence_id for item in request.evidence], **extras,
    }


def scoped(request, data):
    return ScopedTendencyDecision(tendency_request_fingerprint(request),
                                  parse_semantic_tendency_response(data))


def test_single_candidate_missing_identity_binds_correct_object(inputs):
    request = request_for(inputs)
    result = bind_tendency_decision(request, scoped(request, raw(request)))
    assert result.candidate_id == request.context.candidate_id
    assert result.repository_id == request.context.repository_id
    assert not result.audit_warnings
    assert validate_semantic_tendency_response(request, result).rule.rule_id == result.candidate_id


def test_wrong_echo_is_audited_and_caller_identity_is_preserved(inputs):
    request = request_for(inputs)
    result = bind_tendency_decision(request, scoped(request, raw(request, candidate_id="wrong")))
    assert result.candidate_id == request.context.candidate_id
    assert len(result.audit_warnings) == 1 and "wrong" in result.audit_warnings[0]
    assert validate_semantic_tendency_response(request, result).rule.rule_id == result.candidate_id


def test_multi_candidate_responses_cannot_be_swapped(inputs):
    first = request_for(inputs)
    second = replace(first, context=replace(first.context, candidate_id="another-candidate"))
    responses = [scoped(request, raw(request)) for request in (first, second)]
    for request, response in zip((first, second), responses, strict=True):
        bound = bind_tendency_decision(request, response)
        assert bound.candidate_id == request.context.candidate_id
    for request, response in zip((first, second), reversed(responses), strict=True):
        with pytest.raises(ValueError, match="different request"):
            bind_tendency_decision(request, response)


@pytest.mark.parametrize("data", [
    {}, {"decision": "SUPPORTED_TENDENCY", "reason": "x", "evidence_ids": "wrong"},
    {"decision": "INVALID", "reason": "x", "evidence_ids": []},
    {"decision": "SUPPORTED_TENDENCY", "reason": 1, "evidence_ids": []},
    {"decision": "SUPPORTED_TENDENCY", "reason": "x", "evidence_ids": [], "extra": 1},
])
def test_malformed_response_is_rejected(data):
    with pytest.raises(ValueError):
        parse_semantic_tendency_response(data)


def test_unscoped_decision_is_rejected(inputs):
    request = request_for(inputs)
    with pytest.raises(ValueError, match="different request"):
        bind_tendency_decision(request, parse_semantic_tendency_response(raw(request)))


def test_evaluator_binds_transport_and_retains_warning(inputs):
    _, _, history, facts, _, candidates = inputs

    class Model:
        def complete(self, request):
            return scoped(request, raw(request, candidate_id="untrusted-echo"))

    result = SemanticTendencyEvaluator().evaluate(
        history, facts, candidates, Model(), selected_candidates=candidates.candidates,
    )
    accepted, = result.accepted
    assert accepted.rule.rule_id == candidates.candidates[0].candidate_id
    assert accepted.response.audit_warnings
