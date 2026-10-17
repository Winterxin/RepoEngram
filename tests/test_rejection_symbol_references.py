"""Historical symbol mentions in rejections are not published file references."""
from dataclasses import replace

import pytest
from test_semantic_antipatterns import _candidate_result, _history, _valid_response

from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionKind
from skillforge.repository.semantic_antipatterns import (
    _validate_decision,
    build_semantic_antipattern_request,
)


def request():
    history = _history({'paths': ('src/cache.py',),
                        'message': 'Fix Store.lookup and Parser.match; inspect Notes.md'})
    candidates = _candidate_result(history)
    return build_semantic_antipattern_request(history, candidates, candidates.candidates[0])


def test_rejection_can_quote_exact_cited_member_without_creating_knowledge():
    req = request()
    response = replace(_valid_response(req, SemanticAntiPatternDecisionKind.INCOHERENT),
                       reason='Store.lookup and Parser.match are different corrections.')
    draft, rejection, _ = _validate_decision(req, response)
    assert draft is None
    assert rejection.reason == response.reason


@pytest.mark.parametrize('token', ['Unknown.lookup', 'Store.lookups', 'Notes.md',
                                  'src/Store.lookup', 'Other.secret'])
def test_unknown_names_and_file_paths_still_rejected(token):
    req = request()
    response = replace(_valid_response(req, SemanticAntiPatternDecisionKind.INCOHERENT),
                       reason=f'{token} is relevant.')
    with pytest.raises(ValueError, match='unknown path'):
        _validate_decision(req, response)


def test_uncited_commit_message_does_not_authorize_symbol():
    req = request()
    response = replace(_valid_response(req, SemanticAntiPatternDecisionKind.INCOHERENT),
                       reason='Store.lookup is relevant.',
                       decision_evidence_ids=tuple(item.evidence_id for item in req.evidence
                                                   if item.provenance != 'git_log'))
    with pytest.raises(ValueError, match='unknown path'):
        _validate_decision(req, response)


def test_accepted_reason_and_draft_keep_strict_path_validation():
    req = request()
    response = _valid_response(req)
    with pytest.raises(ValueError, match='unknown path'):
        _validate_decision(req, replace(response, reason='Store.lookup is relevant.'))
    with pytest.raises(ValueError, match='unknown path'):
        _validate_decision(req, replace(response, anti_pattern=replace(
            response.anti_pattern, avoid_behavior='Avoid Store.lookup.')))
