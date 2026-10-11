"""Expose published risk knowledge without inventing a workflow or growing discovery."""
from __future__ import annotations

import json

from skillforge.domain.project_dna import ProjectKnowledgeStatus
from skillforge.domain.semantic_antipattern import SemanticAntiPatternScopeKind

_MAX_RISKS = 3
_MAX_RISK_BYTES = 2048
_ALLOWED = frozenset({ProjectKnowledgeStatus.CANDIDATE,
                      ProjectKnowledgeStatus.CONFIRMED,
                      ProjectKnowledgeStatus.MANUAL_OVERRIDE})


def _size(value, *, escaped=False):
    encoded = json.dumps(value, ensure_ascii=False)
    if escaped:
        encoded = json.dumps(encoded, ensure_ascii=False)
    return len(encoded.encode('utf-8'))


def _applies(pattern, scope):
    source = pattern.draft.scope
    if pattern.status not in _ALLOWED:
        return False
    if source.kind == SemanticAntiPatternScopeKind.REPOSITORY:
        return True
    # No guessed module-to-file conversions or directory widening.
    return (source.kind == SemanticAntiPatternScopeKind.FILE
            and scope.get('current_file') in source.selectors)


def discovery_with_risks(response, dna, route, scope, consumed):
    """Fit complete advisory risks into the original raw AND escaped JSON allowance.

    Full canonical routing decisions remain reproducible from the Registry/Router;
    only model-facing diagnostic detail is compacted. Never clip an exception or
    assert that these risks have been independently verified.
    """
    hosted = {risk.anti_pattern_id for match in route.matches
              for risk in getattr(match.skill, 'risks', ())}
    patterns = [p for p in sorted(dna.anti_patterns, key=lambda p: p.anti_pattern_id)
                if p.anti_pattern_id not in hosted and _applies(p, scope)]
    if not patterns:
        return response
    original_trace = response['router_trace']
    trace = {key: original_trace[key] for key in
             ('schema', 'algorithm', 'registry_id', 'source_snapshot_id',
              'selected_skill_ids') if key in original_trace}
    trace['detail_status'] = 'COMPACT_FOR_SCOPED_RISKS'
    trace['candidate_count'] = len(original_trace.get('candidates', ()))
    result = {**response, 'router_trace': trace,
              'source_risks': [], 'omitted_source_risk_count': len(patterns),
              'risk_policy': 'Published advisory knowledge; exceptions apply. '
                             'Not formal verification. Evidence: get_evidence(rule_id=risk_id).'}
    raw_budget, escaped_budget = _size(response), _size(response, escaped=True)
    # An unusually tiny original response has no allowance for this envelope.
    if _size(result) > raw_budget or _size(result, escaped=True) > escaped_budget:
        return response
    for pattern in patterns:
        if len(result['source_risks']) == _MAX_RISKS:
            break
        draft = pattern.draft
        entry = {'risk_id': pattern.anti_pattern_id,
                 'source_status': pattern.status.value,
                 'scope': {'kind': draft.scope.kind.value,
                           'selectors': list(draft.scope.selectors)}}
        if pattern.anti_pattern_id in consumed:
            entry['already_read'] = True
        else:
            entry.update(avoid_behavior=draft.avoid_behavior,
                         exceptions=[e.description for e in draft.exceptions],
                         confidence=draft.confidence.score)
        trial = {**result, 'source_risks': [*result['source_risks'], entry],
                 'omitted_source_risk_count': result['omitted_source_risk_count'] - 1}
        if (_size(trial['source_risks']) > _MAX_RISK_BYTES
                or _size(trial) > raw_budget
                or _size(trial, escaped=True) > escaped_budget):
            continue
        result = trial
        if not entry.get('already_read'):
            consumed.add(pattern.anti_pattern_id)
    return result
