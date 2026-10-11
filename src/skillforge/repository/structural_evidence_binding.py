"""Expand explicitly selected structured facts; never repair semantic answers."""

import hashlib
import json
from dataclasses import replace

from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    StructuralEvidenceGroup,
)
from skillforge.domain.serialization import to_json_value


def evidence_groups(candidate, evidence):
    """Current testing linkages have canonical identities without changing mining."""
    sources = {e.evidence_id: e for e in evidence}
    groups = []
    for link in candidate.test_linkages:
        if not set(link.evidence_ids) <= set(sources):
            raise ValueError("structured linkage missing underlying Evidence")
        refs = [sources[key] for key in link.evidence_ids]
        categories = (
            (f"Python implementation {link.operation_symbol}:", None, None),
            (f"Python implementation {link.test_symbol}:", link.path, None),
            (f"Python public test call {link.test_symbol} -> {link.operation_symbol}:",
             link.path, link.call_line),
            (f"Python behavior assertion {link.test_symbol}:", link.path, link.assertion_line),
        )
        if not all(any(e.description.startswith(prefix) and (path is None or e.path == path)
                       and (line is None or e.line == line) for e in refs)
                   for prefix, path, line in categories):
            raise ValueError("structured linkage lacks production/test/call/assertion provenance")
        module = next((m for m in candidate.source_modules
                       if link.operation_symbol.startswith(m + ".")), None)
        suffix = module.replace(".", "/") + ".py" if module else ""
        scope = tuple(sorted(e.evidence_id for e in refs
                             if e.description.startswith(
                                 f"Python implementation {link.operation_symbol}:")
                             and suffix and (e.path == suffix or e.path.endswith("/" + suffix))))
        if not scope:
            raise ValueError("structured linkage lacks real scope provenance")
        payload = json.dumps((candidate.candidate_id, to_json_value(link)),
                             sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        groups.append(StructuralEvidenceGroup(
            "fact-" + hashlib.sha256(payload.encode()).hexdigest()[:24],
            f"{link.test_symbol} exercises {link.operation_symbol} ({link.assertion_kind})",
            tuple(sorted(set(link.evidence_ids))), scope,
        ))
    return tuple(sorted(groups, key=lambda g: g.fact_id))


def bind_structural_evidence(request, response, candidate):
    """Coverage is explicit; raw-ID legacy responses retain their original gate."""
    if response.supporting_fact_ids is None:
        return response
    if response.decision_evidence_ids or response.scope_evidence_ids:
        raise ValueError("structured references cannot mix with raw Evidence transport")
    canonical = evidence_groups(candidate, request.evidence)
    if not canonical or canonical != request.evidence_groups:
        raise ValueError("structured facts do not match current candidate provenance")
    selected = response.supporting_fact_ids
    if (not isinstance(selected, tuple) or not selected
            or any(not isinstance(key, str) for key in selected)):
        raise ValueError("semantic decision requires explicit structured fact references")
    groups = {g.fact_id: g for g in canonical}
    chosen = set(selected)
    if not chosen <= set(groups):
        raise ValueError("semantic reference does not belong to candidate")
    accepted = response.decision == SemanticStructuralDecisionKind.ARCHITECTURE_RULE
    if accepted and chosen != set(groups):
        raise ValueError("semantic acceptance lacks required structured linkage coverage")
    decision = tuple(sorted({key for fact_id in chosen for key in groups[fact_id].evidence_ids}))
    scope = tuple(sorted({key for fact_id in chosen
                          for key in groups[fact_id].scope_evidence_ids})) if accepted else ()
    return replace(response, decision_evidence_ids=decision, scope_evidence_ids=scope,
                   supporting_fact_ids=None)
