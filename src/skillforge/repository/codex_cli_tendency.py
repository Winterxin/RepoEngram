"""History tendency interpretation through the shared isolated Codex transport."""

from skillforge.domain.semantic_tendency import (
    ScopedTendencyDecision,
    TendencyDecisionKind,
    TendencyTransportDecision,
)
from skillforge.repository.codex_cli_semantic import (
    CodexCliSemanticModel,
    _string,
    _strings,
)
from skillforge.repository.semantic_tendencies import tendency_request_fingerprint
from skillforge.repository.semantic_tendency_prompt import (
    build_semantic_tendency_prompt,
    semantic_tendency_response_schema,
    tendency_citation_schema,
)


def parse_semantic_tendency_response(value):
    required = set(semantic_tendency_response_schema()["required"])
    choices = None
    if isinstance(value, dict) and "citations" in value:
        required = {"decision", "reason", "decision_evidence_id", "citations"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or not value.keys() <= required | {"repository_id", "candidate_id"}):
        raise ValueError("tendency response has unexpected or missing fields")
    data = value
    if "citations" in data:
        citations = data["citations"]
        if not isinstance(citations, dict) or any(not isinstance(k, str) or
                (v is not None and not isinstance(v, str)) for k, v in citations.items()):
            raise ValueError("tendency citations must be an explicit event map")
        choices = tuple(citations.items())
        refs = (_string(data["decision_evidence_id"], "decision Evidence ID"),)
    else:
        refs = _strings(data["evidence_ids"], "Evidence IDs")
    return TendencyTransportDecision(
        TendencyDecisionKind(_string(data["decision"], "decision")),
        _string(data["reason"], "reason"), refs,
        (_string(data["repository_id"], "repository_id") if "repository_id" in data else None),
        (_string(data["candidate_id"], "candidate_id") if "candidate_id" in data else None),
        choices,
    )


class CodexCliTendencySemanticModel(CodexCliSemanticModel):
    schema = staticmethod(semantic_tendency_response_schema)
    schema_for = staticmethod(tendency_citation_schema)
    prompt = staticmethod(build_semantic_tendency_prompt)
    parse = staticmethod(parse_semantic_tendency_response)

    def complete(self, request):
        fingerprint = tendency_request_fingerprint(request)
        decision = super().complete(request)
        return ScopedTendencyDecision(fingerprint, decision)
