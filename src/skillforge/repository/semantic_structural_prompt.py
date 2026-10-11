"""Structured, candidate-scoped prompt for structural rule evaluation."""

import json

from skillforge.domain.semantic_structural_rule import SemanticStructuralRequest


def semantic_structural_response_schema(*, structured: bool = False) -> dict[str, object]:
    ids = {"type": "array", "items": {"type": "string"}}
    exception = {
        "type": "object", "additionalProperties": False,
        "required": ["description", "evidence_ids"],
        "properties": {"description": {"type": "string"}, "evidence_ids": ids},
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["repository_id", "candidate_id", "decision", "reason",
                     "decision_evidence_ids", "source_package", "target_package",
                     "scope_modules", "scope_evidence_ids", "exceptions"],
        "properties": {
            "repository_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "decision": {"type": "string", "enum": [
                "ARCHITECTURE_RULE", "INCOHERENT", "INSUFFICIENT_EVIDENCE"]},
            "reason": {"type": "string"},
            "decision_evidence_ids": ids,
            "source_package": {"type": "string"},
            "target_package": {"type": "string"},
            "scope_modules": ids,
            "scope_evidence_ids": ids,
            "exceptions": {"type": "array", "items": exception},
        },
    }
    if structured:
        for key in ("decision_evidence_ids", "scope_evidence_ids"):
            schema["required"].remove(key)
            del schema["properties"][key]
        schema["required"].append("supporting_fact_ids")
        schema["properties"]["supporting_fact_ids"] = ids
    return schema


def build_semantic_structural_prompt(request: SemanticStructuralRequest) -> str:
    context = json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True)
    schema = json.dumps(semantic_structural_response_schema(
        structured=bool(request.evidence_groups)), ensure_ascii=False, sort_keys=True)
    if request.observation_kind == "static_testing":
        return "\n".join((
            "Evaluate one current-source TESTING CONVENTION / STRUCTURAL PROJECT RULE.",
            "Use only CANDIDATE_CONTEXT. This is not Git workflow knowledge.",
            "Candidate is not Confirmed; association alone is not a requirement.",
            "Choose ARCHITECTURE_RULE only if at least two independent analogous public",
            "operations in the same family each have a formal project behavior test,",
            "resolved public invocation, and assertions of externally visible behavior.",
            "Evaluate actual production and test bodies, receiver bindings and assertion flow.",
            "Reject single samples, helper/internal-only tests, unreliable linkage, incidental",
            "strings, generated or snapshot-only tests, and comparable untested operations.",
            "Acceptance supports only a conditional convention: when adding another operation",
            "in this observed family, add/update the corresponding project-level behavior test",
            "using the same public pattern, unless repository evidence shows an exception.",
            "Do not infer a policy that every feature needs tests. Do not invent intent or files.",
            "Otherwise choose INCOHERENT or INSUFFICIENT_EVIDENCE.",
            "Preserve repository_id, candidate_id, source_package, target_package exactly.",
            "Acceptance scope_modules equals source_modules. Select supporting_fact_ids from",
            "evidence_groups to identify the structured linkages grounding your decision.",
            "Acceptance needs complete coverage of the candidate's behavior linkages; omission",
            "fails validation. Rejection can cite a supporting subset for its explanation.",
            "Do not copy raw Evidence IDs. The deterministic binder expands only validated",
            "selected facts and their real scope provenance, then applies the original gate.",
            "Return empty exceptions; rejection has empty scope arrays.",
            "Give an observational reason without must, should, require or unconditional policy.",
            "Return one JSON object matching RESPONSE_SCHEMA.",
            "RESPONSE_SCHEMA:", schema, "CANDIDATE_CONTEXT:", context,
        ))
    if request.observation_kind == "static_delegation":
        return "\n".join((
            "Evaluate one current-source architecture candidate using only CANDIDATE_CONTEXT.",
            "This is structural knowledge, not historical development workflow evidence.",
            "The operations share a family and complete resolved delegation topology.",
            "Choose ARCHITECTURE_RULE only if the symbols, bindings and actual call edges",
            "support a coherent conditional repository architecture convention.",
            "A new operation in the same family can preserve the observed topology unless",
            "current repository evidence shows an exception. Single examples, helpers or",
            "comparable contradictory topology do not justify a convention.",
            "Otherwise choose INCOHERENT or INSUFFICIENT_EVIDENCE.",
            "Do not invent layers, files, business intent, mandatory policy or exceptions.",
            "Preserve repository_id, candidate_id, source_package and target_package exactly.",
            "Acceptance scope_modules equals source_modules; cite actual supplied Evidence IDs",
            "in sorted unique decision_evidence_ids and scope_evidence_ids.",
            "Return empty exceptions: no concrete exception is present in these observations.",
            "Rejection has empty scope arrays. Every decision cites Evidence and gives a reason.",
            "Do not use must, should, require, always or other unconditional policy claims.",
            "Return one JSON object matching RESPONSE_SCHEMA.",
            "RESPONSE_SCHEMA:", schema, "CANDIDATE_CONTEXT:", context,
        ))
    return "\n".join((
        "Evaluate exactly one deterministic structural candidate using only CANDIDATE_CONTEXT.",
        "Repeated imports are observations, not a project policy or a mandatory rule.",
        "Choose ARCHITECTURE_RULE only if the supplied imports have a coherent, explainable",
        "candidate architecture meaning. Otherwise choose INCOHERENT or INSUFFICIENT_EVIDENCE.",
        "Do not invent files, modules, packages, policy, exceptions, layer names, business",
        "intent, or developer intention. Do not rewrite canonical_observation.",
        "Preserve repository_id, candidate_id, source_package and target_package exactly.",
        "For acceptance, scope_modules must equal source_modules and cite",
        "decision_evidence_ids and scope_evidence_ids from this candidate's evidence.",
        "The supplied Evidence consists only of positive import observations; return no",
        "exceptions. For rejection, return empty scope arrays and empty exceptions.",
        "Every decision, including rejection, requires a reason and Evidence IDs.",
        "Do not use must, should, ought, require, mandate, forbidden, or other policy claims.",
        "Return exactly one JSON object matching RESPONSE_SCHEMA, without extra keys.",
        "RESPONSE_SCHEMA:", schema, "CANDIDATE_CONTEXT:", context,
    ))
