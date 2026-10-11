"""Bind transport reference fields to the exact request; acceptance stays in gates."""

from __future__ import annotations

import json


def reference_domains(context):
    """Use authoritative Evidence/fact collections, never incidental *_ids fields."""
    return {
        "evidence": sorted({item["evidence_id"] for item in context.get("evidence", ())}),
        "facts": sorted({item["fact_id"] for item in context.get("evidence_groups", ())}),
    }


def bind_response_schema(schema, context, *, definition_prefix=""):
    """Constrain identity and all nested reference arrays, sharing each enum once.

    No evidence selection, response repair or knowledge decision occurs here.
    Empty required domains fail before transport rather than granting arbitrary IDs.
    """
    # Schema builders reuse the same Python dict for unrelated array fields.
    # JSON copying breaks those aliases, so binding Evidence cannot bind modules.
    result = json.loads(json.dumps(schema))
    domains = reference_domains(context)
    definitions = result.setdefault("$defs", {})

    def visit(node):
        if not isinstance(node, dict):
            return
        for name, prop in node.get("properties", {}).items():
            kind = (
                "facts"
                if name == "supporting_fact_ids"
                else "evidence"
                if name == "evidence_ids" or name.endswith("_evidence_ids")
                else None
            )
            if kind:
                if not domains[kind]:
                    raise ValueError(f"empty {kind} reference domain")
                key = definition_prefix + kind
                definition = {"type": "string", "enum": domains[kind]}
                if key in definitions and definitions[key] != definition:
                    raise ValueError("reference definition collision")
                definitions[key] = definition
                prop["items"] = {"$ref": f"#/$defs/{key}"}
            visit(prop)
        for key in ("items",):
            visit(node.get(key))
        for key in ("anyOf", "oneOf", "allOf"):
            for child in node.get(key, ()):
                visit(child)

    visit(result)
    # Only the new explicit Tendency port uses this map. The full source request
    # stays intact; each model choice is limited to that event's actual diff.
    if "citations" in result.get("properties", {}):
        sources = {e["evidence_id"]: e for e in context.get("evidence", ())}
        slots = result["properties"]["citations"]["properties"]
        events = context.get("structural_events", ())
        if set(slots) != {f"e{i}_id" for i in range(len(events))}:
            raise ValueError("tendency schema/event identity mismatch")
        for i, event in enumerate(events):
            eligible = sorted(k for k in event["evidence_ids"] if k in sources
                and sources[k]["provenance"] == "git_diff" and sources[k]["path"] == event["path"]
                and sources[k]["commit_id"] == event["commit_id"])
            if not eligible:
                raise ValueError("tendency event has no eligible diff citation")
            slots[f"e{i}_id"] = {"type": ["string", "null"], "enum": [None, *eligible]}
        if not domains["evidence"]:
            raise ValueError("empty tendency decision Evidence domain")
        key = definition_prefix + "evidence"
        definitions[key] = {"type": "string", "enum": domains["evidence"]}
        result["properties"]["decision_evidence_id"] = {"$ref": f"#/$defs/{key}"}
    identity = context.get("context", context)
    for name in ("repository_id", "candidate_id"):
        if name in result.get("properties", {}) and name in identity:
            result["properties"][name] = {**result["properties"][name], "enum": [identity[name]]}
    if not definitions:
        result.pop("$defs")
    return result
