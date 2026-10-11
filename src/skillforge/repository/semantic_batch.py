"""Candidate-scoped batching: transport envelopes never make semantic decisions."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from skillforge.domain.serialization import to_json_value
from skillforge.repository.semantic_permissions import semantic_permission_args

BATCH_VERSION = "semantic-batch-v2"


def request_id(request):
    from skillforge.domain.semantic_antipattern import SemanticAntiPatternRequest
    from skillforge.domain.semantic_workflow import SemanticWorkflowRequest

    context = to_json_value(request)
    if isinstance(request, (SemanticWorkflowRequest, SemanticAntiPatternRequest)):
        # Validator diagnostics do not change the closed source candidate. Only
        # terminal validated responses are published for this stable identity.
        context.pop("validation_feedback", None)
    return hashlib.sha256(
        json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def candidate_identity(request):
    context = getattr(request, "context", request)
    return context.repository_id, context.candidate_id


def supports_semantic_batch(model):
    """Opaque/custom configurations retain the original single transport."""
    from skillforge.repository.repo_snapshot_semantic_cache import (
        _ADAPTERS,
        _CachedModel,
        _complete_custom_identity,
    )

    if not callable(getattr(model, "complete_batch", None)):
        return False
    if isinstance(model, _CachedModel):
        return model.batch_supported
    module = _ADAPTERS.get(type(model).__name__)
    builtin = module and type(model).__module__ == f"skillforge.repository.{module}"
    identity = getattr(model, "semantic_cache_identity", None)
    if identity is not None and not _complete_custom_identity(identity):
        return False
    if builtin and identity is not None and identity["timeout_seconds"] != model._timeout:
        return False
    if builtin and model._runner is subprocess.run:
        executable = shutil.which("codex")
        if not executable or type(model._timeout) is not int or model._timeout <= 0:
            return False
        try:
            Path(executable).read_bytes()  # Match the cache's verifiable executable identity.
        except OSError:
            return False
        return True
    return _complete_custom_identity(identity) and (
        not builtin or identity["timeout_seconds"] == model._timeout
    )


class PrefetchedSemanticModel:
    """Consume each scoped batch result once; retries use the original single path."""

    def __init__(self, model, responses):
        self.model, self.responses = model, dict(responses)
        self.batch_attempts = 0

    def complete(self, request):
        key = request_id(request)
        if key not in self.responses:
            return self.model.complete(request)
        self.batch_attempts += 1
        response = self.responses.pop(key)
        if isinstance(response, Exception):
            raise response
        return response


def prefetch_semantic_batch(model, requests):
    requests = tuple(requests)
    if not requests or not supports_semantic_batch(model):
        return model
    keys = [request_id(r) for r in requests]
    expected = set(keys)
    if len(expected) != len(keys) or len({candidate_identity(r) for r in requests}) != len(keys):
        return model
    try:
        responses = model.complete_batch(requests)
        if not isinstance(responses, dict):
            return model
        responses = {key: value for key, value in responses.items() if key in expected}
    except (ValueError, TypeError, RuntimeError, OSError, subprocess.SubprocessError):
        return model
    return PrefetchedSemanticModel(model, responses)


def _adapter(model, request):
    # Reuse the exact single-candidate prompt/schema/parser and model selector.
    from skillforge.repository.codex_cli_antipattern import (
        CODEX_ANTIPATTERN_MODEL,
        parse_semantic_antipattern_response,
    )
    from skillforge.repository.codex_cli_semantic import CODEX_SEMANTIC_MODEL
    from skillforge.repository.codex_cli_structural import (
        CODEX_STRUCTURAL_MODEL,
        parse_semantic_structural_response,
    )
    from skillforge.repository.semantic_antipattern_prompt import (
        build_semantic_antipattern_prompt,
        semantic_antipattern_response_schema,
    )
    from skillforge.repository.semantic_structural_prompt import (
        build_semantic_structural_prompt,
        semantic_structural_response_schema,
    )

    name = type(model).__name__
    if name == "CodexCliStructuralSemanticModel":
        return (
            CODEX_STRUCTURAL_MODEL,
            build_semantic_structural_prompt,
            semantic_structural_response_schema(structured=bool(request.evidence_groups)),
            parse_semantic_structural_response,
        )
    if name == "CodexCliAntiPatternSemanticModel":
        return (
            CODEX_ANTIPATTERN_MODEL,
            build_semantic_antipattern_prompt,
            semantic_antipattern_response_schema(),
            parse_semantic_antipattern_response,
        )
    if name in {"CodexCliSemanticModel", "CodexCliTendencySemanticModel"}:
        schema = model.schema_for(request) if hasattr(model, "schema_for") else model.schema()
        return CODEX_SEMANTIC_MODEL, model.prompt, schema, model.parse
    raise NotImplementedError("no proven CLI batch adapter for this model type")


def parse_cli_batch(raw, requests, parser):
    """Salvage only uniquely proven envelopes; malformed siblings cannot erase successes."""
    if not isinstance(raw, dict) or set(raw) != {"results"} or not isinstance(raw["results"], list):
        raise ValueError("batch response must contain a results array")
    expected = {request_id(r): r for r in requests}
    counts = Counter(
        item.get("request_id")
        for item in raw["results"]
        if isinstance(item, dict) and isinstance(item.get("request_id"), str)
    )
    responses = {}
    for item in raw["results"]:
        if not isinstance(item, dict):
            continue
        key = item.get("request_id")
        if not isinstance(key, str) or key not in expected or counts[key] != 1:
            continue
        request = expected[key]
        if set(item) != {"request_id", "repository_id", "candidate_id", "response", "error"} or (
            item["repository_id"],
            item["candidate_id"],
        ) != candidate_identity(request):
            continue
        if item["error"] is not None:
            if isinstance(item["error"], str) and item["response"] is None:
                responses[key] = RuntimeError(item["error"])
            continue
        try:
            response = parser(item["response"])
            # Tendency's identity-free transport decision remains bound to this exact request.
            from skillforge.domain.semantic_tendency import (
                ScopedTendencyDecision,
                TendencyTransportDecision,
            )

            if isinstance(response, TendencyTransportDecision):
                from skillforge.repository.semantic_tendencies import tendency_request_fingerprint

                response = ScopedTendencyDecision(tendency_request_fingerprint(request), response)
            responses[key] = response
        except (ValueError, TypeError, KeyError) as error:
            from skillforge.repository.codex_cli_semantic import CodexCliSemanticError

            responses[key] = CodexCliSemanticError(
                f"batch candidate response cannot be parsed: {error}",
                raw_response=item["response"],
                invalid_response=True,
            )
    return responses


def cli_batch_payload(model, requests):
    """Build the exact existing payload without opening a model transport."""
    requests = tuple(requests)
    if not requests or not supports_semantic_batch(model):
        raise NotImplementedError("batch transport/configuration is unavailable")
    selector, _, _, parser = _adapter(model, requests[0])
    entries, envelopes, response_schemas = [], {}, []
    for request in requests:
        current_selector, prompt, schema, _ = _adapter(model, request)
        if current_selector != selector:
            raise ValueError("batch model selector mismatch")
        repository_id, candidate_id = candidate_identity(request)
        entries.append(
            {
                "request_id": request_id(request),
                "repository_id": repository_id,
                "candidate_id": candidate_id,
                "original_prompt": prompt(request),
            }
        )
        response_schemas.append(schema)
        from skillforge.repository.semantic_reference_schema import bind_response_schema

        bound_schema = bind_response_schema(
            schema, request.to_dict(), definition_prefix=request_id(request) + "_"
        )
        envelopes[request_id(request)] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["request_id", "repository_id", "candidate_id", "response", "error"],
            "properties": {
                "request_id": {"type": "string", "enum": [request_id(request)]},
                "repository_id": {"type": "string", "enum": [repository_id]},
                "candidate_id": {"type": "string", "enum": [candidate_id]},
                "response": {"anyOf": [bound_schema, {"type": "null"}]},
                "error": {"type": ["string", "null"]},
            },
        }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "object",
                "additionalProperties": False,
                "required": list(envelopes),
                "properties": envelopes,
            }
        },
    }
    # References inside a response resolve against the document root, not the envelope.
    schema["$defs"] = {}
    for envelope in envelopes.values():
        schema["$defs"].update(envelope["properties"]["response"]["anyOf"][0].pop("$defs", {}))
    prompt = (
        f"{BATCH_VERSION}: "
        "Evaluate each candidate independently using its original_prompt verbatim. "
        "Only that candidate's supplied context/Evidence may support its decision. "
        "Do not share evidence, combine candidates, change criteria or infer missing facts. "
        "Return results as an object keyed by each exact request_id; every key is required. "
        "Each value is one envelope copying request_id/repository_id/candidate_id. "
        "response follows that candidate's original schema; error is null on success. "
        "If a candidate cannot be evaluated, return response=null and a string error for it.\n"
        "SEMANTIC_BATCH:\n" + json.dumps(entries, ensure_ascii=False)
    )
    from skillforge.repository.semantic_wire import CompactBatch, compact_enabled

    compact = CompactBatch(entries, response_schemas) if compact_enabled(model) else None
    if compact is not None:
        legacy_bytes = len(prompt.encode("utf-8")) + len(
            json.dumps(schema, ensure_ascii=False).encode("utf-8")
        )
        compact_bytes = len(compact.prompt.encode("utf-8")) + len(
            json.dumps(compact.schema, ensure_ascii=False).encode("utf-8")
        )
        if compact_bytes >= legacy_bytes:
            compact = None  # Tiny batches keep the cheaper wire BEFORE any transport.
    if compact is not None:
        prompt, schema = compact.prompt, compact.schema
    return selector, prompt, schema, compact, parser


def invoke_cli_batch(model, selector, prompt, schema):
    """One governed transport; both homogeneous and compatible cohorts use it."""
    with tempfile.TemporaryDirectory(prefix="skillforge-semantic-batch-") as directory:
        root = Path(directory)
        schema_path, output_path = root / "response-schema.json", root / "response.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            *semantic_permission_args(model),
            "-m",
            selector,
            "-C",
            str(root),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-",
        ]
        from skillforge.repository.semantic_execution import run_semantic
        result = run_semantic(model,
            command,
            input=prompt,
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=model._timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Codex CLI batch exited with {result.returncode}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def parse_batch_payload(raw, requests, parser, compact):
    if compact is not None:
        raw, alias_failures = compact.normalize(raw)
        responses = parse_cli_batch(raw, requests, parser)
        from skillforge.repository.codex_cli_semantic import CodexCliSemanticError

        for key, (message, response) in alias_failures.items():
            if key in responses:
                responses[key] = CodexCliSemanticError(
                    message, raw_response=response, invalid_response=True
                )
        return responses
    # Normalize complete keyed wire into the original isolated candidate parser.
    if isinstance(raw, dict) and set(raw) == {"results"} and isinstance(raw["results"], dict):
        raw = {
            "results": [value for key, value in raw["results"].items()
                        if isinstance(value, dict) and value.get("request_id") == key]
        }
    return parse_cli_batch(raw, requests, parser)


def complete_cli_batch(model, requests):
    requests = tuple(requests)
    selector, prompt, schema, compact, parser = cli_batch_payload(model, requests)
    return parse_batch_payload(invoke_cli_batch(model, selector, prompt, schema),
                               requests, parser, compact)
