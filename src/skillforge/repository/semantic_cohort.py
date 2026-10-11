"""One compatible cold transport; original candidates and gates remain isolated."""

from __future__ import annotations

import json
import subprocess

from skillforge.repository.repo_snapshot_semantic_cache import _ADAPTERS, _CachedModel
from skillforge.repository.semantic_batch import (
    PrefetchedSemanticModel,
    _adapter,
    candidate_identity,
    cli_batch_payload,
    invoke_cli_batch,
    parse_batch_payload,
    request_id,
    supports_semantic_batch,
)
from skillforge.repository.semantic_wire import CompactBatch, compact_enabled

COHORT_VERSION = "semantic-cohort-v2"
MAX_COHORT_CANDIDATES = 11  # Default; only an exact declared budget plan can expand it.


def _original(model):
    return model.model if isinstance(model, _CachedModel) else model


def _compatible(groups):
    originals = [_original(model) for model, _ in groups.values()]
    first = originals[0]
    for model in originals:
        module = _ADAPTERS.get(type(model).__name__)
        if (
            not module
            or type(model).__module__ != f"skillforge.repository.{module}"
            or not supports_semantic_batch(model)
            or not compact_enabled(model)
        ):
            return False
        if (
            model._runner is not first._runner
            or model._timeout != first._timeout
            or model._sandbox_mode != first._sandbox_mode
            or getattr(model, "effect_budget", None) is not getattr(first, "effect_budget", None)
            or getattr(model, "cost_ledger", None) is not getattr(first, "cost_ledger", None)
            or getattr(model, "semantic_cache_identity", None)
            != getattr(first, "semantic_cache_identity", None)
        ):
            return False
        if (
            model._runner is not subprocess.run
            and getattr(model, "semantic_cache_identity", {})
            .get("response_parameters", {})
            .get("semantic_cohort")
            != COHORT_VERSION
        ):
            return False  # Custom/frozen transports must explicitly attest this path.
    return True


def _size(prompt, schema):
    return len(prompt.encode("utf-8")) + len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))


def prefetch_semantic_cohort(groups):
    """Return proven prefetched channels, or None before any incompatible dispatch.

    Cache lookup and publication remain owned by each original channel wrapper.
    A full cohort error never buys unbounded replacement batches: only cached or
    returned siblings are prefetched, and existing bounded single repair follows.
    Fatal budget exceptions are deliberately outside this local failure handler.
    """
    active = {
        name: (model, tuple(requests)) for name, (model, requests) in groups.items() if requests
    }
    requests = [request for _, items in active.values() for request in items]
    limit = MAX_COHORT_CANDIDATES
    if active:
        budget = getattr(_original(next(iter(active.values()))[0]), "effect_budget", None)
        shape = getattr(budget, "cold_shape", None)
        if shape is not None:
            counts = {name: len(items) for name, (_, items) in groups.items()}
            budget.validate_shape(counts)
            limit = sum(shape.values())  # Explicit exact-shape plan; default remains eleven.
    if (
        len(active) < 2
        or len(requests) > limit
        or len({request_id(request) for request in requests}) != len(requests)
        or len({candidate_identity(request) for request in requests}) != len(requests)
        or not _compatible(active)
    ):
        return None
    responses, pending, entries, schemas, parsers = {}, {}, [], [], {}
    selector = None
    # Validate/build every original request/schema before lookup or dispatch.
    for name, (model, items) in active.items():
        adapter = _original(model)
        for request in items:
            current, prompt, schema, parser = _adapter(adapter, request)
            if selector is not None and current != selector:
                return None
            selector = current
            parsers[name] = parser
        responses[name] = {}
    for name, (model, items) in active.items():
        for request in items:
            identity = key = response = None
            if isinstance(model, _CachedModel):
                identity, key, response = model._lookup(request)
            if response is not None:
                responses[name][request_id(request)] = response
                continue
            if isinstance(model, _CachedModel):
                request = model.resume_request(request)
            pending[request_id(request)] = (name, model, request, identity, key)
            _, prompt, schema, _ = _adapter(_original(model), request)
            repository, candidate = candidate_identity(request)
            entries.append(
                {
                    "request_id": request_id(request),
                    "repository_id": repository,
                    "candidate_id": candidate,
                    "original_prompt": prompt(request),
                }
            )
            schemas.append(schema)
    if not entries:
        return {
            name: PrefetchedSemanticModel(model, responses[name])
            for name, (model, _) in active.items()
        }
    try:
        compact = CompactBatch(entries, schemas)
        separate_size = 0
        for _name, (model, items) in active.items():
            missing = tuple(request for request in items if request_id(request) in pending)
            if missing:
                _, prompt, schema, _, _ = cli_batch_payload(_original(model), missing)
                separate_size += _size(prompt, schema)
        if _size(compact.prompt, compact.schema) >= separate_size:
            return None  # Never launch a more expensive combined payload.
    except (ValueError, TypeError, RuntimeError, OSError, subprocess.SubprocessError):
        return None  # No transport happened: keep the proven channel batch path.
    try:
        dispatch = _original(next(iter(active.values()))[0])
        raw = invoke_cli_batch(dispatch, selector, compact.prompt, compact.schema)
        for name, (model, items) in active.items():
            missing = tuple(request for request in items if request_id(request) in pending)
            if not missing:
                continue
            parsed = parse_batch_payload(raw, missing, parsers[name], compact)
            for request in missing:
                rid = request_id(request)
                if rid not in parsed:
                    continue
                response = parsed[rid]
                _, _, _, identity, key = pending[rid]
                if isinstance(model, _CachedModel) and not isinstance(response, Exception):
                    response = model._remember(request, identity, key, response)
                responses[name][rid] = response
    except (ValueError, TypeError, RuntimeError, OSError, subprocess.SubprocessError):
        pass  # Preserve proven hits/siblings; repair uses the original shared budget.
    return {
        name: PrefetchedSemanticModel(model, responses[name]) for name, (model, _) in active.items()
    }
