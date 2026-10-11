"""Exact, local reuse of model responses published only after deterministic validation.

Hits still traverse the original evaluator and evidence gates. Custom transports must
declare their complete model/config identity; opaque injected models are never cached.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import get_type_hints

from skillforge.domain.serialization import to_json_value
from skillforge.filesystem_safety import is_link
from skillforge.productization.evolution_artifacts import _restore
from skillforge.repository.identity import repository_identity
from skillforge.repository.semantic_permissions import semantic_permission_args

CACHE_VERSION = "repo-snapshot-semantic-v1"
CUSTOM_TRANSPORT_IDENTITY_SCHEMA = "skillforge-custom-semantic-transport-v1"
_ADAPTERS = {
    "CodexCliSemanticModel": "codex_cli_semantic",
    "CodexCliAntiPatternSemanticModel": "codex_cli_antipattern",
    "CodexCliStructuralSemanticModel": "codex_cli_structural",
    "CodexCliTendencySemanticModel": "codex_cli_tendency",
}
_RESPONSE_TYPES = {
    "semantic_workflow.SemanticDecisionResponse",
    "semantic_antipattern.SemanticAntiPatternDecisionResponse",
    "semantic_structural_rule.SemanticStructuralResponse",
    "semantic_tendency.ScopedTendencyDecision",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _complete_custom_identity(identity):
    """Require an explicit configuration contract before trusting custom reuse.

    The transport owner must include every response-affecting parameter. A label
    alone cannot attest to executable, CLI, environment or permission settings.
    """
    if not isinstance(identity, dict):
        return False

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def digest(value):
        return (isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value))

    def section(name, requirements):
        value = identity.get(name)
        return isinstance(value, dict) and all(
            predicate(value.get(key)) for key, predicate in requirements.items()
        )

    timeout = identity.get("timeout_seconds")
    valid = (
        identity.get("schema") == CUSTOM_TRANSPORT_IDENTITY_SCHEMA
        and text(identity.get("model_selector"))
        and type(timeout) is int and timeout > 0
        and section("executable", {"path": text, "sha256": digest})
        and section("transport", {"name": text, "source_sha256": digest})
        and section("cli_config", {"argv_sha256": digest, "environment_sha256": digest,
                                   "config_policy": text})
        and section("permission_wrapper", {"version": text, "config_sha256": digest})
        and isinstance(identity.get("response_parameters"), dict)
        and bool(identity["response_parameters"])
    )
    if not valid:
        return False
    try:
        _json(identity)
    except (ValueError, TypeError):
        return False
    return True


def _restore_response(kind, value):
    # Evolution's strict data-only decoder handles every existing response type.
    # Tendency declarations belong to a newer optional domain module.
    if is_dataclass(kind) and kind.__module__.endswith("semantic_tendency"):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(kind)}:
            raise ValueError("invalid cached tendency fields")
        hints = get_type_hints(kind)
        result = kind(**{f.name: _restore_response(hints[f.name], value[f.name])
                         for f in fields(kind)})
        if to_json_value(result) != value:
            raise ValueError("cached tendency identity mismatch")
        return result
    return _restore(kind, value)


def _snapshot(root):
    """Bind HEAD, tree, index-visible file names and current file bytes."""
    def git(*args):
        result = subprocess.run(
            ("git", "-c", f"safe.directory={root}", *args), cwd=root,
            capture_output=True, timeout=10, check=True,
        )
        return result.stdout

    head, tree = git("rev-parse", "HEAD", "HEAD^{tree}").decode().splitlines()
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    content = []
    for name in sorted(set(names.split(b"\0")) - {b""}):
        relative = os.fsdecode(name)
        # Product output is not part of the source snapshot. Candidate requests
        # remain hashed in full, including all evidence supplied to the model.
        if Path(relative).parts[0] in {".skillforge", ".codex", ".claude", ".agents"}:
            continue
        path = root / relative
        if any(is_link(p) for p in (path, *path.parents)):
            raise ValueError("cache snapshot contains a link")
        if path.is_dir():
            raise ValueError("cache snapshot contains a submodule")
        content.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.exists() else None))
    return {"head": head, "tree": tree, "working_content": _hash(content)}


def _production_version():
    # Conservative automatic invalidation includes prompts, schemas, parsers,
    # request builders, domain identities and all deterministic gates.
    source = Path(__file__).resolve().parents[1]
    return _hash([(p.relative_to(source).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in sorted(source.rglob("*.py"))])


class RepoSnapshotSemanticCache:
    """A preparation-scoped transaction backed by cross-task JSON artifacts."""

    def __init__(self, root: Path, cache_root: Path | None = None):
        self.root = root.resolve()
        self.directory = cache_root or Path.home() / ".cache/skillforge/repo-semantic-v1"
        self.pending = {}
        self.hits = 0
        self.misses = 0
        self.context = None
        try:
            self.context = {"cache_version": CACHE_VERSION,
                            "repository_id": repository_identity(self.root),
                            "snapshot": _snapshot(self.root),
                            "production_semantic_version": _production_version()}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # An unverifiable snapshot always takes the original path.

    def wrap(self, model):
        validator = getattr(model, "semantic_cache_validator", None)
        if callable(validator):
            validator()
        def bypass(reason):
            if getattr(model, "semantic_cache_required", False):
                from skillforge.cost_control import BudgetExceeded
                raise BudgetExceeded("unexpected semantic cache bypass: " + reason)
            return model
        identity = getattr(model, "semantic_cache_identity", None)
        name = type(model).__name__
        module = _ADAPTERS.get(name)
        builtin = module and type(model).__module__ == f"skillforge.repository.{module}"
        custom = not builtin or model._runner is not subprocess.run
        if custom and not _complete_custom_identity(identity):
            return bypass("CUSTOM_RUNNER_IDENTITY_BLOCKED")
        if custom and builtin and identity["timeout_seconds"] != model._timeout:
            return bypass("CUSTOM_TIMEOUT_MISMATCH")
        if identity is None and builtin and model._runner is subprocess.run:
            executable = shutil.which("codex")
            if executable:
                try:
                    identity = {"transport": "isolated-codex-cli-v2",
                                "permission_argv": semantic_permission_args(model),
                                "executable": executable,
                                "executable_hash": hashlib.sha256(
                                    Path(executable).read_bytes()).hexdigest(),
                                "environment": {key: os.environ.get(key) for key in (
                                    "CODEX_HOME", "OPENAI_BASE_URL", "OPENAI_ORG_ID",
                                    "OPENAI_PROJECT_ID")}}
                except OSError:
                    return bypass("EXECUTABLE_UNREADABLE")
        if self.context is None or identity is None:
            return bypass("SNAPSHOT_UNVERIFIABLE" if self.context is None else "IDENTITY_MISSING")
        config = {"declared_identity": identity,
                  "evaluator_type": f"{type(model).__module__}.{name}",
                  "timeout": getattr(model, "_timeout", None)}
        if builtin:
            namespace = vars(importlib.import_module(f"skillforge.repository.{module}"))
            # Tendency shares workflow transport/model selection.
            if name == "CodexCliTendencySemanticModel":
                namespace = vars(importlib.import_module(
                    "skillforge.repository.codex_cli_semantic"))
            config["models"] = {k: v for k, v in namespace.items()
                                if k.startswith("CODEX_") and k.endswith("MODEL")}
        try:
            _json(config)
        except (ValueError, TypeError):
            return bypass("CONFIG_NOT_CANONICAL")
        return _CachedModel(self, model, config)

    def publish(self, *results):
        """Publish only candidate IDs actually emitted by the unchanged gates."""
        try:
            if self.context is None or _snapshot(self.root) != self.context["snapshot"]:
                self.pending.clear()
                return
        except (OSError, ValueError, subprocess.SubprocessError):
            self.pending.clear()
            return
        successful = set()
        for result in results:
            if result is None:
                continue
            for field in ("workflows", "anti_patterns", "accepted", "rejections"):
                for item in getattr(result, field, ()):
                    candidate_id = getattr(item, "candidate_id", None)
                    if candidate_id is None:
                        candidate_id = item.response.candidate_id
                    successful.add(candidate_id)
        selected = {key: value for key, value in self.pending.items()
                    if value[0] in successful}
        self._persist(selected)
        self.pending.clear()

    def checkpoint_candidate(self, candidate_id, model_config):
        """Persist validated responses for one candidate/config; retain other pending work."""
        selected = {key: value for key, value in self.pending.items()
                    if (value[0] == candidate_id
                        and value[1]["identity"]["model_config"] == model_config)}
        if not selected:
            return
        if not self._snapshot_is_current():
            self.pending.clear()
            return
        self._persist(selected)
        for key in selected:
            self.pending.pop(key, None)
            diagnostic = self.directory / "validation-diagnostics" / f"{key}.json"
            if not any(is_link(p) for p in diagnostic.parents):
                try:
                    diagnostic.unlink(missing_ok=True)
                except OSError:
                    pass

    def _snapshot_is_current(self):
        try:
            return self.context is not None and _snapshot(self.root) == self.context["snapshot"]
        except (OSError, ValueError, subprocess.SubprocessError):
            return False

    def _persist(self, entries, *, directory=None):
        destination = directory if directory is not None else self.directory
        for key, (_, entry) in entries.items():
            try:
                if any(is_link(p) for p in (destination, *destination.parents)):
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=destination, suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    json.dump(entry, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                try:
                    os.replace(temporary, destination / f"{key}.json")
                finally:
                    temporary.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass  # Persistence cannot change preparation decisions.


def _diagnostic_request(request):
    from skillforge.domain.semantic_antipattern import SemanticAntiPatternRequest
    from skillforge.domain.semantic_workflow import SemanticWorkflowRequest

    return isinstance(request, (SemanticWorkflowRequest, SemanticAntiPatternRequest))


class _CachedModel:
    def __init__(self, cache, model, config):
        self.cache, self.model, self.config = cache, model, config

    def _identity(self, request):
        from skillforge.repository.semantic_batch import request_id

        return {**self.cache.context, "model_config": self.config,
                "semantic_schema_version": CACHE_VERSION,
                "candidate_input_hash": request_id(request)}

    def remember_validation_feedback(self, request, feedback):
        """Keep only bounded validator diagnostics, never an invalid response."""
        if not _diagnostic_request(request) or not self.cache._snapshot_is_current():
            return
        value = str(feedback).encode("utf-8")[:512].decode("utf-8", errors="ignore")
        if not value:
            return
        identity = self._identity(request)
        key = _hash(identity)
        entry = {"identity": identity, "validation_feedback": value,
                 "feedback_hash": _hash(value)}
        self.cache._persist({key: (request.candidate_id, entry)},
                            directory=self.cache.directory / "validation-diagnostics")

    def resume_request(self, request):
        """Resume this exact failed input without treating diagnostics as evidence."""
        if not _diagnostic_request(request) or request.validation_feedback:
            return request
        identity = self._identity(request)
        path = self.cache.directory / "validation-diagnostics" / f"{_hash(identity)}.json"
        try:
            if any(is_link(p) for p in (path, *path.parents)) or path.stat().st_size > 16384:
                return request
            entry = json.loads(path.read_text(encoding="utf-8"))
            value = entry["validation_feedback"]
            if (set(entry) != {"identity", "validation_feedback", "feedback_hash"}
                    or entry["identity"] != identity or not isinstance(value, str)
                    or not 0 < len(value.encode("utf-8")) <= 512
                    or entry["feedback_hash"] != _hash(value)
                    or not self.cache._snapshot_is_current()):
                return request
            return replace(request, validation_feedback=value)
        except (OSError, ValueError, TypeError, KeyError):
            return request

    def _lookup(self, request):
        validator = getattr(self.model, "semantic_cache_validator", None)
        if callable(validator):
            validator()
        started = time.monotonic_ns()
        identity = self._identity(request)
        key = _hash(identity)
        path = self.cache.directory / f"{key}.json"
        try:
            if any(is_link(p) for p in (path, *path.parents)):
                raise ValueError("cache artifact is a link")
            if path.stat().st_size > 16 * 1_048_576:
                raise ValueError("oversized cache artifact")
            entry = json.loads(path.read_text(encoding="utf-8"))
            if (set(entry) != {"identity", "response_type", "response", "response_hash"}
                    or entry["identity"] != identity
                    or entry["response_type"] not in _RESPONSE_TYPES
                    or entry["response_hash"] != _hash(entry["response"])):
                raise ValueError("cache identity or integrity mismatch")
            module, name = entry["response_type"].split(".")
            kind = getattr(importlib.import_module(f"skillforge.domain.{module}"), name)
            response = _restore_response(kind, entry["response"])
            self.cache.hits += 1
            self._record_lookup(request, started, "hit")
            return identity, key, response
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError):
            pass
        self.cache.misses += 1
        self._record_lookup(request, started, "miss")
        return identity, key, None

    def _record_lookup(self, request, started, state):
        ledger = getattr(self.model, "cost_ledger", None)
        if ledger is None:
            return
        data = _json(to_json_value(request)).encode("utf-8")
        ended = time.monotonic_ns()
        ledger.append({"kind": "semantic-cache", "counter": self.cache.hits + self.cache.misses,
                       "start_ns": started, "end_ns": ended, "seconds": (ended - started) / 1e9,
                       "payload_sha256": hashlib.sha256(data).hexdigest(),
                       "payload_bytes": len(data),
                       "cache": state, "bypass_reason": None, "usage": None})

    def _remember(self, request, identity, key, response):
        context = getattr(request, "context", request)
        response_type = f"{type(response).__module__.rsplit('.', 1)[-1]}.{type(response).__name__}"
        if response_type in _RESPONSE_TYPES:
            try:
                value = to_json_value(response)
                entry = {"identity": identity, "response_type": response_type,
                         "response": value, "response_hash": _hash(value)}
                # A retry replaces the pending unvalidated response for this exact input.
                self.cache.pending[key] = (context.candidate_id, entry)
            except (ValueError, TypeError):
                pass  # Invalid raw responses still reach the original gate/retry path.
        return response

    def checkpoint_candidate(self, candidate_id):
        self.cache.checkpoint_candidate(candidate_id, self.config)

    @property
    def batch_supported(self):
        from skillforge.repository.semantic_batch import supports_semantic_batch

        return supports_semantic_batch(self.model)

    def complete(self, request):
        identity, key, response = self._lookup(request)
        if response is not None:
            return response
        resumed = self.resume_request(request)
        return self._remember(request, identity, key, self.model.complete(resumed))

    def complete_batch(self, requests):
        from skillforge.repository.semantic_batch import request_id

        responses, missing = {}, []
        for request in requests:
            identity, key, response = self._lookup(request)
            if response is None:
                missing.append((self.resume_request(request), identity, key))
            else:
                responses[request_id(request)] = response
        if missing:
            try:
                batch = self.model.complete_batch(tuple(item[0] for item in missing))
                expected = {request_id(item[0]) for item in missing}
                if not isinstance(batch, dict):
                    return responses
                batch = {key: value for key, value in batch.items() if key in expected}
            except (ValueError, TypeError, RuntimeError, OSError, subprocess.SubprocessError):
                return responses
            for request, identity, key in missing:
                rid = request_id(request)
                if rid in batch:
                    response = batch[rid]
                    responses[rid] = (response if isinstance(response, Exception) else
                                      self._remember(request, identity, key, response))
        return responses
