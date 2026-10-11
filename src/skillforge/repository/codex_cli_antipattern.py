"""Isolated Codex CLI adapter for evidence-bounded Anti-pattern decisions."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
    SemanticAntiPatternException,
    SemanticAntiPatternRequest,
    SemanticAntiPatternResponse,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.repository.semantic_antipattern_prompt import (
    build_semantic_antipattern_prompt,
    semantic_antipattern_response_schema,
)
from skillforge.repository.semantic_permissions import (
    resolve_semantic_sandbox,
    semantic_permission_args,
)
from skillforge.repository.semantic_reference_schema import bind_response_schema

CODEX_ANTIPATTERN_MODEL = "gpt-6-luna"
Runner = Callable[..., subprocess.CompletedProcess[str]]


class CodexCliAntiPatternError(RuntimeError):
    """A local Codex invocation or structured response failed."""


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must be an object with keys: {', '.join(sorted(keys))}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _parse_anti_pattern(raw: object) -> SemanticAntiPatternResponse:
    schema = semantic_antipattern_response_schema()["properties"]["anti_pattern"]
    data = _object(raw, set(schema["required"]), "anti_pattern")
    scope = _object(
        data["scope"],
        {"kind", "selectors", "description", "reason", "evidence_ids"},
        "scope",
    )
    exceptions = tuple(
        SemanticAntiPatternException(
            _string(item["description"], "exception description"),
            _strings(item["evidence_ids"], "exception evidence_ids"),
        )
        for item in (
            _object(raw_item, {"description", "evidence_ids"}, "exception")
            for raw_item in _array(data["exceptions"], "exceptions")
        )
    )
    return SemanticAntiPatternResponse(
        name=_string(data["name"], "name"),
        name_evidence_ids=_strings(data["name_evidence_ids"], "name_evidence_ids"),
        description=_string(data["description"], "description"),
        description_evidence_ids=_strings(
            data["description_evidence_ids"], "description_evidence_ids"
        ),
        avoid_behavior=_string(data["avoid_behavior"], "avoid_behavior"),
        avoid_behavior_evidence_ids=_strings(
            data["avoid_behavior_evidence_ids"], "avoid_behavior_evidence_ids"
        ),
        negative_evidence_summary=_string(
            data["negative_evidence_summary"], "negative_evidence_summary"
        ),
        negative_evidence_ids=_strings(data["negative_evidence_ids"], "negative_evidence_ids"),
        scope=SemanticAntiPatternScope(
            SemanticAntiPatternScopeKind(_string(scope["kind"], "scope kind")),
            _strings(scope["selectors"], "scope selectors"),
            _string(scope["description"], "scope description"),
            _string(scope["reason"], "scope reason"),
            _strings(scope["evidence_ids"], "scope evidence_ids"),
        ),
        exceptions=exceptions,
        evidence_ids=_strings(data["evidence_ids"], "evidence_ids"),
    )


def parse_semantic_antipattern_response(value: object) -> SemanticAntiPatternDecisionResponse:
    """Parse schema shape and primitive types; the abstracter validates semantics."""
    schema = semantic_antipattern_response_schema()
    data = _object(value, set(schema["required"]), "response")
    score = data["confidence_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("confidence_score must be a number")
    anti_pattern = data["anti_pattern"]
    return SemanticAntiPatternDecisionResponse(
        repository_id=_string(data["repository_id"], "repository_id"),
        candidate_id=_string(data["candidate_id"], "candidate_id"),
        decision=SemanticAntiPatternDecisionKind(_string(data["decision"], "decision")),
        reason=_string(data["reason"], "reason"),
        decision_evidence_ids=_strings(data["decision_evidence_ids"], "decision_evidence_ids"),
        confidence_score=float(score),
        anti_pattern=None if anti_pattern is None else _parse_anti_pattern(anti_pattern),
    )


class CodexCliAntiPatternSemanticModel:
    """Call Codex once in a fresh directory with only the supplied request."""

    def __init__(self, *, runner: Runner = subprocess.run, timeout: int = 180,
                 sandbox_mode: str | None = None) -> None:
        self._runner = runner
        self._timeout = timeout
        self._sandbox_mode = resolve_semantic_sandbox(runner, sandbox_mode)

    def complete_batch(self, requests):
        from skillforge.repository.semantic_batch import complete_cli_batch

        return complete_cli_batch(self, requests)

    def complete(self, request: SemanticAntiPatternRequest) -> SemanticAntiPatternDecisionResponse:
        with tempfile.TemporaryDirectory(prefix="skillforge-antipattern-") as directory:
            root = Path(directory)
            schema_path = root / "response-schema.json"
            output_path = root / "response.json"
            schema_path.write_text(
                json.dumps(
                    bind_response_schema(semantic_antipattern_response_schema(), request.to_dict()),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                *semantic_permission_args(self),
                "-m",
                CODEX_ANTIPATTERN_MODEL,
                "-C",
                str(root),
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            try:
                from skillforge.repository.semantic_execution import run_semantic
                result = run_semantic(self,
                    command,
                    input=build_semantic_antipattern_prompt(request),
                    cwd=root,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except FileNotFoundError as error:
                raise CodexCliAntiPatternError("Codex CLI executable was not found") from error
            except subprocess.TimeoutExpired as error:
                raise CodexCliAntiPatternError(
                    f"Codex CLI timed out after {self._timeout} seconds"
                ) from error
            except OSError as error:
                raise CodexCliAntiPatternError(f"Codex CLI could not start: {error}") from error
            if result.returncode != 0:
                output = result.stderr or result.stdout or ""
                if "usage limit" in output.lower():
                    detail = "Codex account usage limit reached"
                else:
                    error_matches = list(re.finditer(r"(?im)^\s*error:", output))
                    excerpt = output[error_matches[-1].start() :] if error_matches else output
                    detail = re.sub(r"\s+", " ", excerpt).strip()[:1200]
                    if not detail:
                        detail = "no diagnostic output"
                raise CodexCliAntiPatternError(
                    f"Codex CLI failed (exit {result.returncode}; check login and connectivity): "
                    f"{detail}"
                )
            if not output_path.is_file():
                raise CodexCliAntiPatternError(
                    "Codex CLI did not create its structured output file"
                )
            try:
                raw = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise CodexCliAntiPatternError(
                    f"Codex CLI output is malformed JSON: {error}"
                ) from error
            try:
                return parse_semantic_antipattern_response(raw)
            except (TypeError, ValueError) as error:
                raise CodexCliAntiPatternError(
                    f"Codex CLI structured response cannot be parsed: {error}"
                ) from error
