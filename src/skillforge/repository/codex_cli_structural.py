"""Isolated Codex CLI adapter; semantic validation happens in the gate."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind,
    SemanticStructuralException,
    SemanticStructuralRequest,
    SemanticStructuralResponse,
)
from skillforge.repository.semantic_permissions import (
    resolve_semantic_sandbox,
    semantic_permission_args,
)
from skillforge.repository.semantic_reference_schema import bind_response_schema
from skillforge.repository.semantic_structural_prompt import (
    build_semantic_structural_prompt,
    semantic_structural_response_schema,
)

CODEX_STRUCTURAL_MODEL = "gpt-6-luna"
Runner = Callable[..., subprocess.CompletedProcess[str]]


class CodexCliStructuralError(RuntimeError):
    """Model execution or parsing failed; the candidate remains unevaluated."""


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has unexpected fields")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def parse_semantic_structural_response(value: object) -> SemanticStructuralResponse:
    structured = isinstance(value, dict) and "decision_evidence_ids" not in value
    schema = semantic_structural_response_schema(structured=structured)
    keys = set(schema["required"])
    if not structured and isinstance(value, dict) and "supporting_fact_ids" in value:
        keys.add("supporting_fact_ids")
    data = _object(value, keys, "response")
    raw_exceptions = data["exceptions"]
    if not isinstance(raw_exceptions, list):
        raise ValueError("exceptions must be an array")
    exceptions = tuple(
        SemanticStructuralException(
            _string(item["description"], "exception description"),
            _strings(item["evidence_ids"], "exception Evidence"),
        )
        for item in (
            _object(raw, {"description", "evidence_ids"}, "exception") for raw in raw_exceptions
        )
    )
    return SemanticStructuralResponse(
        _string(data["repository_id"], "repository_id"),
        _string(data["candidate_id"], "candidate_id"),
        SemanticStructuralDecisionKind(_string(data["decision"], "decision")),
        _string(data["reason"], "reason"),
        _strings(data.get("decision_evidence_ids", []), "decision Evidence"),
        _string(data["source_package"], "source_package"),
        _string(data["target_package"], "target_package"),
        _strings(data["scope_modules"], "scope modules"),
        _strings(data.get("scope_evidence_ids", []), "scope Evidence"),
        exceptions,
        _strings(data["supporting_fact_ids"], "supporting facts")
        if data.get("supporting_fact_ids") is not None
        else None,
    )


class CodexCliStructuralSemanticModel:
    def __init__(self, *, runner: Runner = subprocess.run, timeout: int = 180,
                 sandbox_mode: str | None = None) -> None:
        self._runner = runner
        self._timeout = timeout
        self._sandbox_mode = resolve_semantic_sandbox(runner, sandbox_mode)

    def complete_batch(self, requests):
        from skillforge.repository.semantic_batch import complete_cli_batch

        return complete_cli_batch(self, requests)

    def complete(self, request: SemanticStructuralRequest) -> SemanticStructuralResponse:
        with tempfile.TemporaryDirectory(prefix="skillforge-structural-") as directory:
            root = Path(directory)
            schema_path = root / "response-schema.json"
            output_path = root / "response.json"
            schema_path.write_text(
                json.dumps(
                    bind_response_schema(
                        semantic_structural_response_schema(
                            structured=bool(request.evidence_groups)
                        ),
                        request.to_dict(),
                    ),
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
                CODEX_STRUCTURAL_MODEL,
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
                    input=build_semantic_structural_prompt(request),
                    cwd=root,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise CodexCliStructuralError(f"Codex CLI failed: {error}") from error
            if result.returncode != 0:
                raise CodexCliStructuralError(f"Codex CLI exited with {result.returncode}")
            try:
                raw = json.loads(output_path.read_text(encoding="utf-8"))
                return parse_semantic_structural_response(raw)
            except (OSError, UnicodeError, ValueError, TypeError) as error:
                raise CodexCliStructuralError(
                    "Codex CLI structured response is malformed"
                ) from error
