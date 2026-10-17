"""Isolated Codex CLI adapter for evidence-bounded semantic evaluation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from skillforge.domain.evidence import Support
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticEntityReferences,
    SemanticStepClassification,
    SemanticWorkflowException,
    SemanticWorkflowRequest,
    SemanticWorkflowResponse,
    SemanticWorkflowRole,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.repository.semantic_permissions import (
    resolve_semantic_sandbox,
    semantic_permission_args,
)
from skillforge.repository.semantic_reference_schema import bind_response_schema
from skillforge.repository.semantic_workflow_prompt import (
    CLASSIFICATION_PROTOCOL,
    build_semantic_workflow_prompt,
    legacy_semantic_workflow_response_schema,
    semantic_workflow_response_schema,
)


class CodexCliSemanticError(RuntimeError):
    """A local Codex invocation or structured response failed."""

    def __init__(self, message: str, *, raw_response=None, invalid_response=False) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.invalid_response = invalid_response


Runner = Callable[..., subprocess.CompletedProcess[str]]
CODEX_SEMANTIC_MODEL = "gpt-6-luna"


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


def _nonempty_strings(value: object, label: str) -> tuple[str, ...]:
    result = _strings(value, label)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _parse_v2_ordered_steps(value: object):
    """Parse the v2 single-classification wire form into the existing domain fields."""
    parsed_steps = []
    groups = {"required": [], "optional": [], "conditional": []}
    seen_step_ids = set()
    for raw_step in _array(value, "ordered_steps"):
        item = _object(raw_step, {
            "step_id", "action", "evidence_ids", "classification_kind",
            "classification_reason", "classification_evidence_ids",
            "classification_condition",
        }, "step")
        step_id = _string(item["step_id"], "step_id")
        if step_id in seen_step_ids:
            raise ValueError("ordered_steps contains duplicate step_id")
        seen_step_ids.add(step_id)
        parsed_steps.append(SemanticWorkflowStep(
            step_id,
            _string(item["action"], "action"),
            _nonempty_strings(item["evidence_ids"], "step evidence_ids"),
        ))
        kind = _string(item["classification_kind"], "classification_kind")
        if kind not in groups:
            raise ValueError("classification_kind must be required, optional, or conditional")
        condition = item["classification_condition"]
        if kind == "conditional":
            condition = _string(condition, "classification_condition")
            if not condition:
                raise ValueError("conditional classification requires a non-empty condition")
        elif condition is not None:
            raise ValueError("required and optional classifications require a null condition")
        groups[kind].append(SemanticStepClassification(
            step_id,
            _string(item["classification_reason"], "classification_reason"),
            _nonempty_strings(
                item["classification_evidence_ids"], "classification evidence_ids"
            ),
            condition,
        ))
    return (
        tuple(parsed_steps),
        tuple(groups["required"]),
        tuple(groups["optional"]),
        tuple(groups["conditional"]),
    )


def _parse_workflow(
    raw: object, repository_id: str, candidate_id: str, confidence_score: float,
    *, protocol_v2: bool = False,
) -> SemanticWorkflowResponse:
    response_schema = (
        semantic_workflow_response_schema() if protocol_v2
        else legacy_semantic_workflow_response_schema()
    )
    workflow_schema = response_schema["properties"]["workflow"]
    data = _object(raw, set(workflow_schema["required"]), "workflow")

    def role(raw: object) -> SemanticWorkflowRole:
        item = _object(raw, {"role_id", "name", "responsibility", "evidence_ids"}, "role")
        return SemanticWorkflowRole(
            _string(item["role_id"], "role_id"),
            _string(item["name"], "role name"),
            _string(item["responsibility"], "responsibility"),
            _strings(item["evidence_ids"], "role evidence_ids"),
        )

    def step(raw: object) -> SemanticWorkflowStep:
        item = _object(raw, {"step_id", "action", "evidence_ids"}, "step")
        return SemanticWorkflowStep(
            _string(item["step_id"], "step_id"),
            _string(item["action"], "action"),
            _strings(item["evidence_ids"], "step evidence_ids"),
        )

    def classification(raw: object) -> SemanticStepClassification:
        item = _object(raw, {"step_id", "reason", "evidence_ids", "condition"}, "classification")
        condition = item["condition"]
        if condition is not None:
            condition = _string(condition, "condition")
        return SemanticStepClassification(
            _string(item["step_id"], "classification step_id"),
            _string(item["reason"], "reason"),
            _strings(item["evidence_ids"], "classification evidence_ids"),
            condition,
        )

    scope = _object(
        data["scope"],
        {"kind", "selectors", "description", "reason", "evidence_ids"},
        "scope",
    )
    refs = _object(data["references"], {"commit_ids", "module_names", "paths"}, "references")
    if protocol_v2:
        ordered_steps, required_steps, optional_steps, conditional_steps = (
            _parse_v2_ordered_steps(data["ordered_steps"])
        )
    else:
        ordered_steps = tuple(
            step(item) for item in _array(data["ordered_steps"], "ordered_steps")
        )
        required_steps = tuple(
            classification(item)
            for item in _array(data["required_steps"], "required_steps")
        )
        optional_steps = tuple(
            classification(item)
            for item in _array(data["optional_steps"], "optional_steps")
        )
        conditional_steps = tuple(
            classification(item)
            for item in _array(data["conditional_steps"], "conditional_steps")
        )
    return SemanticWorkflowResponse(
        repository_id=repository_id,
        candidate_id=candidate_id,
        name=_string(data["name"], "name"),
        name_evidence_ids=_strings(data["name_evidence_ids"], "name_evidence_ids"),
        intent=_string(data["intent"], "intent"),
        intent_evidence_ids=_strings(data["intent_evidence_ids"], "intent_evidence_ids"),
        roles=tuple(role(item) for item in _array(data["roles"], "roles")),
        ordered_steps=ordered_steps,
        required_steps=required_steps,
        optional_steps=optional_steps,
        conditional_steps=conditional_steps,
        scope=SemanticWorkflowScope(
            SemanticWorkflowScopeKind(_string(scope["kind"], "scope kind")),
            _strings(scope["selectors"], "scope selectors"),
            _string(scope["description"], "scope description"),
            _string(scope["reason"], "scope reason"),
            _strings(scope["evidence_ids"], "scope evidence_ids"),
        ),
        exceptions=tuple(
            SemanticWorkflowException(
                _string(item["description"], "exception description"),
                _strings(item["evidence_ids"], "exception evidence_ids"),
            )
            for item in (
                _object(raw, {"description", "evidence_ids"}, "exception")
                for raw in _array(data["exceptions"], "exceptions")
            )
        ),
        confidence_score=confidence_score,
        references=SemanticEntityReferences(
            _strings(refs["commit_ids"], "reference commit_ids"),
            _strings(refs["module_names"], "reference module_names"),
            _strings(refs["paths"], "reference paths"),
        ),
    )


def parse_semantic_workflow_response(value: object) -> SemanticDecisionResponse:
    """Decode the structured decision; SkillForge validation follows separately."""
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    protocol_v2 = "classification_protocol" in value
    schema = (
        semantic_workflow_response_schema() if protocol_v2
        else legacy_semantic_workflow_response_schema()
    )
    data = _object(value, set(schema["required"]), "response")
    if protocol_v2 and data["classification_protocol"] != CLASSIFICATION_PROTOCOL:
        raise ValueError(
            f"unsupported classification_protocol: {data['classification_protocol']!r}"
        )
    repository_id = _string(data["repository_id"], "repository_id")
    candidate_id = _string(data["candidate_id"], "candidate_id")
    score = data["confidence_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("confidence_score must be a number")
    support = _object(data["support"], {"supporting", "examined"}, "support")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in support.values()):
        raise ValueError("support counts must be integers")
    workflow = data["workflow"]
    return SemanticDecisionResponse(
        repository_id=repository_id,
        candidate_id=candidate_id,
        decision=SemanticDecisionKind(_string(data["decision"], "decision")),
        reason=_string(data["reason"], "reason"),
        decision_evidence_ids=_strings(data["decision_evidence_ids"], "decision_evidence_ids"),
        confidence_score=float(score),
        support=Support(support["supporting"], support["examined"]),
        workflow=(
            None
            if workflow is None
            else _parse_workflow(
                workflow, repository_id, candidate_id, float(score), protocol_v2=protocol_v2
            )
        ),
    )


class CodexCliSemanticModel:
    """Call Codex once in a fresh directory with only the supplied request."""

    def __init__(self, *, runner: Runner = subprocess.run, timeout: int = 180,
                 sandbox_mode: str | None = None) -> None:
        self._runner = runner
        self._timeout = timeout
        self._sandbox_mode = resolve_semantic_sandbox(runner, sandbox_mode)

    schema = staticmethod(semantic_workflow_response_schema)
    prompt = staticmethod(build_semantic_workflow_prompt)
    parse = staticmethod(parse_semantic_workflow_response)

    def complete_batch(self, requests):
        from skillforge.repository.semantic_batch import complete_cli_batch

        return complete_cli_batch(self, requests)

    def complete(self, request: SemanticWorkflowRequest) -> SemanticDecisionResponse:
        with tempfile.TemporaryDirectory(prefix="skillforge-semantic-") as directory:
            root = Path(directory)
            schema_path = root / "response-schema.json"
            output_path = root / "response.json"
            schema_path.write_text(
                json.dumps(
                    bind_response_schema(self.schema_for(request) if hasattr(self, "schema_for")
                                         else self.schema(), request.to_dict()), ensure_ascii=False
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
                CODEX_SEMANTIC_MODEL,
                "-C",
                str(root),
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            prompt = self.prompt(request)
            try:
                from skillforge.repository.semantic_execution import run_semantic
                result = run_semantic(self,
                    command,
                    input=prompt,
                    cwd=root,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except FileNotFoundError as error:
                raise CodexCliSemanticError("Codex CLI executable was not found") from error
            except subprocess.TimeoutExpired as error:
                raise CodexCliSemanticError(
                    f"Codex CLI timed out after {self._timeout} seconds"
                ) from error
            except OSError as error:
                raise CodexCliSemanticError(f"Codex CLI could not start: {error}") from error
            if result.returncode != 0:
                output = result.stderr or result.stdout or ""
                if "usage limit" in output.lower():
                    detail = "Codex account usage limit reached"
                else:
                    diagnostics = [
                        line.strip()
                        for line in output.splitlines()
                        if line.strip().startswith(("ERROR:", "error:", "Error:"))
                        or any(
                            marker in line.lower()
                            for marker in ("not logged", "unauthorized", "authentication failed")
                        )
                    ]
                    detail = (
                        diagnostics[-1][:300] if diagnostics else "no concise diagnostic output"
                    )
                raise CodexCliSemanticError(
                    f"Codex CLI failed (exit {result.returncode}; check login and connectivity): "
                    f"{detail}"
                )
            if not output_path.is_file():
                raise CodexCliSemanticError("Codex CLI did not create its structured output file")
            raw_text = None
            try:
                raw_text = output_path.read_text(encoding="utf-8")
                raw = json.loads(raw_text)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise CodexCliSemanticError(
                    f"Codex CLI output is malformed JSON: {error}",
                    raw_response=raw_text,
                    invalid_response=True,
                ) from error
            try:
                return self.parse(raw)
            except (TypeError, ValueError) as error:
                raise CodexCliSemanticError(
                    f"Codex CLI structured response cannot be parsed: {error}",
                    raw_response=raw,
                    invalid_response=True,
                ) from error
