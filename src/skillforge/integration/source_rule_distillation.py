"""Evidence-validated, opt-in distillation of current source behavior."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillforge.domain.agent_integration import CodeAgentTarget, digest
from skillforge.domain.project_dna import ProjectScope
from skillforge.domain.routing import SkillRouteResult
from skillforge.integration.knowledge_trace import KnowledgeTrace
from skillforge.integration.source_constraints import (
    SourceConstraintExportPlan,
    SourceConstraintFile,
)
from skillforge.integration.source_evidence_binding import (
    MAX_EVIDENCE_ITEMS,
    MAX_QUOTE_CHARACTERS,
    MAX_RULE_FIELD_CHARACTERS,
    MAX_RULES,
    EvidenceBindingError,
    bind_evidenced_entrypoints,
    bind_source_lines,
)
from skillforge.integration.source_rule_coverage import (
    coverage_report,
    requested_bindings,
    rule_bindings,
)
from skillforge.integration.source_rule_delivery import budget_source_rules
from skillforge.integration.source_rule_sources import collect_source_units, legacy_source_rows
from skillforge.integration.source_span_selection import select_distributed_spans
from skillforge.integration.source_task_focus import source_task_focus
from skillforge.integration.source_task_relevance import source_task_relevance

LINE_REFERENCE_CONTRACT = (
    "Ranges split losslessly into 500-character quotes; at most 16 resulting quotes/rule. "
)

_SOURCE_RULE_INSTRUCTIONS = (
    "JSON; advisory CURRENT_SOURCE_RULE_CANDIDATEs, not history/policy/proof. "
    "Prefer cross-entrypoint boundaries with evidenced lookup/delegation preconditions, "
    "not method inventories. Keep conditions/exceptions/order. "
    "Entrypoints need same-source_id evidence. "
)

def _normalize_newlines(value: str) -> str:
    """Normalize line endings only; preserve all other source and quote characters."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


class SourceRuleValidationError(ValueError):
    """The model response could not be fully bound to supplied source evidence."""


class ModelProtocol(Protocol):
    def complete(self, request: dict) -> dict: ...


def _resolve_codex_executable(*, which=shutil.which, windows: bool = os.name == "nt") -> str:
    """Resolve the native CLI, avoiding Windows PowerShell/cmd shims for Popen."""
    executable = which("codex.exe") or which("codex") or "codex"
    if not windows or Path(executable).suffix.lower() not in {".cmd", ".ps1"}:
        return executable
    shim = Path(executable).absolute()
    relative_binary = Path(
        "node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/"
        "vendor/x86_64-pc-windows-msvc/bin/codex.exe"
    )
    for parent in shim.parents:
        candidate = parent / relative_binary
        if candidate.is_file():
            return str(candidate)
    return executable


@dataclass(frozen=True, slots=True)
class _Request:
    repository_id: str
    candidate_id: str
    sources: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "repository_id": self.repository_id,
            "candidate_id": self.candidate_id,
            "sources": list(self.sources),
        }


def source_rule_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rules"],
        "properties": {
            "rules": {
                "type": "array",
                "maxItems": MAX_RULES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "condition",
                        "action",
                        "exceptions",
                        "entrypoints",
                        "ordering",
                        "evidence",
                    ],
                    "properties": {
                        "condition": {"type": "string", "maxLength": MAX_RULE_FIELD_CHARACTERS},
                        "action": {"type": "string", "maxLength": MAX_RULE_FIELD_CHARACTERS},
                        "exceptions": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": MAX_RULE_FIELD_CHARACTERS},
                        },
                        "ordering": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": MAX_RULE_FIELD_CHARACTERS},
                        },
                        "entrypoints": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 12,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source_id", "symbol"],
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "symbol": {"type": "string", "maxLength": 160},
                                },
                            },
                        },
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_EVIDENCE_ITEMS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source_id", "quote"],
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "quote": {"type": "string", "maxLength": MAX_QUOTE_CHARACTERS},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def source_rule_prompt(request: _Request) -> str:
    return (
        _SOURCE_RULE_INSTRUCTIONS
        + "Evidence: exact supplied quotes, not paraphrases.\nRequest:\n"
        + json.dumps(request.to_dict(), ensure_ascii=False)
    )


def _line_reference_schema(sources: tuple[dict, ...]) -> dict:
    schema = source_rule_response_schema()
    rule = schema["properties"]["rules"]["items"]
    rule["required"].remove("entrypoints")
    del rule["properties"]["entrypoints"]
    rule["properties"]["evidence"]["maxItems"] = 12
    evidence = schema["properties"]["rules"]["items"]["properties"]["evidence"]["items"]
    evidence["required"] = ["source_id", "start_line", "end_line"]
    evidence["properties"] = {
        "source_id": {"type": "string", "enum": [source["source_id"] for source in sources]},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
    }
    return schema


def _line_reference_prompt(request: _Request) -> str:
    sources = []
    for source in request.sources:
        numbered = "\n".join(
            f"{index}: {line}" for index, line in enumerate(source["text"].splitlines(), 1)
        )
        sources.append({**source, "text": numbered})
    wire = _Request(request.repository_id, request.candidate_id, tuple(sources))
    return (
        _SOURCE_RULE_INSTRUCTIONS
        + "Evidence: source_id, inclusive 1-based start_line/end_line. "
        + LINE_REFERENCE_CONTRACT +
        "\nRequest:\n"
        + json.dumps(wire.to_dict(), ensure_ascii=False)
    )


def _bind_line_references(raw: object, sources: tuple[dict, ...]) -> object:
    try:
        return bind_source_lines(raw, sources)
    except EvidenceBindingError as exc:
        raise SourceRuleValidationError(str(exc)) from exc


class CodexCliSourceRuleModel:
    """Single-call gpt-6-luna adapter using Codex CLI's bounded semantic transport."""

    def __init__(self, *, runner=None, timeout: int = 150, sandbox_mode: str | None = None):
        from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel

        self._transport = CodexCliSemanticModel(
            **({} if runner is None else {"runner": runner}),
            timeout=timeout,
            sandbox_mode=sandbox_mode,
        )

    def complete(self, request: _Request | dict) -> dict:
        import subprocess
        import tempfile

        from skillforge.repository.semantic_permissions import semantic_permission_args

        request_obj = (
            request
            if isinstance(request, _Request)
            else _Request(
                request["repository_id"], request["candidate_id"], tuple(request["sources"])
            )
        )
        with tempfile.TemporaryDirectory(prefix="skillforge-source-rules-") as directory:
            root = Path(directory)
            schema = root / "schema.json"
            output = root / "response.json"
            schema.write_text(
                json.dumps(_line_reference_schema(request_obj.sources)), encoding="utf-8"
            )
            executable = (
                _resolve_codex_executable()
                if self._transport._runner is subprocess.run
                else "codex"
            )
            command = [
                executable,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                *semantic_permission_args(self._transport),
                "-m",
                "gpt-6-luna",
                "-C",
                str(root),
                "--output-schema",
                str(schema),
                "-o",
                str(output),
                "-",
            ]
            from skillforge.repository.semantic_execution import run_semantic

            try:
                result = run_semantic(
                    self._transport,
                    command,
                    input=_line_reference_prompt(request_obj),
                    cwd=root,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=self._transport._timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"source rule model call failed: {exc}") from exc
            if result.returncode or not output.is_file():
                raise RuntimeError("source rule model call did not produce structured output")
            bound = _bind_line_references(
                json.loads(output.read_text(encoding="utf-8")), request_obj.sources
            )
            try:
                return bind_evidenced_entrypoints(bound, request_obj.sources)
            except EvidenceBindingError as exc:
                raise SourceRuleValidationError(str(exc)) from exc


def _chunks(
    root: Path, scope: ProjectScope, *, task_description: str = "",
    focus_task: str = "",
    trace: KnowledgeTrace | None = None,
) -> tuple[list[dict], dict[str, str]]:
    files = collect_source_units(root, scope)
    old_rows = legacy_source_rows(files)
    functions = [(file.path, symbol, node, segment) for file in files
                 for symbol, node, segment in file.functions]
    relevance = source_task_relevance(functions, task_description)
    focus = source_task_focus(functions, focus_task)
    if focus.anchors:
        relevance = {key: 2 if key in focus.anchors else 1 for key in focus.selected}
    fixed_request = _Request("r" * 64, "c" * 64, tuple(old_rows))

    def prompt_sizes(prompt: str) -> tuple[int, int]:
        escaped = json.dumps(prompt, ensure_ascii=True).encode("utf-8")
        return len(prompt.encode("utf-8")), len(escaped)

    source_prompt_budget = prompt_sizes(source_rule_prompt(fixed_request))
    line_prompt_budget = prompt_sizes(_line_reference_prompt(fixed_request))

    def fits_serialized(candidate: list[tuple[str, str, str, str]]) -> bool:
        sources = tuple({"source_id": f"s{index}", "path": path, "sha256": sha,
                         "symbol": symbol, "text": text}
                        for index, (path, sha, symbol, text) in enumerate(candidate, 1))
        request = _Request("r" * 64, "c" * 64, sources)
        return all(size <= limit for size, limit in zip(
            (*prompt_sizes(source_rule_prompt(request)),
             *prompt_sizes(_line_reference_prompt(request))),
            (*source_prompt_budget, *line_prompt_budget), strict=True,
        ))

    selection_files = [file.selection_file() for file in files]
    selection_limits = dict(max_spans=len(old_rows),
        max_spans_per_file=32,
        raw_byte_budget=sum(len(row["text"].encode("utf-8")) for row in old_rows),
        serialized_fits=fits_serialized, relevance=relevance,
        dependencies={(file.path, symbol): names for file in files
                      for symbol, names in file.dependencies.items()},
    )
    chosen, skipped = select_distributed_spans(
        selection_files, **selection_limits, allowed_symbols=focus.selected or None,
    )
    selected_keys = {(path, symbol) for path, _, symbol, _ in chosen}
    focus_applied = bool(focus.anchors) and focus.anchors <= selected_keys
    if focus.anchors and not focus_applied:
        # A helper-only excerpt must not replace an oversized requested entrypoint.
        selection_limits["relevance"] = source_task_relevance(functions, task_description)
        chosen, skipped = select_distributed_spans(selection_files, **selection_limits)
    rows = [{"source_id": f"s{index}", "path": path, "sha256": sha,
             "symbol": symbol, "text": text}
            for index, (path, sha, symbol, text) in enumerate(chosen, 1)]
    if trace is not None:
        trace.record("source_focus", "APPLIED" if focus_applied else
                     "ANCHOR_EXCEEDS_BUDGET" if focus.anchors else "NO_EXPLICIT_ANCHOR",
                     count=len(focus.anchors))
        trace.record("source_selection", "TASK_RANKED" if any(relevance.values())
                     else "NO_LEXICAL_SIGNAL", count=sum(score > 0 for score in relevance.values()))
        for path, symbol in dict.fromkeys((path, symbol) for path, _, symbol, _ in chosen):
            trace.record("source_selection", "SELECTED", path=path, object_id=symbol,
                         count=relevance.get((path, symbol), 0))
        for path, symbol, reason in skipped:
            trace.record("source_selection", reason.upper(), path=path, object_id=symbol)
    return rows, {row["path"]: row["sha256"] for row in rows}


def _validate(raw: object, sources: list[dict]) -> list[dict]:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"rules"}
        or not isinstance(raw["rules"], list)
        or len(raw["rules"]) > MAX_RULES
    ):
        raise SourceRuleValidationError("response must contain only a bounded rules array")
    by_id = {s["source_id"]: s for s in sources}
    accepted = []
    for rule in raw["rules"]:
        keys = {"condition", "action", "exceptions", "entrypoints", "ordering", "evidence"}
        if not isinstance(rule, dict) or set(rule) != keys:
            raise SourceRuleValidationError("rule fields are invalid")
        for key in ("condition", "action"):
            if (not isinstance(rule[key], str) or not rule[key].strip()
                    or len(rule[key]) > MAX_RULE_FIELD_CHARACTERS):
                raise SourceRuleValidationError(f"invalid {key}")
        for key in ("exceptions", "ordering"):
            if (
                not isinstance(rule[key], list)
                or len(rule[key]) > 8
                or any(not isinstance(x, str) or len(x) > MAX_RULE_FIELD_CHARACTERS
                       for x in rule[key])
            ):
                raise SourceRuleValidationError(f"invalid {key}")
        eps, evs = rule["entrypoints"], rule["evidence"]
        if (
            not isinstance(eps, list)
            or not eps
            or len(eps) > 12
            or not isinstance(evs, list)
            or not evs
            or len(evs) > MAX_EVIDENCE_ITEMS
        ):
            raise SourceRuleValidationError("entrypoints and evidence are required")
        quotes: dict[str, list[str]] = {}
        for ev in evs:
            if not isinstance(ev, dict) or set(ev) != {"source_id", "quote"}:
                raise SourceRuleValidationError("invalid evidence shape")
            sid, q = ev["source_id"], ev["quote"]
            if (
                sid not in by_id
                or not isinstance(q, str)
                or not q
                or len(q) > MAX_QUOTE_CHARACTERS
                or _normalize_newlines(q) not in _normalize_newlines(by_id[sid]["text"])
            ):
                raise SourceRuleValidationError(
                    "evidence quote is not an exact supplied source span"
                )
            quotes.setdefault(sid, []).append(q)
        for ep in eps:
            if not isinstance(ep, dict) or set(ep) != {"source_id", "symbol"}:
                raise SourceRuleValidationError("invalid entrypoint shape")
            sid, symbol = ep["source_id"], ep["symbol"]
            if (
                sid not in by_id
                or not isinstance(symbol, str)
                or len(symbol) > 160
                or symbol != by_id[sid]["symbol"]
            ):
                raise SourceRuleValidationError("entrypoint is not a supplied source entity")
            if not quotes.get(sid):
                raise SourceRuleValidationError(
                    "each entrypoint requires an independent source quote"
                )
        accepted.append(rule)
    return accepted


def plan_current_source_rules(
    route: SkillRouteResult,
    target: CodeAgentTarget,
    *,
    repository_root: Path,
    scope: ProjectScope,
    model: ModelProtocol | None = None,
    trace: KnowledgeTrace | None = None,
    task_aware_selection: bool = False,
) -> SourceConstraintExportPlan | None:
    """Return an evidence-bound advisory export, only for an enabled unmatched route."""
    if not isinstance(route, SkillRouteResult) or not isinstance(target, CodeAgentTarget):
        raise ValueError("route and supported target are required")
    if route.matches:
        return None
    if not isinstance(scope, ProjectScope):
        raise ValueError("an explicit scope is required")
    root = Path(repository_root).resolve(strict=True)
    if trace is not None:
        trace.record("source_selection", "TASK_EXPERIMENT_ENABLED"
                     if task_aware_selection is True else "PREVIOUS_SELECTION_RETAINED")
    sources, hashes = _chunks(
        root, scope, task_description=(route.request.task_description
                                      if task_aware_selection is True else ""),
        focus_task=route.request.task_description, trace=trace,
    )
    if not sources:
        return None
    request = _Request(
        route.repository_id,
        hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        tuple(sources),
    )
    response = (model or CodexCliSourceRuleModel()).complete(request.to_dict())
    rules = _validate(response, sources)
    if not rules:
        return None
    # Stale-source guard immediately before export.
    for rel, sha in hashes.items():
        try:
            current = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError as exc:
            raise SourceRuleValidationError("source changed during extraction") from exc
        if current != sha:
            raise SourceRuleValidationError("source changed during extraction")
    preferred = requested_bindings(sources, route.request.task_description)
    delivery = budget_source_rules(rules, sources, scope, preferred_bindings=preferred)
    coverage = coverage_report(preferred, rule_bindings(rules, sources),
                               delivery.retained_rule_indexes)
    body, evdoc = delivery.body, delivery.evidence
    if trace is not None:
        trace.record("source_delivery", "BOUNDED", count=len(delivery.retained_rule_indexes),
                     output_bytes=len(body.encode("utf-8")) + len(evdoc.encode("utf-8")))
        for index, reason in delivery.dropped_rules:
            trace.record("source_delivery", reason.upper(), object_id=str(index))
        for label in ("missing_from_generation", "dropped_by_budget"):
            for path, symbol in coverage[label]:
                trace.record("source_coverage", label.upper(), path=path, object_id=symbol)
    if not delivery.retained_rule_indexes:
        return None
    marker = {
        "schema": "skillforge-current-source-rule-candidate-v1",
        "source_kind": "CURRENT_SOURCE_RULE_CANDIDATE",
        "repository_id": route.repository_id,
        "target": target.value,
        "scope": f"{scope.kind.value}: {', '.join(scope.selectors)}",
        "source_hashes": hashes,
        "response": response,
        "delivery": {"max_document_bytes": 8192,
                     "retained_rule_indexes": list(delivery.retained_rule_indexes),
                     "dropped_rules": [list(item) for item in delivery.dropped_rules]},
        "file_hashes": {"SKILL.md": digest(body), "references/evidence.md": digest(evdoc)},
    }
    if preferred:
        marker["entrypoint_coverage"] = coverage
    ident = json.dumps(
        {"marker": marker, "response": response}, sort_keys=True, separators=(",", ":")
    )
    plan_id = hashlib.sha256(ident.encode()).hexdigest()
    marker["plan_id"] = plan_id
    contents = {
        "SKILL.md": body,
        "references/evidence.md": evdoc,
        ".skillforge.json": json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    }
    files = tuple(SourceConstraintFile(p, c, digest(c)) for p, c in sorted(contents.items()))
    return SourceConstraintExportPlan(plan_id, route.repository_id, target, scope, files)
