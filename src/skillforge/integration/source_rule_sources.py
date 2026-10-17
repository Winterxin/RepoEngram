"""Bounded source reading and safe, complete syntax units for rule extraction."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from skillforge.domain.project_dna import ProjectScope
from skillforge.integration.source_constraints import _selected, _tracked
from skillforge.integration.source_literal_safety import (
    contains_credential_comment,
    contains_hardcoded_token_literal,
)
from skillforge.integration.source_span_selection import FileUnits, Span, SpanUnit
from skillforge.integration.source_structure_priority import (
    SafeFunction,
    explicit_constraint_benefits,
)
from skillforge.integration.source_unit_dependencies import (
    assignment_dependencies,
    assignment_name,
)

_SENSITIVE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)", re.I
)
MAX_FILE_BYTES = 256 * 1024


@dataclass(frozen=True)
class SourceUnits:
    path: str
    sha256: str
    legacy_spans: tuple[Span, ...]
    tiers: tuple[tuple[SpanUnit, ...], ...]
    functions: tuple[SafeFunction, ...]
    dependencies: dict[str, tuple[str, ...] | None]

    def selection_file(self) -> FileUnits:
        return self.path, self.sha256, self.tiers


_NON_TOKEN_SENSITIVE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|private[_-]?key|credential)", re.I
)


def _legacy_sensitive(node: ast.AST) -> bool:
    """The pre-change lexical predicate, retained only to size legacy budgets."""
    return any(
        (isinstance(n, ast.Name) and _SENSITIVE.search(n.id))
        or (isinstance(n, ast.Attribute) and _SENSITIVE.search(n.attr))
        or (isinstance(n, ast.Constant) and isinstance(n.value, str)
            and _SENSITIVE.search(n.value))
        for n in ast.walk(node)
    )


def _safe_sensitive(node: ast.AST, source: str) -> bool:
    return (
        any(
            (isinstance(n, ast.Name) and _NON_TOKEN_SENSITIVE.search(n.id))
            or (isinstance(n, ast.Attribute) and _NON_TOKEN_SENSITIVE.search(n.attr))
            or (isinstance(n, ast.Constant) and isinstance(n.value, str)
                and _NON_TOKEN_SENSITIVE.search(n.value))
            for n in ast.walk(node)
        )
        or contains_hardcoded_token_literal(node)
        or contains_credential_comment(source)
    )


def _python_segments(
    text: str, filename: str,
) -> tuple[list[Span], list[SafeFunction], list[Span], set[str], dict]:
    tree = ast.parse(text, filename=filename)
    lines = text.splitlines()
    segments: list[Span] = []
    legacy_segments: list[Span] = []
    functions: list[SafeFunction] = []
    legacy_function_symbols: set[str] = set()
    safe_assignments: list[ast.AST] = []

    def add(node: ast.AST, symbol: str) -> None:
        segment = ast.get_source_segment(text, node) or ""
        if segment:
            if not _legacy_sensitive(node):
                legacy_segments.append((symbol, segment))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    legacy_function_symbols.add(symbol)
            if not _safe_sensitive(node, segment):
                name = assignment_name(node) if symbol == "module" else None
                segments.append((f"module.{name}" if name else symbol, segment))
                if name:
                    safe_assignments.append(node)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append((symbol, node, segment))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            if not any(_SENSITIVE.search(ast.unparse(x))
                       for x in (*node.bases, *node.decorator_list)):
                segments.append((node.name, lines[node.lineno - 1]))
                legacy_segments.append((node.name, lines[node.lineno - 1]))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(child, f"{node.name}.{child.name}")
                elif not isinstance(child, ast.ClassDef):
                    add(child, node.name)
        else:
            add(node, "module")
    return (segments, functions, legacy_segments, legacy_function_symbols,
            assignment_dependencies(tree, functions, safe_assignments))


def _byte_spans(symbol: str, segment: str) -> list[Span]:
    spans: list[Span] = []
    start = size = 0
    for index, char in enumerate(segment):
        width = len(char.encode("utf-8"))
        if size + width > 1800:
            spans.append((symbol, segment[start:index]))
            start, size = index, 0
        size += width
    if start < len(segment):
        spans.append((symbol, segment[start:]))
    return spans


def _file_units(rel: str, raw: bytes, suffix: str) -> SourceUnits:
    text = raw.decode("utf-8")
    if suffix == ".py":
        (segments, functions, legacy_segments, legacy_symbols,
         dependencies) = _python_segments(text, rel)
    else:
        segments, functions = [("document", text)], []
        legacy_segments, legacy_symbols = list(segments), set()
        dependencies = {}
    symbols = {symbol for symbol, _, _ in functions}
    benefits = explicit_constraint_benefits(functions)
    segments.sort(key=lambda item: 0 if item[0] in symbols
                  else 2 if item[0] == "module" or item[0].startswith("module.") else 1)
    legacy_segments.sort(
        key=lambda item: 0 if item[0] in legacy_symbols
        else 2 if item[0] == "module" else 1
    )
    legacy_groups: list[list[Span]] = []
    function_units: list[SpanUnit] = []
    auxiliary_units: list[SpanUnit] = []
    for symbol, segment in legacy_segments:
        spans = _byte_spans(symbol, segment)
        legacy_groups.append([span for span in spans if span[1].strip()])
    for symbol, segment in segments:
        spans = _byte_spans(symbol, segment)
        if not segment.strip():
            continue
        units = function_units if symbol in symbols else auxiliary_units
        # A complete unit retains whitespace-only interior spans as well.
        units.append((benefits.get(symbol, 0), spans))
    legacy = [group[index] for index in range(max(map(len, legacy_groups), default=0))
              for group in legacy_groups if index < len(group)]
    return SourceUnits(
        rel, hashlib.sha256(raw).hexdigest(), tuple(legacy[:32]),
        (tuple(function_units), tuple(auxiliary_units)), tuple(functions), dependencies,
    )


def collect_source_units(root: Path, scope: ProjectScope) -> tuple[SourceUnits, ...]:
    selected = _selected(root, scope)
    tracked = _tracked(root, [rel for rel, _ in selected])
    files = []
    for rel, path in selected:
        if tracked is not None and rel not in tracked:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                continue
            units = _file_units(rel, raw, path.suffix.lower())
            if units.legacy_spans:
                files.append(units)
        except (OSError, UnicodeError, SyntaxError, ValueError):
            continue
    return tuple(files)


def legacy_source_rows(files: tuple[SourceUnits, ...]) -> list[dict]:
    rows: list[dict] = []
    remaining = 64 * 1024
    for index in range(32):
        for file in files:
            if index >= len(file.legacy_spans):
                continue
            symbol, text = file.legacy_spans[index]
            size = len(text.encode("utf-8"))
            if size > remaining:
                continue
            rows.append({"source_id": f"s{len(rows) + 1}", "path": file.path,
                         "sha256": file.sha256, "symbol": symbol, "text": text})
            remaining -= size
    return rows
