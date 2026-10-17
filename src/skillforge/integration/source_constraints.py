"""Bounded, opt-in observations of current source behavior (never policy)."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from skillforge.domain.agent_integration import CodeAgentTarget, digest
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRouteResult
from skillforge.integration.knowledge_trace import record as trace_record
from skillforge.integration.source_observation_selection import (
    COLD_PACKAGE_BYTES,
    MAX_OBSERVATIONS,
    MAX_PACKAGE_BYTES,
    SourceObservation,
    package_sizes,
    ranked_observations,
)

_SECRET = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)", re.I
)
_BAD_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "tests",
    "test",
    "eval",
    "evals",
    "__pycache__",
}


@dataclass(frozen=True, slots=True)
class SourceConstraintFile:
    relative_path: str
    content: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SourceConstraintExportPlan:
    plan_id: str
    repository_id: str
    target: CodeAgentTarget
    scope: ProjectScope
    files: tuple[SourceConstraintFile, ...]


def _safe_path(relative: str) -> bool:
    p = Path(relative)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in relative
        and not any(part.lower() in _BAD_PARTS for part in p.parts)
        and not any(part.lower().startswith(".env") or _SECRET.search(part) for part in p.parts)
    )


def _selected(root: Path, scope: ProjectScope, trace=None) -> list[tuple[str, Path]]:
    if scope.kind not in (
        ProjectScopeKind.FILE,
        ProjectScopeKind.DIRECTORY,
        ProjectScopeKind.MODULE,
    ):
        raise ValueError("source fallback requires explicit FILE, DIRECTORY, or MODULE scope")
    found = []

    def safe_file(path: Path) -> tuple[str, Path] | None:
        try:
            rel_item = path.relative_to(root).as_posix()
            if not _safe_path(rel_item):
                trace_record(trace, "source_selection", "PATH_EXCLUDED", path=rel_item)
                return None
            current = root
            for part in Path(rel_item).parts:
                current = current / part
                if current.is_symlink():
                    trace_record(trace, "source_selection", "SYMLINK_EXCLUDED", path=rel_item)
                    return None
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            return rel_item, resolved
        except (OSError, ValueError):
            return None

    for selector in scope.selectors[:16]:
        rel = (
            selector.replace(".", "/") + ".py"
            if scope.kind == ProjectScopeKind.MODULE
            else selector
        )
        if not _safe_path(rel):
            trace_record(trace, "source_selection", "PATH_EXCLUDED", path=rel)
            continue
        candidate = root / rel
        # Reject symlinks in every component before resolving.
        cur = root
        symlink = False
        for part in Path(rel).parts:
            cur = cur / part
            if cur.is_symlink():
                symlink = True
                break
        if symlink:
            trace_record(trace, "source_selection", "SYMLINK_EXCLUDED", path=rel)
            continue
        try:
            path = candidate.resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError):
            trace_record(trace, "source_selection", "PATH_UNAVAILABLE", path=rel)
            continue
        if scope.kind == ProjectScopeKind.DIRECTORY:
            if not path.is_dir():
                continue
            candidates = []
            visited = 0
            for directory, dirs, names in os.walk(path, followlinks=False):
                visited += 1
                if visited > 256:
                    break
                base = Path(directory)
                dirs[:] = [
                    name
                    for name in dirs
                    if not (base / name).is_symlink()
                    and name.lower() not in _BAD_PARTS
                    and not name.lower().startswith(".env")
                    and not _SECRET.search(name)
                ]
                for name in sorted(names):
                    if name.endswith(".py"):
                        safe = safe_file(base / name)
                        if safe is not None:
                            candidates.append(safe)
                            if len(found) + len(candidates) >= 16:
                                break
                if len(found) + len(candidates) >= 16:
                    break
            found.extend(candidates)
            if len(found) >= 16:
                break
            continue
        elif path.is_file():
            candidates = [safe_file(path)]
        else:
            continue
        for candidate_item in candidates:
            if candidate_item is not None:
                found.append(candidate_item)
                if len(found) >= 16:
                    break
    unique = {rel: path for rel, path in found}
    return sorted(unique.items())[:16]


def _tracked(root: Path, relatives: list[str]) -> set[str] | None:
    try:
        check = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if check.returncode != 0 or Path(check.stdout.strip()).resolve() != root.resolve():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *relatives],
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return set()
    return {item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\0") if item}


def _sensitive(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and _SECRET.search(node.id)) or (
            isinstance(node, ast.Attribute) and _SECRET.search(node.attr)
        ):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _SECRET.search(node.value):
                return True
    return False


def _collect_observations(relative: str, text: str, tree: ast.Module, sha: str, trace=None):
    lines = text.splitlines()
    observations = []

    def snippet(node: ast.AST) -> str:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        return "\n".join(lines[start - 1 : min(end, start + 4)])[:600]

    def effect(node: ast.stmt) -> bool:
        return isinstance(
            node, (ast.Return, ast.Raise, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)
        ) and (not isinstance(node, ast.Expr) or isinstance(node.value, ast.Call))

    def record(row, ancestry, structured, *, decorator=False):
        # Check the complete emitted text too: context labels and source comments
        # need the same protection as the effect's AST, without dropping siblings.
        if _SECRET.search(row):
            trace_record(trace, "observation", "SENSITIVE_CONTEXT_EXCLUDED", path=relative)
            return
        conditional = any(item.startswith(("if ", "elif ", "else ")) for item in ancestry)
        exceptional = any(item.startswith(("except ", "finally ", "try else "))
                          for item in ancestry)
        priority = 0 if exceptional else 1 if conditional else 2 if structured else 3
        legacy = None if structured else 1 if decorator or not conditional else 0
        observations.append(SourceObservation(row, priority, legacy))

    def expression(node):
        value = ast.unparse(node)
        return value if len(value.encode("utf-8")) <= 160 else None

    def visit(stmts, ancestry=(), structured=False):
        for node in stmts:
            if isinstance(node, ast.If):
                if _sensitive(node.test):
                    trace_record(trace, "observation", "SENSITIVE_PREDICATE_EXCLUDED",
                           path=relative, line=node.lineno)
                    continue
                pred = ast.unparse(node.test)
                # Predicates are rendered literally; they are never evaluated.
                visit(node.body, ancestry + (f"if {pred}",), structured)
                if node.orelse:
                    label = "else" if not isinstance(node.orelse[0], ast.If) else "elif"
                    visit(node.orelse, ancestry + (f"{label} (if {pred} was false)",), structured)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    if not _sensitive(deco):
                        record(f"{relative}:{deco.lineno} [{sha}] decorator on "
                               f"{node.name}: `{ast.unparse(deco)}`", ancestry,
                               structured, decorator=True)
                visit(node.body, ancestry + (f"function {node.name}",), structured)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, ancestry, structured)
            elif isinstance(node, ast.Try):
                label = f"try at line {node.lineno} (execution unknown)"
                visit(node.body, ancestry + (label,), True)
                for handler in node.handlers:
                    caught = expression(handler.type) if handler.type else "bare handler"
                    if caught is None:
                        continue
                    binding = f" as {handler.name}" if handler.name else ""
                    visit(handler.body, ancestry + (label,
                          f"except {caught}{binding} (selection unknown)"), True)
                visit(node.orelse, ancestry + (label,
                      "try else (requires normal try completion)"), True)
                visit(node.finalbody, ancestry + (label,
                      "finally (unwinding context, execution unknown)"), True)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                target, iterable = expression(node.target), expression(node.iter)
                if target is None or iterable is None:
                    continue
                prefix = "async for" if isinstance(node, ast.AsyncFor) else "for"
                label = f"{prefix} {target} in {iterable} (iteration unknown)"
                visit(node.body, ancestry + (label,), True)
                visit(node.orelse, ancestry + (label,
                      "loop else (requires completion without break)"), True)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                managers = [expression(item.context_expr) for item in node.items]
                if any(item is None for item in managers):
                    continue
                label = "with " + ", ".join(managers) + " (entry/exit unknown)"
                visit(node.body, ancestry + (label,), True)
            elif effect(node):
                source = snippet(node)
                if _sensitive(node):
                    trace_record(trace, "observation", "SENSITIVE_EFFECT_EXCLUDED",
                           path=relative, line=node.lineno)
                    continue
                context = " / ".join(ancestry) if ancestry else "unconditional"
                row = f"{relative}:{node.lineno} [{sha}] context `{context}`; effect `{source}`"
                record(row, ancestry, structured)
            # while/match/exception-group bodies remain unsupported, not flattened.

    visit(tree.body)
    return observations


def _observations(relative: str, text: str, tree: ast.Module, sha: str) -> list[str]:
    return [row.text for row in sorted(
        _collect_observations(relative, text, tree, sha), key=lambda row: row.priority,
    )]


def plan_current_source_fallback(
    route: SkillRouteResult, target: CodeAgentTarget, *, repository_root: Path, scope: ProjectScope,
    trace=None,
) -> SourceConstraintExportPlan | None:
    if not isinstance(route, SkillRouteResult) or not isinstance(target, CodeAgentTarget):
        raise ValueError("a route and supported target are required")
    if not isinstance(scope, ProjectScope) or scope.kind not in (
        ProjectScopeKind.FILE,
        ProjectScopeKind.DIRECTORY,
        ProjectScopeKind.MODULE,
    ):
        raise ValueError("an explicit FILE, DIRECTORY, or MODULE scope is required")
    if route.matches:
        trace_record(trace, "source_fallback", "HISTORICAL_ROUTE_AVAILABLE",
                     count=len(route.matches))
        return None
    root = Path(repository_root).resolve(strict=True)
    groups = []
    legacy_conditional, legacy_unconditional = [], []
    legacy_hashes = {}
    legacy_open = True
    hashes: dict[str, str] = {}
    total_bytes = 0
    selected = _selected(root, scope, trace)
    trace_record(trace, "source_selection", "FILES_SELECTED", count=len(selected))
    tracked = _tracked(root, [relative for relative, _ in selected])
    for relative, path in selected:
        if tracked is not None and relative not in tracked:
            trace_record(trace, "source_read", "UNTRACKED_FILE_EXCLUDED", path=relative)
            continue
        try:
            if path.stat().st_size > 256 * 1024:
                trace_record(trace, "source_read", "FILE_SIZE_LIMIT", path=relative)
                continue
            raw = path.read_bytes()
            if total_bytes + len(raw) > 2 * 1024 * 1024:
                trace_record(trace, "source_read", "TOTAL_READ_LIMIT", path=relative)
                break
            total_bytes += len(raw)
            text = raw.decode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()
            tree = ast.parse(text, filename=relative)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            trace_record(trace, "source_read", "READ_OR_PARSE_FAILED", path=relative,
                   object_id=type(exc).__name__)
            continue
        trace_record(trace, "source_read", "PARSED", path=relative, object_id=sha,
                     input_bytes=len(raw))
        # Sensitive observations are filtered locally. Keep the old whole-file
        # eligibility solely for the same-source legacy size allowance.
        legacy_file_safe = not _sensitive(tree)
        rows = _collect_observations(relative, text, tree, sha, trace)
        trace_record(trace, "observation",
                     "CANDIDATES_AVAILABLE" if rows else "NO_SAFE_OBSERVATIONS",
                     path=relative, count=len(rows))
        if rows:
            hashes[relative] = sha
            groups.append((relative, rows))
        if legacy_open and legacy_file_safe:
            legacy = [row for row in rows if row.legacy_priority is not None]
            if legacy:
                legacy_hashes[relative] = sha
            legacy_conditional.extend(row.text for row in legacy if row.legacy_priority == 0)
            legacy_unconditional.extend(row.text for row in legacy if row.legacy_priority == 1)
            if len(legacy_conditional) + len(legacy_unconditional) >= 96:
                legacy_open = False
    legacy_rows = (legacy_conditional + legacy_unconditional)[:MAX_OBSERVATIONS]
    baseline = (_export_observations(route, target, scope, legacy_rows, legacy_hashes)
                if legacy_rows else None)
    budgets = (tuple(min(value, MAX_PACKAGE_BYTES) for value in package_sizes(baseline))
               if baseline else (COLD_PACKAGE_BYTES, COLD_PACKAGE_BYTES))
    trace_record(trace, "budget", "LEGACY_ALLOWANCE" if baseline else "COLD_ALLOWANCE",
           input_bytes=budgets[0], output_bytes=budgets[1])
    selected_rows, selected_hashes = [], {}
    plan = None
    for attempted, (relative, row) in enumerate(ranked_observations(groups)):
        if attempted >= 96 or len(selected_rows) >= MAX_OBSERVATIONS:
            trace_record(trace, "budget", "OBSERVATION_COUNT_LIMIT", count=len(selected_rows))
            break
        if len(row.encode("utf-8")) > MAX_PACKAGE_BYTES:
            trace_record(trace, "budget", "OBSERVATION_SIZE_LIMIT", path=relative)
            continue
        candidate_hashes = {**selected_hashes, relative: hashes[relative]}
        candidate = _export_observations(
            route, target, scope, [*selected_rows, row], candidate_hashes,
        )
        if all(size <= cap for size, cap in zip(package_sizes(candidate), budgets, strict=True)):
            selected_rows.append(row)
            selected_hashes = candidate_hashes
            plan = candidate
            trace_record(trace, "observation", "SELECTED", path=relative,
                   object_id=hashlib.sha256(row.encode("utf-8")).hexdigest())
        else:
            trace_record(trace, "budget", "PACKAGE_SIZE_LIMIT", path=relative,
                   object_id=hashlib.sha256(row.encode("utf-8")).hexdigest())
    return plan


def _export_observations(route, target, scope, observations, hashes):
    scope_text = f"{scope.kind.value}: {', '.join(scope.selectors)}"
    body = (
        """---
name: current-source-observation
description: Advisory observations of existing source behavior in an explicitly selected scope.
---

# Current source observations

"""
        + (
            "These observations describe current source only. They are not historical workflow, "
            "normative policy, or formal verification requirements, and do not freeze bugs. "
            "A requested task change may change an existing condition. Predicates are shown as "
            "source expressions and were not evaluated. Observe existing behavior where "
            "applicable, preserve valid zero and missing-value distinctions, and keep "
            "independent facts separate. Do not invent policy, extend guards, or clear "
            "independent facts."
        )
        + "\n\n## Explicit scope\n\n"
        + f"`{scope_text}`\n\n## Observations\n\n"
        + "\n".join(f"- {row}" for row in observations)
        + "\n\nSee [references/evidence.md](references/evidence.md).\n"
    )
    evidence = (
        "# Current source observation evidence\n\n"
        + "\n".join(f"- `{path}` SHA256 `{sha}`" for path, sha in sorted(hashes.items()))
        + "\n"
    )
    identity = json.dumps(
        {
            "repository_id": route.repository_id,
            "target": target.value,
            "scope": scope_text,
            "file_hashes": hashes,
            "observations": observations,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    plan_id = hashlib.sha256(identity.encode()).hexdigest()
    marker = {
        "schema": "skillforge-current-source-observation-v1",
        "source_kind": "CURRENT_SOURCE_OBSERVATION",
        "plan_id": plan_id,
        "repository_id": route.repository_id,
        "target": target.value,
        "scope": scope_text,
        "file_hashes": hashes,
        "file_hashes_export": {
            "SKILL.md": digest(body),
            "references/evidence.md": digest(evidence),
        },
    }
    contents = {
        "SKILL.md": body,
        "references/evidence.md": evidence,
        ".skillforge.json": json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    }
    files = tuple(SourceConstraintFile(p, c, digest(c)) for p, c in sorted(contents.items()))
    return SourceConstraintExportPlan(plan_id, route.repository_id, target, scope, files)
