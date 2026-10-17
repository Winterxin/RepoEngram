"""Bounded, source-anchored Python dictionary representations, without inference calls."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_reads import SourceReadSession


def _keys(tree: ast.AST) -> set[str]:
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Lookup helpers may receive field names as positional arguments.
            result.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                          and isinstance(arg.value, str) and arg.value.isidentifier())
        key = None
        if isinstance(node, ast.Subscript):
            key = node.slice
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args):
            key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result.add(key.value)
    return result


def source_representations(root: Path, scope: ProjectScope, *,
                           reads: SourceReadSession | None = None) -> str:
    """Report only literal dictionary keys assigned the same local variable.

    This is scoped current-source context, never a new historical rule or global
    schema equivalence. Dynamic expressions, constants, and unpacking are excluded.
    """
    root = root.resolve()
    reads = reads or SourceReadSession(root)
    if scope.kind not in (ProjectScopeKind.MODULE, ProjectScopeKind.FILE):
        return ""
    vocabulary: set[str] = set()
    for selector in scope.selectors[:16]:
        relative = selector.replace(".", "/") + ".py" if (
            scope.kind == ProjectScopeKind.MODULE
        ) else selector
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            if path.stat().st_size <= 262144:
                vocabulary.update(_keys(reads.read_python(path)[1]))
        except (OSError, SyntaxError, ValueError):
            continue
    if not vocabulary:
        return ""
    try:
        listed = reads.tracked_python_files()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""
    rows = []
    read_bytes = 0
    for relative in sorted(filter(None, listed))[:1024]:
        parts = Path(relative).parts
        if (any(part.startswith(".") or part in {"tests", "test", "evals"} for part in parts)
                or Path(relative).name.startswith("test_")):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            size = path.stat().st_size
            if size > 262144:
                continue
            read_bytes += size
            if read_bytes > 4194304:
                break
            content, tree = reads.read_python(path)
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or any(
                not isinstance(key, ast.Constant) or not isinstance(key.value, str)
                for key in node.keys
            ):
                continue
            groups: dict[str, list[str]] = {}
            positions: dict[str, list[int]] = {}
            for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True)):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and 0 < len(key.value) <= 64 and isinstance(value, ast.Name)
                        and len(value.id) <= 64):
                    groups.setdefault(value.id, []).append(key.value)
                    positions.setdefault(value.id, []).append(index)
            # Duplicate literal keys may overwrite an earlier assignment.
            literal_keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
            if len(literal_keys) != len(set(literal_keys)):
                continue
            for variable, keys in sorted(groups.items()):
                if len(keys) < 2 or not vocabulary.intersection(keys):
                    continue
                first, last = min(positions[variable]), max(positions[variable])
                if any(not isinstance(value, (ast.Name, ast.Constant))
                       for value in node.values[first:last + 1]):
                    # Intervening evaluations could change a global variable.
                    continue
                keys = sorted(keys, key=lambda key: (key not in vocabulary, key))[:6]
                rows.append(
                    f"- {relative}:{node.lineno} (file SHA256 "
                    f"{hashlib.sha256(content).hexdigest()}): "
                    f"keys {', '.join(repr(key) for key in sorted(keys))} receive the same "
                    f"local value {variable!r} in this dictionary construction."
                )
                if len(rows) >= 24:
                    break
            if len(rows) >= 24:
                break
        if len(rows) >= 24:
            break
    if not rows:
        return ""
    return ("\n## Current source representations (static context, not historical rules)\n\n"
            "These bounded observations come from current tracked production Python source. "
            "Inspect the cited producer and its callers to establish units, population, "
            "conditions, and compatibility before guarding existing representations. "
            "A same-value assignment at this location is not global interchangeability. "
            "Recheck the file hash if source changes; this context cannot create a new "
            "obligation or prove a task complete.\n\n" + "\n".join(rows) + "\n")
