"""Bounded same-file assignment dependencies, using an already parsed AST."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence

MAX_ASSIGNMENT_DEPENDENCIES = 8


def assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def loaded_global_names(node: ast.AST) -> set[str]:
    """Ignore nested scopes and conservatively exclude all locally bound names."""
    pending = list(ast.iter_child_nodes(node))
    reads: set[str] = set()
    bound: set[str] = set()
    while pending:
        child = pending.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
            continue
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, ast.Name):
            (reads if isinstance(child.ctx, ast.Load) else bound).add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split('.')[0] for alias in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        pending.extend(ast.iter_child_nodes(child))
    return reads - bound


def module_binding_counts(tree: ast.Module) -> Counter:
    """Count declarations/rebindings without resolving or executing imports."""
    writes = Counter()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            writes[node.name] += 1
        else:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    writes[child.id] += 1
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                writes.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
    return writes


def assignment_dependencies(
    tree: ast.Module,
    functions: Sequence[tuple[str, ast.AST, str]],
    safe_assignments: Sequence[ast.AST],
) -> dict[str, tuple[str, ...] | None]:
    """Resolve safe unique module bindings only; overflow is an explicit skip.

    No imports, function-call graph, cross-file reads or runtime evaluation. The
    caller screens every assignment with its existing credential predicate.
    """
    writes = module_binding_counts(tree)
    assignments = {
        name: node for node in safe_assignments
        if (name := assignment_name(node)) and writes[name] == 1
    }
    direct = {name: loaded_global_names(node) & assignments.keys()
              for name, node in assignments.items()}
    result: dict[str, tuple[str, ...] | None] = {}
    subjects = [*functions, *((f"module.{name}", node, "")
                              for name, node in assignments.items())]
    for symbol, node, _text in subjects:
        pending = sorted(loaded_global_names(node) & assignments.keys(), reverse=True)
        seen: set[str] = set()
        ordered: list[str] = []
        while pending and len(seen) <= MAX_ASSIGNMENT_DEPENDENCIES:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            ordered.append(f"module.{name}")
            pending.extend(sorted(direct[name] - seen, reverse=True))
        if ordered:
            result[symbol] = (tuple(ordered) if len(seen) <= MAX_ASSIGNMENT_DEPENDENCIES
                              else None)
    return result
