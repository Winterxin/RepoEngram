"""Pure AST signals for prioritizing bounded current-source context units."""

from __future__ import annotations

import ast
from collections.abc import Sequence

SafeFunction = tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str]


def _function_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.AST, ...]:
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return tuple(found)


def explicit_constraint_benefits(
    functions: Sequence[SafeFunction],
) -> dict[str, int]:
    """Return bounded structural benefit per safe function, independent of its name.

    Conditions, match/try/except branches, assertions, and raises are signals. Loops,
    boolean operators, comments, and identifier words are not. A single unambiguous
    direct same-file call edge can add a small bonus when either endpoint has a signal.
    """
    nodes = [_function_nodes(function) for _, function, _ in functions]
    base: list[int] = []
    by_name: dict[str, list[int]] = {}
    for index, (_, function, _) in enumerate(functions):
        by_name.setdefault(function.name, []).append(index)
        items = nodes[index]
        boundary_count = sum(
            isinstance(item, (ast.If, ast.IfExp, ast.Match)) for item in items
        )
        boundary_count += sum(
            isinstance(item, (ast.Try, ast.ExceptHandler)) for item in items
        )
        boundary_count += 2 * sum(isinstance(item, (ast.Assert, ast.Raise)) for item in items)
        base.append(min(32, boundary_count))

    edges: list[set[int]] = [set() for _ in functions]
    for caller, items in enumerate(nodes):
        for item in items:
            if not isinstance(item, ast.Call):
                continue
            if isinstance(item.func, ast.Name):
                callee_name = item.func.id
            elif (
                isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id in {"self", "cls"}
            ):
                callee_name = item.func.attr
            else:
                continue
            targets = by_name.get(callee_name, ())
            if len(targets) == 1 and targets[0] != caller:
                edges[caller].add(targets[0])

    related: list[set[int]] = [set() for _ in functions]
    for caller, callees in enumerate(edges):
        for callee in callees:
            if base[caller] and base[callee] == 0:
                related[callee].add(caller)
            if base[callee] and base[caller] == 0:
                related[caller].add(callee)

    result: dict[str, int] = {}
    for index, (symbol, _, _) in enumerate(functions):
        benefit = base[index] + min(2, len(related[index]))
        result[symbol] = max(result.get(symbol, 0), benefit)
    return result
