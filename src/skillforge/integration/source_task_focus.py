"""Select source functions explicitly named by a bounded task description."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass

SafeFunction = tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef, str]
FunctionKey = tuple[str, str]

_MAX_TASK_BYTES = 8192
_MAX_ANCHORS = 8
_MAX_RELATED = 16
_IDENT = re.compile(r"[A-Za-z_]\w*", re.UNICODE)
_QUALIFIED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", re.UNICODE)


@dataclass(frozen=True)
class SourceFocus:
    anchors: frozenset[FunctionKey]
    related: frozenset[FunctionKey]

    @property
    def selected(self) -> frozenset[FunctionKey]:
        return self.anchors | self.related


def _bounded_task(task_description: str) -> str:
    if not isinstance(task_description, str) or not task_description:
        return ""
    return task_description.encode("utf-8")[:_MAX_TASK_BYTES].decode("utf-8", errors="ignore")


def _quoted(task: str, name: str) -> bool:
    return re.search(r"`" + re.escape(name) + r"`", task) is not None


def _followed_by_call(task: str, name: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(name) + r"\s*\(", task) is not None


def _token_present(task: str, name: str) -> bool:
    return re.search(
        r"(?<![\w.])" + re.escape(name) + r"(?!\w|\.[A-Za-z_])", task
    ) is not None


def _leaf(symbol: str) -> str:
    return symbol.rsplit(".", 1)[-1]


def _explicit_matches(bindings: Sequence[FunctionKey], task: str) -> set[int]:
    if not task:
        return set()
    qualified_refs = set(_QUALIFIED.findall(task))
    identifiers = set(_IDENT.findall(task))
    by_leaf: dict[str, list[int]] = {}
    for index, (_, symbol) in enumerate(bindings):
        by_leaf.setdefault(_leaf(symbol), []).append(index)

    matches: set[int] = set()
    qualified_matches: set[int] = set()
    for ref in qualified_refs:
        candidates = [
            index
            for index, (_, symbol) in enumerate(bindings)
            if ref == symbol or ref.endswith("." + symbol)
        ]
        if candidates:
            longest = max(len(bindings[index][1]) for index in candidates)
            best = [index for index in candidates if len(bindings[index][1]) == longest]
            if len(best) == 1:
                qualified_matches.add(best[0])

    for index, (_, symbol) in enumerate(bindings):
        leaf = _leaf(symbol)
        # A qualified reference can identify a method even if another class has
        # a method with the same leaf name.
        if index in qualified_matches:
            matches.add(index)
            continue
        if len(by_leaf[leaf]) != 1 or leaf not in identifiers:
            continue
        identifier_style = "_" in leaf or any(char.isupper() for char in leaf[1:])
        if identifier_style or _quoted(task, leaf) or _followed_by_call(task, leaf):
            if _token_present(task, leaf):
                matches.add(index)
    return matches


def explicit_source_bindings(
    bindings: Sequence[FunctionKey], task_description: str,
) -> frozenset[FunctionKey]:
    """Match only unique explicit source names; no lexical relevance inference."""
    indexes = _explicit_matches(bindings, _bounded_task(task_description))
    if len(indexes) > _MAX_ANCHORS:
        return frozenset()
    return frozenset(bindings[index] for index in indexes)


def _body_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef):
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            # Keep the declaration visible for lexical binding, but never enter
            # the nested scope while resolving this function's calls.
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _bound_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef, nodes: Sequence[ast.AST],
) -> set[str]:
    names = {
        arg.arg
        for arg in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    names.update(arg.arg for arg in (function.args.vararg, function.args.kwarg) if arg)
    globals_: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            globals_.update(node.names)
    return names - globals_


def _class_name(symbol: str) -> str | None:
    return symbol.split(".", 1)[0] if "." in symbol else None


def _call_target(
    node: ast.Call,
    path: str,
    symbol: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    bound_names: set[str],
    assigned_names: set[str],
    by_file_symbol: dict[tuple[str, str], list[int]],
) -> list[int]:
    if isinstance(node.func, ast.Name):
        if node.func.id in bound_names:
            return []
        return by_file_symbol.get((path, node.func.id), [])
    if not (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"self", "cls"}
        and _class_name(symbol) is not None
    ):
        return []
    receiver = node.func.value.id
    args = (*function.args.posonlyargs, *function.args.args)
    if not args or args[0].arg != receiver or receiver in assigned_names:
        return []
    class_symbol = f"{_class_name(symbol)}.{node.func.attr}"
    return by_file_symbol.get((path, class_symbol), [])


def _targets(functions: Sequence[SafeFunction]) -> dict[int, set[int]]:
    """Resolve only local direct calls whose target is unambiguous."""
    by_file_symbol: dict[tuple[str, str], list[int]] = {}
    for index, (path, symbol, _, _) in enumerate(functions):
        by_file_symbol.setdefault((path, symbol), []).append(index)

    edges: dict[int, set[int]] = {i: set() for i in range(len(functions))}
    for caller, (path, symbol, function, _) in enumerate(functions):
        nodes = list(_body_nodes(function))
        local_names = _bound_names(function, nodes)
        assigned_names = {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        }
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            possible = _call_target(
                node, path, symbol, function, local_names, assigned_names, by_file_symbol
            )
            possible = [target for target in possible if target != caller]
            if len(possible) == 1:
                edges[caller].add(possible[0])
    return edges


def source_task_focus(
    functions: Sequence[SafeFunction], task_description: str,
) -> SourceFocus:
    """Find unique explicit function references, plus unambiguous one-hop calls.

    Input functions are assumed to have been screened by the caller. This helper
    performs no file access, ranking, or model calls.
    """
    anchors = explicit_source_bindings([(p, s) for p, s, _, _ in functions], task_description)
    if not anchors:
        return SourceFocus(anchors, frozenset())
    indexes = {i for i, (path, symbol, _, _) in enumerate(functions)
               if (path, symbol) in anchors}

    edges = _targets(functions)
    related_indexes: set[int] = set()
    for anchor in indexes:
        related_indexes.update(edges[anchor])
        related_indexes.update(i for i, targets in edges.items() if anchor in targets)
    related_indexes.difference_update(indexes)
    related_keys = {
        (functions[i][0], functions[i][1]) for i in related_indexes
    }
    related_keys.difference_update(anchors)
    if len(related_keys) > _MAX_RELATED:
        return SourceFocus(anchors, frozenset())
    return SourceFocus(anchors, frozenset(related_keys))
