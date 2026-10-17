"""Bounded lexical ranking of safe source units against an external task."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence

from skillforge.domain.routing import task_tokens

SafeFunction = tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef, str]

_MAX_QUERY_BYTES = 8192
_MAX_QUERY_TOKENS = 256
_MAX_AST_NODES_PER_FUNCTION = 20_000
_STOPWORDS = frozenset("""
and as assert async await be break class continue def del elif else except finally for
from global if import in is lambda nonlocal not or pass raise return try while with yield
add change create ensure fix implement improve refactor update use
""".split())


def _tokens(value: str) -> set[str]:
    return {token for token in task_tokens(value) if token not in _STOPWORDS}


def _body_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(function))
    while stack and len(nodes) < _MAX_AST_NODES_PER_FUNCTION:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _condition_roots(nodes: Sequence[ast.AST]) -> list[ast.AST]:
    roots: list[ast.AST] = []
    for node in nodes:
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            roots.append(node.test)
        elif isinstance(node, ast.comprehension):
            roots.extend(node.ifs)
        elif isinstance(node, ast.match_case) and node.guard is not None:
            roots.append(node.guard)
    return roots


def _features(
    symbol: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    weights: dict[str, int] = {}

    def add(value: str, weight: int) -> None:
        for token in _tokens(value):
            weights[token] = max(weights.get(token, 0), weight)

    add(symbol, 3)
    nodes = _body_nodes(function)
    for node in nodes:
        if isinstance(node, ast.Name):
            add(node.id, 2)
        elif isinstance(node, ast.Attribute):
            add(node.attr, 2)
        elif isinstance(node, ast.arg):
            add(node.arg, 2)
        elif isinstance(node, ast.keyword) and node.arg:
            add(node.arg, 2)

    # Short string literals count only when they occur in explicit conditions;
    # large presentation templates and docstrings never become task features.
    for root in _condition_roots(nodes):
        for node in ast.walk(root):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip()
                if value and len(value) <= 64 and len(_tokens(value)) <= 8:
                    add(value, 3)
    return weights


def source_task_relevance(
    functions: Sequence[SafeFunction], task_description: str,
) -> dict[tuple[str, str], int]:
    """Score already-safe functions by distinct task-token overlap and corpus rarity.

    The task is clipped to 8 KiB and 256 unique tokens before scoring. This helper
    is deterministic and local: it has no repository, model, or oracle access.
    A zero score means no lexical evidence, not that a function is inapplicable.
    """
    if not isinstance(task_description, str) or not task_description:
        query: set[str] = set()
    else:
        bounded = task_description.encode("utf-8")[:_MAX_QUERY_BYTES].decode(
            "utf-8", errors="ignore"
        )
        query = set(sorted(_tokens(bounded))[:_MAX_QUERY_TOKENS])
    if not query:
        return {(path, symbol): 0 for path, symbol, _, _ in functions}

    rows = [
        ((path, symbol), _features(symbol, function))
        for path, symbol, function, _ in functions
    ]
    frequency = Counter(token for _, features in rows for token in features)
    corpus_size = max(1, len(rows))
    scores: dict[tuple[str, str], int] = {}
    for key, features in rows:
        score = 0
        for token in query.intersection(features):
            documents = frequency[token]
            rarity = 1 + (4 * (corpus_size - documents + 1)) // (corpus_size + 1)
            score += features[token] * rarity
        scores[key] = max(scores.get(key, 0), score)
    return scores
