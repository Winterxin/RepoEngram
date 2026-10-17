"""Contextual checks for hard-coded credential literals in Python source."""

from __future__ import annotations

import ast
import io
import re
import tokenize

_TOKEN_LABEL = re.compile(r"token", re.I)
_COMMENT_CREDENTIAL = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)", re.I
)


def _label_contains_token(value: str | None) -> bool:
    return bool(value and _TOKEN_LABEL.search(value))


def _static_string(node: ast.AST) -> bool:
    """True for a literal string value, rather than a value computed at runtime."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _target_contains_token(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return _label_contains_token(target.id)
    if isinstance(target, ast.Attribute):
        return _label_contains_token(target.attr)
    if isinstance(target, ast.Subscript):
        key = target.slice
        return (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and _label_contains_token(key.value)
        )
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_contains_token(item) for item in target.elts)
    return False


def contains_hardcoded_token_literal(node: ast.AST) -> bool:
    """Find literal strings stored in token-labelled fields or token getters.

    A mention of ``token`` in an identifier or explanatory text is harmless. A
    literal string assigned to a token field, passed as a token keyword, or
    returned from a token-named getter is treated as a credential and excluded.
    """
    for item in ast.walk(node):
        if isinstance(item, ast.Assign):
            if (
                any(_target_contains_token(target) for target in item.targets)
                and _static_string(item.value)
            ):
                return True
        elif isinstance(item, ast.AnnAssign):
            if (
                _target_contains_token(item.target)
                and item.value is not None
                and _static_string(item.value)
            ):
                return True
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*item.args.posonlyargs, *item.args.args]
            defaults = [None] * (len(args) - len(item.args.defaults)) + list(item.args.defaults)
            kw_defaults = item.args.kw_defaults
            if any(_label_contains_token(arg.arg) and value is not None and _static_string(value)
                   for arg, value in zip(args, defaults, strict=True)):
                return True
            if any(_label_contains_token(arg.arg) and value is not None and _static_string(value)
                   for arg, value in zip(item.args.kwonlyargs, kw_defaults, strict=True)):
                return True
            if _label_contains_token(item.name) and any(
                isinstance(child, ast.Return) and child.value is not None
                and _static_string(child.value)
                for child in ast.walk(item)
            ):
                return True
        elif isinstance(item, ast.Call):
            if any(_label_contains_token(keyword.arg) and _static_string(keyword.value)
                   for keyword in item.keywords):
                return True
        elif isinstance(item, ast.Dict):
            if any(isinstance(key, ast.Constant) and isinstance(key.value, str)
                   and _label_contains_token(key.value) and _static_string(value)
                   for key, value in zip(item.keys, item.values, strict=True)):
                return True
    return False


def contains_credential_comment(source: str) -> bool:
    """Exclude comments that could carry hard-coded credentials in source excerpts."""
    try:
        return any(
            token.type == tokenize.COMMENT and _COMMENT_CREDENTIAL.search(token.string)
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
        )
    except (IndentationError, tokenize.TokenError):
        return False
