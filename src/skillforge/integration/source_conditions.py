"""Expose current boolean producers to prevent expanding historical trigger conditions."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_reads import SourceReadSession
from skillforge.integration.source_representations import _keys


def _local_nodes(node: ast.AST):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield current
        stack.extend(reversed(list(ast.iter_child_nodes(current))))


def _boolean_expression(value, bindings, assignments, boolean_functions, seen=frozenset()):
    if isinstance(value, ast.Compare):
        return True
    if isinstance(value, ast.Constant):
        return isinstance(value.value, bool)
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id == "bool" or value.func.id in boolean_functions
    if isinstance(value, ast.BoolOp):
        return all(_boolean_expression(item, bindings, assignments, boolean_functions, seen)
                   for item in value.values)
    if (isinstance(value, ast.Name) and value.id not in seen
            and assignments.get(value.id) == 1 and value.id in bindings):
        return _boolean_expression(bindings[value.id], bindings, assignments,
                                   boolean_functions, seen | {value.id})
    return False


def source_conditions(root: Path, scope: ProjectScope, *,
                      reads: SourceReadSession | None = None) -> str:
    """Read bounded scoped source; report expressions, never infer their truth."""
    root = root.resolve()
    reads = reads or SourceReadSession(root)
    if scope.kind not in (ProjectScopeKind.MODULE, ProjectScopeKind.FILE):
        return ""
    rows = []
    for selector in scope.selectors[:16]:
        relative = selector.replace(".", "/") + ".py" if (
            scope.kind == ProjectScopeKind.MODULE
        ) else selector
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            if path.stat().st_size > 262144:
                continue
            content, tree = reads.read_python(path)
        except (OSError, SyntaxError, ValueError):
            continue
        keys = _keys(tree)
        boolean_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            and isinstance(node.returns, ast.Name) and node.returns.id == "bool"
        }
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(function.name) > 96:
                continue
            # Do not resolve another function's bindings or conditional reassignments.
            bindings = {}
            assignments = {}
            for statement in function.body:
                for node in _local_nodes(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        assignments[node.id] = assignments.get(node.id, 0) + 1
            for statement in function.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = statement.value
                for node in _local_nodes(statement):
                    if not isinstance(node, ast.Dict):
                        continue
                    for key, value in zip(node.keys, node.values, strict=True):
                        if (not isinstance(key, ast.Constant) or not isinstance(key.value, str)
                                or key.value not in keys or len(key.value) > 64):
                            continue
                        expression = bindings.get(value.id, value) if (
                            isinstance(value, ast.Name) and assignments.get(value.id) == 1
                        ) else value
                        if not _boolean_expression(expression, bindings, assignments,
                                                   boolean_functions):
                            continue
                        rendered = ast.unparse(expression)
                        if len(rendered) > 320:
                            continue
                        rows.append(
                            f"- {relative}:{key.lineno}, function {function.name}, file SHA256 "
                            f"{hashlib.sha256(content).hexdigest()}: {key.value!r} is produced "
                            f"from the boolean expression {rendered!r}."
                        )
                        if len(rows) >= 24:
                            break
                    if len(rows) >= 24:
                        break
                if len(rows) >= 24:
                    break
            if len(rows) >= 24:
                break
        if len(rows) >= 24:
            break
    if not rows:
        return ""
    return ("\n## Current source conditions (not evaluated applicability)\n\n"
            "These expressions are current-source observations, not evaluated truth values "
            "or new historical obligations. Read the cited function and actual input before "
            "using a historical action. Distinct condition keys are distinct evidence scopes; "
            "absence under one condition does not establish absence under another. Do not "
            "broaden an action beyond its original trigger, or invent a new prerequisite to "
            "retain existing supported fields. Resolve uncertain conditions before editing.\n\n"
            + "\n".join(rows) + "\n")
