"""Small complete literal bindings referenced by current top-level functions."""

import ast
import re
from collections import Counter

MAX_LITERAL_CONTEXT_BYTES = 512
_SENSITIVE = re.compile(r"secret|password|credential|token|auth|private.?key|api.?key", re.I)


def module_literals(tree):
    """Observe unique immutable tuples; imports, rebindings and dynamic values skip."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name)
                and node.func.id in {'exec', 'eval', 'globals', 'locals', 'vars'}
                or isinstance(node.func, ast.Attribute) and node.func.attr in {'exec', 'eval'}):
            return {}
    counts, definitions, forbidden = Counter(), {}, set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] += 1
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            counts[node.id] += 1
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            counts.update(a.asname or a.name.split('.')[0] for a in node.names)
        pending.extend(ast.iter_child_nodes(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            forbidden.update(node.names)
    for node in tree.body:
        target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                  else node.target if isinstance(node, ast.AnnAssign) else None)
        if not isinstance(target, ast.Name):
            continue
        name, value = target.id, node.value
        if (counts[name] != 1 or name in forbidden or not name.isupper()
                or _SENSITIVE.search(name) or not isinstance(value, ast.Tuple)
                or len(value.elts) > 32):
            continue
        if not all(isinstance(e, ast.Constant)
                   and type(e.value) in {str, int, float, bool, type(None)}
                   and not (isinstance(e.value, str) and (len(e.value) > 128
                            or e.value.startswith(('sk-', 'ghp_', '-----BEGIN'))))
                   for e in value.elts):
            continue
        definition = ast.unparse(ast.Assign(targets=[target], value=value, lineno=node.lineno))
        definitions[name] = (node.lineno, definition)
    return definitions


def literal_context(function, definitions):
    """Never execute code, quote partial tuples, or infer historical policy."""
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
           for statement in function.body for n in ast.walk(statement)):
        return ''
    bound = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args,
                                *function.args.kwonlyargs)}
    bound.update(a.arg for a in (function.args.vararg, function.args.kwarg) if a)
    bound.update(n.id for n in ast.walk(function)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)))
    for node in ast.walk(function):
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split('.')[0] for a in node.names)
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
             and isinstance(n.ctx, ast.Load) and n.id in definitions} - bound
    rows = []
    for name in sorted(names):
        line, definition = definitions[name]
        row = f'# Current module binding at line {line}\n{definition}'
        proposal = '\n# Bounded literal observations; dynamic changes unknown.\n' + '\n'.join(
            [*rows, row]) + f'\n# Omitted referenced bindings: {len(names) - len(rows) - 1}'
        if len(proposal.encode('utf-8')) <= MAX_LITERAL_CONTEXT_BYTES:
            rows.append(row)
    if not rows:
        return ''
    return ('\n# Bounded literal observations; dynamic changes unknown.\n' + '\n'.join(rows)
            + f'\n# Omitted referenced bindings: {len(names) - len(rows)}')
