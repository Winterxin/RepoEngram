"""Small, whole current implementations for history-backed workflow candidates.

These are observations, not new project requirements or historical code snapshots.
Reuse original AST Evidence rather than generating summaries or cutting bodies.
"""

import ast
import io
import json
import tokenize

from skillforge.domain.evidence import EvidenceType
from skillforge.domain.static_analysis import PythonSymbolKind

MAX_CODE_CONTEXT_BYTES = 3072
MAX_IMPLEMENTATION_BYTES = 1536
MAX_PATCH_IMPLEMENTATION_BYTES = 2304
MAX_IMPLEMENTATIONS_PER_MODULE = 1
MAX_PATCH_HELPERS = 3
MAX_HELPER_DEPTH = 2
_FUNCTIONS = {
    PythonSymbolKind.FUNCTION, PythonSymbolKind.ASYNC_FUNCTION,
    PythonSymbolKind.METHOD, PythonSymbolKind.ASYNC_METHOD,
}


def _reference_size(item):
    # Count the actual semantic Evidence representation, including its locator.
    value = {
        "evidence_id": item.evidence_id, "kind": item.kind.value,
        "path": item.location.path, "commit_id": item.location.commit,
        "line": item.location.line, "description": item.description,
        "provenance": item.provenance,
    }
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")) + 1


def _helper_closure(functions, name, limit=MAX_IMPLEMENTATION_BYTES):
    closure = set()

    def visit(value, depth):
        if value in closure:
            return True
        if depth > MAX_HELPER_DEPTH or functions[value][1] > limit:
            return False
        closure.add(value)
        return all(visit(child, depth + 1)
                   for child in sorted(functions[value][2] & functions.keys()))

    return closure if visit(name, 0) else set()


class WorkflowCodeEvidence:
    """Index once, then select bounded examples across candidate modules fairly."""

    def __init__(self, facts):
        evidence = {item.evidence_id: item for item in facts.evidence}
        symbols = {symbol.qualified_name: symbol
                   for module in facts.modules for symbol in module.symbols}
        connected = set()
        for call in facts.calls:
            source, target = symbols.get(call.source_symbol), symbols.get(call.target_symbol)
            if source and target and source.module != target.module:
                connected.update((call.source_symbol, call.target_symbol))
        self.examples = {}
        self.local_functions = {}
        for module in facts.modules:
            examples = []
            for symbol in module.symbols:
                if symbol.kind not in _FUNCTIONS:
                    continue
                for key in symbol.evidence_ids:
                    item = evidence.get(key)
                    prefix = f"Python implementation {symbol.qualified_name}: "
                    if item is None:
                        raise ValueError("Workflow code symbol Evidence is missing")
                    if not item.description.startswith(prefix):
                        continue
                    if (item.repository_id != facts.repository_id
                            or symbol.repository_id != facts.repository_id
                            or item.kind != EvidenceType.SOURCE
                            or item.provenance != "python_ast"
                            or item.location.commit is not None
                            or item.location.path != symbol.path
                            or item.location.line != symbol.line):
                        raise ValueError("Workflow implementation Evidence locator mismatch")
                    size = _reference_size(item)
                    if (symbol.kind in {PythonSymbolKind.FUNCTION, PythonSymbolKind.ASYNC_FUNCTION}
                            and symbol.qualified_name.startswith(module.module + ".")
                            and "." not in symbol.qualified_name[len(module.module) + 1:]):
                        tree = ast.parse(item.description[len(prefix):])
                        calls = {node.func.id for node in ast.walk(tree)
                                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
                        branches = sum(isinstance(node, (ast.If, ast.IfExp))
                                       for node in ast.walk(tree))
                        annotation = tree.body[0].returns
                        predicate = (isinstance(annotation, ast.Name) and annotation.id == "bool"
                                     or isinstance(annotation, ast.Constant)
                                     and annotation.value == "bool")
                        name = symbol.qualified_name[len(module.module) + 1:]
                        self.local_functions.setdefault(symbol.path, {})[name] = (
                            key, size, calls, branches, symbol.line, predicate,
                        )
                    if size > MAX_IMPLEMENTATION_BYTES:
                        continue
                    private = symbol.qualified_name.rsplit(".", 1)[-1].startswith("_")
                    examples.append((
                        (symbol.qualified_name not in connected, private,
                         symbol.line, symbol.qualified_name, key), key, size,
                    ))
            self.examples[module.module] = tuple(sorted(examples))

    def select(self, modules):
        """No snippets, inference, new reads or task-specific names are involved."""
        groups = [self.examples.get(name, ()) for name in sorted(set(modules))]
        selected, used = [], 0
        for index in range(MAX_IMPLEMENTATIONS_PER_MODULE):
            for group in groups:
                if index >= len(group):
                    continue
                _, key, size = group[index]
                if key not in selected and used + size <= MAX_CODE_CONTEXT_BYTES:
                    selected.append(key)
                    used += size
        return tuple(sorted(selected))

    def select_for_patches(self, patches):
        """Whole same-file helper closures, within the existing byte ceiling.

        Added/context calls identify current definitions, not historical policy.
        No attribute calls, removed lines, cross-file lookup or new source reads.
        A closure that cannot fit is skipped rather than partially quoted.
        If an added condition has no usable local call, one same-file boolean
        predicate may illustrate current exceptions, without claiming history.
        """
        roots = []
        for patch in sorted(patches, key=lambda item: item.evidence_id):
            functions = self.local_functions.get(patch.location.path, {})
            names = {}
            changed_condition = False
            for line in patch.description.splitlines()[1:]:
                if not line.startswith(("+", " ")) or line.startswith("+++"):
                    continue
                tokens = []
                generator = tokenize.generate_tokens(io.StringIO(line[1:].strip()).readline)
                try:
                    for token in generator:
                        tokens.append(token)
                except (tokenize.TokenError, IndentationError):
                    # Incomplete control-flow/continuation lines still have useful tokens.
                    pass
                conditional_line = bool(tokens) and tokens[0].string in {
                    "if", "elif", "while", "assert"}
                changed_condition |= line[0] == "+" and conditional_line
                condition_end, depth = len(tokens), 0
                for index, token in enumerate(tokens):
                    if token.type != tokenize.OP:
                        continue
                    if token.string == ":" and depth == 0:
                        condition_end = index
                        break
                    depth += (token.string in {"(", "[", "{"}) - (
                        token.string in {")", "]", "}"})
                for index, token in enumerate(tokens[:-1]):
                    if (token.type == tokenize.NAME and tokens[index + 1].string == "("
                            and (index == 0 or tokens[index - 1].string
                                 not in {".", "def", "class"})):
                        condition = (conditional_line
                                     and index < condition_end
                                     and token.string in functions
                                     and functions[token.string][5])
                        priority = 0 if line[0] == "+" and condition else (
                            1 if line[0] == "+" else 2)
                        names[token.string] = min(names.get(token.string, 2), priority)
            for name in names.keys() & functions.keys():
                item = functions[name]
                roots.append((names[name], -item[3], patch.location.path, item[4], name))
            if changed_condition and not any(
                _helper_closure(functions, name, MAX_PATCH_IMPLEMENTATION_BYTES)
                for name in names.keys() & functions.keys()
            ):
                # Observed current source is a fallback, below all historical call sites.
                predicates = []
                for name, item in functions.items():
                    if not item[5] or not item[3]:
                        continue
                    closure = _helper_closure(functions, name, MAX_PATCH_IMPLEMENTATION_BYTES)
                    if (closure and len(closure) <= MAX_PATCH_HELPERS
                            and sum(functions[key][1] for key in closure)
                            <= MAX_CODE_CONTEXT_BYTES):
                        predicates.append((3, -item[3], patch.location.path, item[4], name))
                if predicates:
                    roots.append(min(predicates))
        selected, used = set(), 0
        for _, _, path, _, name in sorted(set(roots)):
            functions = self.local_functions[path]
            closure = _helper_closure(functions, name, MAX_PATCH_IMPLEMENTATION_BYTES)
            if not closure:
                continue
            additions = {functions[value][0]: functions[value][1] for value in closure
                         if functions[value][0] not in selected}
            size = sum(additions.values())
            if (len(selected) + len(additions) <= MAX_PATCH_HELPERS
                    and used + size <= MAX_CODE_CONTEXT_BYTES):
                selected.update(additions)
                used += size
        return tuple(sorted(selected))
