"""Conservative source-only public behavior linkage and testing observations."""

import ast
import hashlib
from collections import defaultdict
from pathlib import PurePosixPath

from skillforge.domain.evidence import Support
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.static_analysis import PythonTestLinkage
from skillforge.domain.structural_rules import TESTING_METHOD, StructuralRuleCandidate
from skillforge.repository.static_analysis import PythonStaticAnalyzer

_ASSERTIONS = {
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse", "assertIs",
    "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn",
    "assertGreater", "assertLess", "assertGreaterEqual", "assertLessEqual",
    "assertAlmostEqual", "assertCountEqual", "assertIsInstance",
}


def _test_file(path):
    name = PurePosixPath(path).name
    return name.startswith("test_") and name.endswith(".py")


def extract_test_linkages(repository_id, modules, trees, evidence, sources):
    """Resolve concrete construction, not name co-occurrence or dynamic helpers.

    Supports unittest setup and direct pytest construction. Unknown fixtures,
    branches, receiver reassignments and snapshot assertions are deliberately absent.
    """
    symbols = {s.qualified_name: s for m in modules for s in m.symbols}
    result = []
    for module in modules:
        tree = trees.get(module.path)
        if not _test_file(module.path) or tree is None or module.module is None:
            continue
        header = sources.get(module.path, "").lower().splitlines()[:10]
        if any("generated" in line or "do not edit" in line for line in header):
            continue
        bindings, import_refs = {}, []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                owner = PythonStaticAnalyzer._relative_target(module, node.level, node.module)
                if owner:
                    for alias in node.names:
                        name = alias.asname or alias.name
                        bindings[name] = owner + "." + alias.name
                import_refs.extend(e.evidence_ids[0] for e in module.imports
                                   if e.line == node.lineno)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name] = alias.name

        def resolve(expr, bindings=bindings):
            if isinstance(expr, ast.Name):
                return bindings.get(expr.id)
            if isinstance(expr, ast.Attribute):
                base = resolve(expr.value)
                return base + "." + expr.attr if base else None
            return None

        def constructions(body):
            assigned = defaultdict(list)
            for statement in body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        owner = resolve(statement.value.func) if isinstance(
                            statement.value, ast.Call) else None
                        assigned[ast.unparse(target)].append((owner, statement))
            return {name: (rows[0][0], rows[0][1]) for name, rows in assigned.items()
                    if len(rows) == 1 and rows[0][0] in symbols
                    and symbols[rows[0][0]].kind.value == "class"}

        groups = [(None, [n for n in tree.body if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef))])]
        groups.extend((cls, [n for n in cls.body if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef))]) for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            and any(resolve(base) == "unittest.TestCase" for base in cls.bases))
        for cls, methods in groups:
            setup = next((n for n in methods if n.name == "setUp"), None)
            inherited = constructions(setup.body) if setup else {}
            for test in methods:
                if not test.name.startswith("test_"):
                    continue
                test_name = ".".join(filter(None, (module.module,
                                                   cls.name if cls else None, test.name)))
                if test_name not in symbols:
                    continue
                receivers = {**inherited, **constructions(test.body)}
                # Any store in a test invalidates an inherited setup binding.
                for node in ast.walk(test):
                    if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
                        node.ctx, ast.Store) and ast.unparse(node) in inherited:
                        receivers.pop(ast.unparse(node), None)
                returned = {}
                for statement in test.body:
                    assertions = []
                    assertion_kind = "result"
                    if isinstance(statement, ast.Assert):
                        assertions = [statement.test]
                    elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                        call = statement.value
                        if (isinstance(call.func, ast.Attribute)
                                and isinstance(call.func.value, ast.Name)
                                and call.func.value.id == "self" and call.func.attr in _ASSERTIONS):
                            assertions = call.args
                    elif isinstance(statement, ast.With):
                        contexts = [item.context_expr for item in statement.items]
                        def raises(call, cls=cls):
                            return isinstance(call, ast.Call) and (
                                resolve(call.func) == "pytest.raises" or (
                                    cls is not None and isinstance(call.func, ast.Attribute)
                                    and ast.unparse(call.func.value) == "self"
                                    and call.func.attr == "assertRaises"))
                        if len(contexts) == 1 and raises(contexts[0]):
                            assertion_kind = "raises"
                            assertions = [s.value for s in statement.body
                                          if isinstance(s, ast.Expr)
                                          and isinstance(s.value, ast.Call)]
                    # Follow only straight-line result assignment; discard overwritten names.
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            returned.pop(ast.unparse(target), None)
                            if (isinstance(target, ast.Name)
                                    and isinstance(statement.value, ast.Call)):
                                returned[target.id] = statement.value
                    for assertion in assertions:
                        if any(isinstance(n, (ast.Lambda, ast.GeneratorExp))
                               for n in ast.walk(assertion)):
                            continue
                        calls = [n for n in ast.walk(assertion) if isinstance(n, ast.Call)]
                        calls.extend(returned[n.id] for n in ast.walk(assertion)
                                     if isinstance(n, ast.Name) and n.id in returned)
                        for call in calls:
                            binding = None
                            if isinstance(call.func, ast.Attribute):
                                binding = receivers.get(ast.unparse(call.func.value))
                                target = binding[0] + "." + call.func.attr if binding else None
                            else:
                                target = resolve(call.func)
                            operation = symbols.get(target)
                            if (operation is None or _test_file(operation.path)
                                    or not operation.operation_family
                                    or any(p.startswith("_") for p in target.split("."))):
                                continue
                            refs = set(import_refs) | set(operation.evidence_ids)
                            refs.update(symbols[test_name].evidence_ids)
                            if binding:
                                node = binding[1]
                                refs.add(PythonStaticAnalyzer._record_evidence(
                                    repository_id, module.path, node.lineno, node.col_offset + 1,
                                    f"test-binding:{node.lineno}:{node.col_offset}:"
                                    f"{ast.dump(node)}",
                                    f"Python test receiver binding: {ast.unparse(node)}", evidence))
                            refs.add(PythonStaticAnalyzer._record_evidence(
                                repository_id, module.path, call.lineno, call.col_offset + 1,
                                f"test-call:{call.lineno}:{call.col_offset}:"
                                f"{test_name}:{target}:{ast.dump(call)}",
                                f"Python public test call {test_name} -> {target}: "
                                + ast.unparse(call), evidence))
                            refs.add(PythonStaticAnalyzer._record_evidence(
                                repository_id, module.path, statement.lineno,
                                statement.col_offset + 1,
                                f"test-assertion:{statement.lineno}:{statement.col_offset}:"
                                f"{test_name}:{ast.dump(statement)}",
                                f"Python behavior assertion {test_name}: "
                                + ast.unparse(statement), evidence))
                            result.append(PythonTestLinkage(
                                target, test_name, module.path, call.lineno, statement.lineno,
                                tuple(sorted(refs)), assertion_kind))
    return tuple(sorted(set(result), key=lambda r: (
        r.operation_symbol, r.test_symbol, r.call_line, r.assertion_line)))


def testing_candidates(facts):
    symbols = {s.qualified_name: s for m in facts.modules for s in m.symbols}
    incoming = {edge.target_symbol for edge in facts.calls
                if not _test_file(symbols[edge.source_symbol].path)}
    families = defaultdict(list)
    for name, symbol in sorted(symbols.items()):
        if (symbol.operation_family and not _test_file(symbol.path) and name not in incoming
                and not any(p.startswith("_") for p in name.split("."))):
            families[(name.rpartition(".")[0], symbol.operation_family)].append(name)
    evidence = {e.evidence_id: e for e in facts.evidence}
    for link in facts.test_linkages:
        production, test = symbols.get(link.operation_symbol), symbols.get(link.test_symbol)
        if (production is None or test is None or test.path != link.path
                or not _test_file(test.path) or _test_file(production.path)
                or not set((*production.evidence_ids, *test.evidence_ids))
                <= set(link.evidence_ids)):
            raise ValueError("testing linkage symbol mismatch")
        for prefix, line in (
            (f"Python public test call {link.test_symbol} -> {link.operation_symbol}:",
             link.call_line),
            (f"Python behavior assertion {link.test_symbol}:", link.assertion_line),
        ):
            if not any(key in evidence and evidence[key].location.path == link.path
                       and evidence[key].location.line == line
                       and evidence[key].description.startswith(prefix)
                       for key in link.evidence_ids):
                raise ValueError("testing linkage call/assertion location mismatch")
    result = []
    for (owner, family), operations in sorted(families.items()):
        if len(operations) < 2:
            continue
        linked = [r for r in facts.test_linkages if r.operation_symbol in operations]
        # Every comparable entry needs linked behavior coverage: no majority inference.
        if {r.operation_symbol for r in linked} != set(operations):
            continue
        # Stable shared project test entry, with independent test symbols per operation.
        paths = set.intersection(*({r.path for r in linked if r.operation_symbol == op}
                                  for op in operations))
        for path in sorted(paths):
            rows = tuple(r for r in linked if r.path == path)
            if len({r.test_symbol for r in rows}) < len(operations):
                continue
            refs = tuple(sorted({key for r in rows for key in r.evidence_ids}))
            if any(key not in evidence or evidence[key].repository_id != facts.repository_id
                   for key in refs):
                raise ValueError("testing Evidence closure mismatch")
            source_modules = tuple(sorted({symbols[op].module for op in operations}))
            target_modules = tuple(sorted({symbols[r.test_symbol].module for r in rows}))
            identity = "\0".join((facts.repository_id, owner, *family, path))
            rule = ProjectRule(
                "structural-testing-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                facts.repository_id, RuleType.ARCHITECTURE,
                "Observed public behavior testing convention for operations "
                + ", ".join(operations) + f" in family {family}: "
                + "; ".join(f"{r.operation_symbol} -> {r.test_symbol} "
                            f"(call {r.path}:{r.call_line}, assertion {r.assertion_line})"
                            for r in rows),
                ProjectScope(ProjectScopeKind.MODULE, source_modules),
                ProjectKnowledgeStatus.CANDIDATE, refs, Support(len(operations), len(operations)),
            )
            result.append(StructuralRuleCandidate(
                rule, owner, target_modules[0], source_modules, target_modules,
                tuple(sorted(set(source_modules) | set(target_modules))), TESTING_METHOD,
                tuple(operations), (), family, rows,
            ))
    return tuple(sorted(result, key=lambda c: c.candidate_id))
