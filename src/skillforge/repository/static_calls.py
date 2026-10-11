"""Conservative repository-local call facts from explicit AST bindings only."""

import ast
from collections import defaultdict
from dataclasses import dataclass

from skillforge.domain.static_analysis import PythonCallEdge, PythonSymbolKind
from skillforge.repository.static_analysis import PythonStaticAnalyzer

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE = (*_FUNCTION, ast.ClassDef, ast.Lambda)
_METHOD = {PythonSymbolKind.METHOD, PythonSymbolKind.ASYNC_METHOD}
_FUNCTION_KIND = {PythonSymbolKind.FUNCTION, PythonSymbolKind.ASYNC_FUNCTION}


@dataclass(frozen=True)
class _Binding:
    name: str
    refs: tuple[str, ...]
    imported_module: str | None = None


def _nodes(body):
    """Walk a lexical body once, stopping at nested definitions and lambdas."""
    pending = list(reversed(body))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, _SCOPE):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _parameters(function):
    args = function.args
    return (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *((args.vararg,) if args.vararg else ()),
        *((args.kwarg,) if args.kwarg else ()),
    )


def _path(node):
    """A cheap lexical attribute path; calls/subscripts are never name bindings."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        if parent:
            return parent + "." + node.attr
    return None


def _writes(nodes):
    writes = defaultdict(list)
    for node in nodes:
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            name = _path(node)
            if name:
                writes[name].append(node)
        elif isinstance(node, (*_FUNCTION, ast.ClassDef)):
            writes[node.name].append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (
                    alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
                )
                writes[name].append(node)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            writes[node.name].append(node)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            writes[node.name].append(node)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            writes[node.rest].append(node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                writes[name].extend((node, node))
    return writes


class _Resolver:
    def __init__(self, repository_id, modules, trees, evidence):
        self.repository_id = repository_id
        self.evidence = evidence
        groups = defaultdict(list)
        for module in modules:
            for symbol in module.symbols:
                groups[symbol.qualified_name].append(symbol)
        self.symbols = {
            name: values[0]
            for name, values in groups.items()
            if len(values) == 1 and values[0].repository_id == repository_id
        }
        self.modules = {m.path: m for m in modules if m.module and m.path in trees}
        self.aliases = defaultdict(list)
        packages = {m.module: m for m in modules if m.module and m.is_package}
        # An explicit package below non-package directories proves a source-root boundary.
        # A second root or canonical module participates in ambiguity, never wins by order.
        for module in modules:
            if not module.module:
                continue
            self.aliases[module.module].append((module.module, ()))
            parts = module.module.split(".")
            for index in range(1, len(parts)):
                package = packages.get(".".join(parts[: index + 1]))
                if package and not any(
                    ".".join(parts[:i]) in packages for i in range(1, index + 1)
                ):
                    self.aliases[".".join(parts[index:])].append(
                        (module.module, package.evidence_ids)
                    )
                    break
        self.bindings = {}
        self.module_writes = {}
        self.scopes = {}
        self.fields = {}
        self.patched = set()
        self.class_shadows = {}
        self.factories = {}
        for path, module in self.modules.items():
            tree = trees[path]
            nodes = tuple(_nodes(tree.body))
            writes = _writes(nodes)
            self.module_writes[path] = writes
            self.bindings[path] = self._bindings(module, tree.body, writes)
            for name, mutations in writes.items():
                if len(mutations) > 1:
                    self.patched.add(f"{module.module}.{name}")
            for node in tree.body:
                if isinstance(node, _FUNCTION):
                    self.scopes[f"{module.module}.{node.name}"] = (module, node, None)
                elif isinstance(node, ast.ClassDef):
                    owner = f"{module.module}.{node.name}"
                    class_writes = _writes(tuple(_nodes(node.body)))
                    self.class_shadows[owner] = set(class_writes)
                    for name, mutations in class_writes.items():
                        if len(mutations) > 1:
                            self.patched.add(owner + "." + name)
                    self.fields[owner] = self._fields(module, node)
                    for method in node.body:
                        if isinstance(method, _FUNCTION):
                            self.scopes[f"{owner}.{method.name}"] = (module, method, owner)
            for name in writes:
                parts = name.split(".")
                binding = self.bindings[path].get(parts[0])
                if binding and len(parts) > 1:
                    self.patched.add(".".join((binding.name, *parts[1:])))
            for node in nodes:
                if isinstance(node, ast.Call):
                    self.patch_call(node, self.bindings[path], {})
        for name, (module, function, owner) in self.scopes.items():
            # A small wrapper is provable without a return-type propagation engine.
            body = function.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if (
                owner
                or not isinstance(function, ast.FunctionDef)
                or function.decorator_list
                or len(body) != 1
                or not isinstance(body[0], ast.Return)
                or not isinstance(body[0].value, ast.Call)
            ):
                continue
            bindings = {
                local: value
                for local, value in self.bindings[module.path].items()
                if local not in {a.arg for a in _parameters(function)}
                and local not in _writes(tuple(_nodes(body)))
            }
            annotated = self.class_type(function.returns, self.bindings[module.path])
            constructed = self.class_type(body[0].value.func, bindings)
            if (
                annotated
                and constructed
                and annotated.name == constructed.name
                and name in self.symbols
            ):
                self.factories[name] = _Binding(
                    constructed.name,
                    (
                        *annotated.refs,
                        *constructed.refs,
                        *self.symbols[name].evidence_ids,
                        self.ref(module, function.returns),
                    ),
                )

    def ref(self, module, node, description="receiver binding"):
        return PythonStaticAnalyzer._record_evidence(
            self.repository_id,
            module.path,
            node.lineno,
            node.col_offset + 1,
            f"{description}:{ast.dump(node)}",
            f"Python {description}: {ast.unparse(node)}",
            self.evidence,
        )

    def _module(self, name):
        matches = self.aliases.get(name, ())
        if len(matches) != 1:
            return None
        # A short import also needs a unique package root, not just a unique leaf file.
        if name and matches[0][1] and len(self.aliases.get(name.split(".")[0], ())) != 1:
            return None
        return matches[0]

    def _bindings(self, module, body, writes, *, definitions=True):
        bindings = {}
        if "*" in writes:
            return bindings
        for node in body:
            if isinstance(node, ast.ImportFrom):
                raw = PythonStaticAnalyzer._relative_target(module, node.level, node.module)
                target = self._module(raw)
                for alias in node.names:
                    local = alias.asname or alias.name
                    if target and len(writes[local]) == 1:
                        name, refs = target
                        bindings[local] = _Binding(
                            name + "." + alias.name,
                            (*refs, self.ref(module, node, "import binding")),
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    target = self._module(alias.name)
                    if target and len(writes[local]) == 1:
                        name, refs = target
                        prefix = (
                            name
                            if alias.asname
                            else ".".join(
                                name.split(".")[
                                    : len(name.split(".")) - len(alias.name.split(".")) + 1
                                ]
                            )
                        )
                        bindings[local] = _Binding(
                            prefix,
                            (*refs, self.ref(module, node, "import binding")),
                            name,
                        )
            elif (
                definitions
                and isinstance(node, (ast.ClassDef, *_FUNCTION))
                and len(writes[node.name]) == 1
            ):
                name = f"{module.module}.{node.name}"
                if name in self.symbols:
                    bindings[node.name] = _Binding(name, self.symbols[name].evidence_ids)
        return bindings

    def qualified(self, node, bindings):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in bindings:
            return None
        binding = bindings[node.id]
        name = ".".join((binding.name, *reversed(parts)))
        symbol = self.symbols.get(name)
        if binding.imported_module and symbol and symbol.module != binding.imported_module:
            return None
        return _Binding(name, binding.refs)

    def class_type(self, node, bindings):
        binding = self.qualified(node, bindings)
        symbol = self.symbols.get(binding.name) if binding else None
        if (
            symbol
            and symbol.kind == PythonSymbolKind.CLASS
            and not self.is_patched(symbol.qualified_name)
        ):
            return _Binding(binding.name, (*binding.refs, *symbol.evidence_ids))
        return None

    def _fields(self, module, cls):
        class_writes = _writes(tuple(_nodes(cls.body)))
        bindings = {
            name: value
            for name, value in self.bindings[module.path].items()
            if name not in class_writes
        }
        definitions = defaultdict(list)
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                definitions[node.target.id].append(
                    (self.class_type(node.annotation, bindings), node)
                )
            elif isinstance(node, _FUNCTION) and any(
                isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
            ):
                resolved = (
                    self.class_type(node.returns, bindings)
                    if len(class_writes[node.name]) == 1
                    and "property" not in class_writes
                    and "property" not in self.module_writes[module.path]
                    else None
                )
                definitions[node.name].append((resolved, node))
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id].append((None, node))
        for init in (n for n in cls.body if isinstance(n, _FUNCTION) and n.name == "__init__"):
            nodes = tuple(_nodes(init.body))
            writes = _writes(nodes)
            typed = {
                a.arg: self.class_type(a.annotation, bindings)
                for a in _parameters(init)
                if a.annotation and not writes.get(a.arg)
            }
            init_bindings = {
                name: value
                for name, value in bindings.items()
                if name not in writes and name not in {a.arg for a in _parameters(init)}
            }
            for statement in init.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                value = statement.value
                resolved = (
                    self.class_type(value.func, init_bindings)
                    if isinstance(value, ast.Call)
                    else typed.get(value.id)
                    if isinstance(value, ast.Name)
                    else None
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        if len(writes[_path(target)]) != 1:
                            resolved = None
                        if resolved and isinstance(value, ast.Name):
                            arg = next(a for a in _parameters(init) if a.arg == value.id)
                            resolved = _Binding(
                                resolved.name, (*resolved.refs, self.ref(module, arg))
                            )
                        definitions[target.attr].append((resolved, statement))
            for name in writes:
                if name.startswith("self.") and name.count(".") == 1:
                    field = name.split(".")[1]
                    if not any(node in init.body for _, node in definitions[field]):
                        definitions[field].append((None, init))
        result = {}
        # A write in another method means constructor/field typing cannot prove this field.
        other_writes = {
            name
            for method in cls.body
            if isinstance(method, _FUNCTION) and method.name != "__init__"
            for name in _writes(tuple(_nodes(method.body)))
        }
        for name, values in definitions.items():
            if "self." + name in other_writes:
                continue
            if all(binding for binding, _ in values) and len({b.name for b, _ in values}) == 1:
                refs = tuple(
                    ref
                    for binding, node in values
                    for ref in (*binding.refs, self.ref(module, node))
                )
                result[name] = _Binding(values[0][0].name, refs)
        return result

    def function(self, module, function, owner):
        statements = [(statement, tuple(_nodes([statement]))) for statement in function.body]
        nodes = tuple(node for _, children in statements for node in children)
        writes = _writes(nodes)
        bindings = {
            name: value for name, value in self.bindings[module.path].items() if name not in writes
        }
        params = _parameters(function)
        for arg in params:
            bindings.pop(arg.arg, None)
        typed = {}
        annotation_bindings = {
            name: value
            for name, value in self.bindings[module.path].items()
            if name not in self.class_shadows.get(owner, ())
        }
        for arg in params:
            resolved = self.class_type(arg.annotation, annotation_bindings)
            if (
                resolved
                and arg.arg not in writes
                and arg is not function.args.vararg
                and arg is not function.args.kwarg
            ):
                typed[arg.arg] = _Binding(resolved.name, (*resolved.refs, self.ref(module, arg)))
        if (
            owner
            and owner in self.symbols
            and params
            and params[0].arg == "self"
            and "self" not in writes
            and not any(
                ast.unparse(d) in {"staticmethod", "classmethod"} for d in function.decorator_list
            )
        ):
            typed["self"] = _Binding(owner, self.symbols[owner].evidence_ids)
        flowing = {
            id(n.value)
            for n in nodes
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        }
        for index, statement in enumerate(function.body):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
            )
            for target in targets:
                if not isinstance(target, ast.Name) or len(writes[target.id]) != 1:
                    continue
                if isinstance(statement.value, ast.Call) and any(
                    isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == target.id
                    for n in function.body[index + 1 :]
                ):
                    flowing.add(id(statement.value))
        calls = []
        for statement, children in statements:
            snapshot, imported = dict(typed), dict(bindings)
            calls.extend(
                (call, snapshot, imported) for call in children if isinstance(call, ast.Call)
            )
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                bindings.update(self._bindings(module, [statement], writes, definitions=False))
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                value = statement.value
                resolved = (
                    self.class_type(statement.annotation, bindings)
                    if isinstance(statement, ast.AnnAssign)
                    else self.class_type(value.func, bindings)
                    if isinstance(value, ast.Call)
                    else None
                )
                if isinstance(statement, ast.AnnAssign) and value is not None:
                    actual = (
                        self.class_type(value.func, bindings)
                        if isinstance(value, ast.Call)
                        else None
                    )
                    if not actual or not resolved or actual.name != resolved.name:
                        resolved = None
                for target in targets:
                    if isinstance(target, ast.Name) and len(writes[target.id]) == 1 and resolved:
                        typed[target.id] = _Binding(
                            resolved.name, (*resolved.refs, self.ref(module, statement))
                        )
        return calls, writes, flowing, bindings, typed

    def receiver(self, node, bindings, typed, writes):
        if isinstance(node, ast.Name):
            return typed.get(node.id)
        if isinstance(node, ast.Call):
            constructed = self.class_type(node.func, bindings)
            if constructed:
                return constructed
            factory = self.qualified(node.func, bindings)
            proven = self.factories.get(factory.name) if factory else None
            if proven and not self.is_patched(factory.name):
                return _Binding(proven.name, (*factory.refs, *proven.refs))
        if isinstance(node, ast.Attribute):
            text = _path(node)
            if text and any(text == name or text.startswith(name + ".") for name in writes):
                return None
            parent = self.receiver(node.value, bindings, typed, writes)
            if parent and self.is_patched(parent.name + "." + node.attr):
                return None
            field = self.fields.get(parent.name, {}).get(node.attr) if parent else None
            if field:
                return _Binding(field.name, (*parent.refs, *field.refs))
        return None

    def is_patched(self, name):
        parts = name.split(".")
        return any(".".join(parts[:index]) in self.patched for index in range(1, len(parts) + 1))

    def patch_call(self, call, bindings, typed):
        if (
            not isinstance(call.func, ast.Name)
            or call.func.id not in {"setattr", "delattr"}
            or len(call.args) < 2
        ):
            return
        receiver = self.receiver(call.args[0], bindings, typed, {})
        receiver = receiver or self.qualified(call.args[0], bindings)
        if receiver:
            member = call.args[1]
            target = (
                receiver.name + "." + member.value
                if isinstance(member, ast.Constant) and isinstance(member.value, str)
                else receiver.name
            )
            self.patched.add(target)


def extract_calls(repository_id, modules, trees, evidence):
    resolver = _Resolver(repository_id, modules, trees, evidence)
    edges = []
    prepared = []
    for source, (module, function, owner) in resolver.scopes.items():
        if source not in resolver.symbols:
            continue
        calls, writes, flowing, bindings, typed = resolver.function(module, function, owner)
        prepared.append((source, module, calls, writes, flowing))
        for call, snapshot, imported in calls:
            resolver.patch_call(call, imported, snapshot)
        # Known patches in any function also make other functions' targets unreliable.
        for name, mutations in writes.items():
            expression = next((n for n in mutations if isinstance(n, ast.Attribute)), None)
            if expression is not None:
                patched = resolver.qualified(expression, bindings)
                receiver = resolver.receiver(expression.value, bindings, typed, {})
                if receiver:
                    if (
                        source.endswith(".__init__")
                        and isinstance(expression.value, ast.Name)
                        and expression.value.id == "self"
                        and expression.attr in resolver.fields.get(receiver.name, {})
                    ):
                        continue
                    patched = _Binding(receiver.name + "." + expression.attr, ())
            elif any(isinstance(n, ast.Global) for n in mutations):
                patched = resolver.bindings[module.path].get(name)
            else:
                patched = None
            if patched:
                resolver.patched.add(patched.name)
    for source, module, calls, writes, flowing in prepared:
        for call, typed, bindings in calls:
            if isinstance(call.func, ast.Name):
                resolved = bindings.get(call.func.id)
                allowed = _FUNCTION_KIND
            elif isinstance(call.func, ast.Attribute):
                if _path(call.func) in writes:
                    continue
                receiver = resolver.receiver(call.func.value, bindings, typed, writes)
                resolved = (
                    _Binding(receiver.name + "." + call.func.attr, receiver.refs)
                    if receiver
                    else None
                )
                allowed = _METHOD
            else:
                continue
            target = resolver.symbols.get(resolved.name) if resolved else None
            if (
                not target
                or target.kind not in allowed
                or resolver.is_patched(target.qualified_name)
            ):
                continue
            ref = PythonStaticAnalyzer._record_evidence(
                repository_id,
                module.path,
                call.lineno,
                call.col_offset + 1,
                f"call:{source}:{target.qualified_name}:{call.lineno}:{call.col_offset}",
                f"Python call {source} -> {target.qualified_name}: {ast.unparse(call)}",
                evidence,
            )
            refs = set(
                (*resolved.refs, ref, *resolver.symbols[source].evidence_ids, *target.evidence_ids)
            )
            if any(
                key not in evidence or evidence[key].repository_id != repository_id for key in refs
            ):
                raise ValueError("call Evidence closure mismatch")
            edges.append(
                PythonCallEdge(
                    source,
                    target.qualified_name,
                    module.path,
                    call.lineno,
                    id(call) in flowing,
                    tuple(sorted(refs)),
                )
            )
    return tuple(sorted(edges, key=lambda e: (e.source_symbol, e.line, e.target_symbol)))
