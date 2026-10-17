from textwrap import dedent

import pytest

from skillforge.repository.static_analysis import PythonStaticAnalyzer


def analyze(tmp_path, caller, extra=None):
    files = {
        "pkg/__init__.py": "",
        "pkg/service.py": "class Service:\n def run(self): pass\ndef run_analysis(): pass\n",
        "pkg/context.py": "from pkg.service import Service\nclass Context:\n store: Service\n",
        "pkg/caller.py": dedent(caller),
        **(extra or {}),
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(content), encoding="utf-8")
    return PythonStaticAnalyzer().analyze(tmp_path)


@pytest.mark.parametrize(
    "caller,source,target",
    [
        (
            "from pkg.service import Service\ndef route(store: Service):\n return store.run()",
            "route",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nasync def route():\n return Service().run()",
            "route",
            "Service.run",
        ),
        (
            "from pkg.service import run_analysis as run\nclass Entry:\n"
            " def f(self):\n  return run()",
            "Entry.f",
            "run_analysis",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n def f(self):\n"
            "  store = Service()\n  return store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n def f(self, *, store: 'Service'):\n"
            "  return store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.context import Context\nclass Entry:\n def f(self, ctx: Context):\n"
            "  return ctx.store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n store: Service\n def f(self):\n"
            "  return self.store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n @property\n"
            " def store(self) -> Service:\n"
            "  return Service()\n def f(self):\n  return self.store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n def __init__(self):\n"
            "  self.store = Service()\n def f(self):\n  return self.store.run()",
            "Entry.f",
            "Service.run",
        ),
        (
            "from pkg.service import Service\nclass Entry:\n def f(self):\n"
            "  store: Service\n  return store.run()",
            "Entry.f",
            "Service.run",
        ),
    ],
)
def test_proven_call_forms_and_evidence_closure(tmp_path, caller, source, target):
    facts = analyze(tmp_path, caller)
    (edge,) = facts.calls
    assert (edge.source_symbol, edge.target_symbol) == (
        "pkg.caller." + source,
        "pkg.service." + target,
    )
    assert edge.returns_value
    evidence = {e.evidence_id: e for e in facts.evidence}
    symbols = {s.qualified_name: s for m in facts.modules for s in m.symbols}
    for name in (edge.source_symbol, edge.target_symbol):
        assert symbols[name].repository_id == facts.repository_id
        assert set(symbols[name].evidence_ids) <= set(edge.evidence_ids)
    assert all(evidence[ref].repository_id == facts.repository_id for ref in edge.evidence_ids)
    assert all(evidence[ref].location.line is not None for ref in edge.evidence_ids)
    assert facts == PythonStaticAnalyzer().analyze(tmp_path)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("return store.run()", True),
        ("result = store.run()\nreturn result", True),
        ("store.run()\nreturn None", False),
        ("result = store.run()\nresult = 0\nreturn result", False),
        ("result = store.run()\nreturn str(result)", False),
        ("result = store.run()\ndef nested():\n return result\nreturn None", False),
    ],
)
def test_value_flow_is_only_proven_return(tmp_path, body, expected):
    caller = "from pkg.service import Service\ndef f(store: Service):\n"
    caller += "\n".join(" " + line for line in body.splitlines())
    (edge,) = analyze(tmp_path, caller).calls
    assert edge.returns_value is expected


@pytest.mark.parametrize(
    "caller",
    [
        "from pkg.service import Service\nfrom other import Service\n"
        "def f():\n return Service().run()",
        "from pkg.service import Service\ndef f(store: Service):\n store = unknown()\n"
        " return store.run()",
        "from pkg.service import Service\ndef f():\n store = Service()\n store = Service()\n"
        " return store.run()",
        "def f(store: Any):\n return store.run()",
        "from external import Service\ndef f(store: Service):\n return store.run()",
        "from pkg.service import Service\ndef f(store: Service):\n return getattr(store, 'run')()",
        "from pkg.service import Service\nfrom pkg.other import Service\ndef f():\n"
        " return Service().run()",
        "from pkg.service import run_analysis\ndef f(run_analysis):\n return run_analysis()",
        "from pkg.service import Service\ndef f():\n if flag:\n  store = Service()\n"
        " return store.run()",
        "from pkg.service import Service\ndef f():\n def nested(store: Service):\n"
        "  return store.run()\n return None",
        "from pkg.context import Context\ndef f(ctx: Context):\n ctx.store = external()\n"
        " return ctx.store.run()",
        "from pkg.context import Context\nContext.store = external\n"
        "def f(ctx: Context):\n return ctx.store.run()",
        "from pkg.service import Service\nService.run = external\n"
        "def f():\n return Service().run()",
        "from pkg.service import Service\ndef f(store: Service):\n store.run = external\n"
        " return store.run()",
        "from pkg.service import Service\nsetattr(Service, 'run', external)\n"
        "def f():\n return Service().run()",
        "from pkg.service import Service\ndef f(store: Service):\n"
        " setattr(store, name, external)\n return store.run()",
        "from pkg.service import Service\nclass Entry:\n @property\n def store(self) -> Any:\n"
        "  pass\n def f(self):\n  return self.store.run()",
    ],
)
def test_uncertain_calls_fail_closed(tmp_path, caller):
    assert (
        analyze(tmp_path, caller, {"pkg/other.py": "class Service:\n def run(self): pass"}).calls
        == ()
    )


def test_source_root_alias_requires_unique_package_layout(tmp_path):
    files = {
        "components/backend/pkg/__init__.py": "",
        "components/backend/pkg/service.py": "class Service:\n def run(self): pass",
        "components/backend/pkg/caller.py": (
            "from pkg.service import Service\ndef route():\n return Service().run()"
        ),
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    (edge,) = facts.calls
    assert edge.target_symbol == "components.backend.pkg.service.Service.run"
    shadow = tmp_path / "another/pkg"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    (shadow / "service.py").write_text("class Service:\n def run(self): pass")
    assert PythonStaticAnalyzer().analyze(tmp_path).calls == ()


@pytest.mark.parametrize(
    "caller,extra",
    [
        (
            "from pkg.service import Service\ndef f():\n return Service().run()",
            {"pkg/service.py": "class Service:\n def run(self): pass\nService = external"},
        ),
        (
            "from pkg.service import run_analysis\ndef f():\n return run_analysis()",
            {"pkg/service.py": "def run_analysis(): pass\nrun_analysis = external"},
        ),
        (
            "from pkg.service import Service\ndef f():\n return Service().run()",
            {"pkg/service.py": "class Service:\n def run(self): pass\n run = external"},
        ),
        (
            "from pkg.service import run_analysis\ndef f():\n def run_analysis(): pass\n"
            " return run_analysis()",
            {},
        ),
        (
            "from pkg.service import Service\nclass Entry:\n store: Service\n def replace(self):\n"
            "  self.store = external()\n def f(self):\n  return self.store.run()",
            {},
        ),
        (
            "from pkg.service import Service\nclass Entry:\n def __init__(self, Service):\n"
            "  self.store = Service()\n def f(self):\n  return self.store.run()",
            {},
        ),
        (
            "from pkg.service import Service\nclass Entry:\n @property\n"
            " def store(self) -> Service:\n"
            "  pass\n @property\n def store(self) -> Any:\n  pass\n def f(self):\n"
            "  return self.store.run()",
            {},
        ),
        (
            "from pkg.service import Service\nclass Entry:\n @property\n"
            " def store(self) -> Service:\n"
            "  pass\n @store.setter\n def store(self, value): pass\n def f(self):\n"
            "  return self.store.run()",
            {},
        ),
        (
            "from pkg.service import Service\nproperty = external\nclass Entry:\n @property\n"
            " def store(self) -> Service:\n  pass\n def f(self):\n  return self.store.run()",
            {},
        ),
        (
            "from pkg.service import Service\nclass Entry:\n Service = external\n"
            " def f(self, store: Service):\n  return store.run()",
            {},
        ),
        ("from pkg.service import Service\ndef f(*stores: Service):\n return stores.run()", {}),
        (
            "def f():\n result = run_analysis()\n from pkg.service import run_analysis\n"
            " return result",
            {},
        ),
        (
            "from pkg.service import Service\ndef patch():\n Service.run = external\n"
            "def f():\n return Service().run()",
            {},
        ),
    ],
)
def test_shadowed_definitions_and_fields_fail_closed(tmp_path, caller, extra):
    assert analyze(tmp_path, caller, extra).calls == ()


@pytest.mark.parametrize(
    "body",
    [
        "from pkg.service import Service\nreturn Service().run()",
        "from pkg.service import run_analysis\nreturn run_analysis()",
    ],
)
def test_function_local_import_is_proven_before_call(tmp_path, body):
    caller = "def f():\n" + "\n".join(" " + line for line in body.splitlines())
    assert len(analyze(tmp_path, caller).calls) == 1


def test_single_annotated_constructor_assignment(tmp_path):
    caller = "from pkg.service import Service\ndef f():\n store: Service = Service()\n"
    caller += " return store.run()"
    assert len(analyze(tmp_path, caller).calls) == 1


def test_duplicate_symbols_do_not_win_by_file_order(tmp_path):
    caller = "from pkg.service import Service\ndef f():\n return Service().run()"
    assert (
        analyze(
            tmp_path,
            caller,
            {
                "pkg/service.pyi": "class Service:\n def run(self): ...",
            },
        ).calls
        == ()
    )


@pytest.mark.parametrize(
    "import_line,receiver",
    [
        ("import pkg.service", "pkg.service.Service"),
        ("import pkg.service as service", "service.Service"),
        ("from .service import Service", "Service"),
    ],
)
def test_explicit_module_binding_constructor(tmp_path, import_line, receiver):
    facts = analyze(tmp_path, import_line + "\ndef f():\n return " + receiver + "().run()")
    (edge,) = facts.calls
    assert edge.target_symbol == "pkg.service.Service.run"


@pytest.mark.parametrize("import_line", ["import pkg", "import pkg.context", "import pkg.unknown"])
def test_import_does_not_prove_a_different_submodule_is_loaded(tmp_path, import_line):
    facts = analyze(tmp_path, import_line + "\ndef f():\n return pkg.service.Service().run()")
    assert facts.calls == ()


def test_explicit_constructor_wrapper_receiver(tmp_path):
    facts = analyze(
        tmp_path,
        "from pkg.service import Service\n"
        "def build() -> Service:\n return Service()\n"
        "def route():\n return build().run()",
    )
    assert {e.target_symbol for e in facts.calls} == {"pkg.caller.build", "pkg.service.Service.run"}
    assert next(e for e in facts.calls if e.target_symbol.endswith(".run")).returns_value


@pytest.mark.parametrize(
    "factory",
    [
        "def build() -> Service:\n return unknown()",
        "def build() -> Service:\n if flag:\n  return Service()\n return unknown()",
        "def build() -> Any:\n return Service()",
        "def build(Service) -> Service:\n return Service()",
        "@external\ndef build() -> Service:\n return Service()",
        "async def build() -> Service:\n return Service()",
        "def build() -> Service:\n return Service(Service := unknown())",
    ],
)
def test_unproven_constructor_wrapper_does_not_supply_receiver(tmp_path, factory):
    facts = analyze(
        tmp_path,
        "from pkg.service import Service\n" + factory + "\ndef route():\n return build().run()",
    )
    assert not any(e.target_symbol.endswith("Service.run") for e in facts.calls)


def test_package_alias_collision_even_when_leaf_only_exists_in_one_root(tmp_path):
    caller = "from alias.service import Service\ndef f():\n return Service().run()"
    assert (
        analyze(
            tmp_path,
            caller,
            {
                "one/alias/__init__.py": "",
                "two/alias/__init__.py": "",
                "one/alias/service.py": "class Service:\n def run(self): pass",
            },
        ).calls
        == ()
    )
