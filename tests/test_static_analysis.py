import json

import skillforge.repository.static_analysis as static_analysis
from skillforge.cli import main
from skillforge.domain.static_analysis import ImportResolution, PythonSymbolKind
from skillforge.repository.static_analysis import PythonStaticAnalyzer


def _write(root, path, content=""):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_analyzes_python_symbols_decorators_and_locations(tmp_path):
    _write(
        tmp_path,
        "pkg/api.py",
        """@router.get('/users')
def list_users():
    return []

async def fetch_users():
    return []

class UserService:
    @property
    def active(self):
        return True

    async def load(self):
        return []
""",
    )

    result = PythonStaticAnalyzer().analyze(tmp_path)
    module = result.modules[0]
    symbols = {symbol.qualified_name: symbol for symbol in module.symbols}

    assert module.module == "pkg.api"
    assert {symbol.kind for symbol in symbols.values()} == {
        PythonSymbolKind.FUNCTION,
        PythonSymbolKind.ASYNC_FUNCTION,
        PythonSymbolKind.CLASS,
        PythonSymbolKind.METHOD,
        PythonSymbolKind.ASYNC_METHOD,
    }
    assert symbols["pkg.api.list_users"].decorators == ("router.get('/users')",)
    assert symbols["pkg.api.list_users"].line == 2
    assert symbols["pkg.api.list_users"].end_line == 3
    assert symbols["pkg.api.UserService.active"].decorators == ("property",)
    assert symbols["pkg.api.UserService.load"].line == 13


def test_detects_packages_src_layout_and_evidence_locations(tmp_path):
    _write(tmp_path, "src/shop/__init__.py", "")
    _write(tmp_path, "src/shop/models/user.py", "class User: pass\n")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    modules = {module.module: module for module in result.modules}
    evidence = {item.evidence_id: item for item in result.evidence}

    assert modules["shop"].is_package is True
    assert modules["shop"].path == "src/shop/__init__.py"
    assert modules["shop.models.user"].is_package is False
    symbol = modules["shop.models.user"].symbols[0]
    assert symbol.path == "src/shop/models/user.py"
    assert evidence[symbol.evidence_ids[0]].location.line == 1
    assert evidence[symbol.evidence_ids[0]].location.path == symbol.path


def test_discovers_and_maps_pyi_modules_and_packages(tmp_path):
    _write(tmp_path, "src/stubs/__init__.pyi", "")
    _write(tmp_path, "src/stubs/api.pyi", "class User: ...\n")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    modules = {module.path: module for module in result.modules}

    assert modules["src/stubs/__init__.pyi"].module == "stubs"
    assert modules["src/stubs/__init__.pyi"].is_package is True
    assert modules["src/stubs/api.pyi"].module == "stubs.api"
    assert modules["src/stubs/api.pyi"].is_package is False
    assert modules["src/stubs/api.pyi"].parse_issue is None
    assert modules["src/stubs/api.pyi"].symbols[0].qualified_name == "stubs.api.User"


def test_resolves_absolute_relative_and_external_imports(tmp_path):
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/api/users.py", """import app.services.user
from app.services.user import UserService
from . import helpers
from .service import Service
from ..models import User
import requests
""")
    _write(tmp_path, "app/api/__init__.py", "")
    _write(tmp_path, "app/api/helpers.py", "")
    _write(tmp_path, "app/api/service.py", "")
    _write(tmp_path, "app/services/__init__.py", "")
    _write(tmp_path, "app/services/user.py", "")
    _write(tmp_path, "app/models/__init__.py", "")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    users_module = next(module for module in result.modules if module.path == "app/api/users.py")
    imports = users_module.imports
    by_raw = {item.raw_module: item for item in imports}

    assert by_raw["app.services.user"].source_module == "app.api.users"
    assert by_raw["app.services.user"].target_module == "app.services.user"
    assert by_raw["app.services.user"].resolution == ImportResolution.INTERNAL
    assert by_raw["app.services.user"].imported_names == ("UserService",)
    assert by_raw["."].target_module == "app.api"
    assert by_raw["."].is_relative is True
    assert by_raw[".service"].target_module == "app.api.service"
    assert by_raw[".service"].resolution == ImportResolution.INTERNAL
    assert by_raw["..models"].target_module == "app.models"
    assert by_raw["..models"].resolution == ImportResolution.INTERNAL
    assert by_raw["requests"].resolution == ImportResolution.EXTERNAL
    evidence_ids = {item.evidence_id for item in result.evidence}
    assert all(
        evidence_id in evidence_ids
        for relation in imports
        for evidence_id in relation.evidence_ids
    )
    assert by_raw["requests"].line == 6


def test_unresolved_relative_and_missing_internal_imports_are_explicit(tmp_path):
    _write(
        tmp_path,
        "pkg/source.py",
        "import pkg.not_present\nfrom .missing import Thing\n",
    )
    _write(tmp_path, "pkg/__init__.py", "")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    imports = result.modules[1].imports

    assert [item.resolution for item in imports] == [
        ImportResolution.UNRESOLVED,
        ImportResolution.UNRESOLVED,
    ]
    assert all(item.is_internal is False for item in imports)
    assert imports[1].target_module == "pkg.missing"


def test_relative_import_at_repository_root_is_unresolved(tmp_path):
    _write(tmp_path, "orphan.py", "from .missing import Thing\n")

    relation = PythonStaticAnalyzer().analyze(tmp_path).modules[0].imports[0]

    assert relation.target_module is None
    assert relation.resolution == ImportResolution.UNRESOLVED


def test_beyond_top_level_relative_import_in_single_package_is_unresolved(tmp_path):
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/module.py", "from ..x import y\n")
    _write(tmp_path, "x.py", "value = 1\n")

    module = next(
        item for item in PythonStaticAnalyzer().analyze(tmp_path).modules
        if item.path == "app/module.py"
    )
    relation = module.imports[0]

    assert relation.target_module is None
    assert relation.resolution == ImportResolution.UNRESOLVED


def test_beyond_top_level_relative_import_in_nested_package_is_unresolved(tmp_path):
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/api/__init__.py", "")
    _write(tmp_path, "app/api/users.py", "from ...models import User\n")
    _write(tmp_path, "models.py", "class User: ...\n")

    module = next(
        item for item in PythonStaticAnalyzer().analyze(tmp_path).modules
        if item.path == "app/api/users.py"
    )
    relation = module.imports[0]

    assert relation.target_module is None
    assert relation.resolution == ImportResolution.UNRESOLVED


def test_invalid_path_components_do_not_become_fake_module_names(tmp_path):
    _write(tmp_path, ".scratch.fixtures/pkg-file.py", "def run(): pass\n")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    module = result.modules[0]

    assert module.module is None
    assert module.parse_issue is None
    assert module.symbols[0].module is None
    assert module.symbols[0].qualified_name == "run"
    assert result.issues[0].issue_type.value == "module_name_error"
    assert result.issues[0].path == ".scratch.fixtures/pkg-file.py"
    assert result.issues[0].evidence_ids[0] in {
        evidence.evidence_id for evidence in result.evidence
    }


def test_syntax_and_encoding_errors_do_not_stop_other_modules(tmp_path):
    _write(tmp_path, "bad_syntax.py", "def broken(:\n")
    _write(tmp_path, "good.py", "def works(): pass\n")
    (tmp_path / "bad_encoding.py").write_bytes(b"# coding: ascii\nname = '\xff'\n")

    result = PythonStaticAnalyzer().analyze(tmp_path)
    modules = {module.path: module for module in result.modules}
    issues = {issue.path: issue for issue in result.issues}

    assert modules["bad_syntax.py"].parse_status.value == "failed"
    assert issues["bad_syntax.py"].issue_type.value == "syntax_error"
    assert issues["bad_syntax.py"].line == 1
    assert modules["bad_encoding.py"].parse_status.value == "failed"
    assert issues["bad_encoding.py"].issue_type.value == "encoding_error"
    assert modules["good.py"].symbols[0].qualified_name == "good.works"
    assert set(issues["bad_syntax.py"].evidence_ids) <= {
        item.evidence_id for item in result.evidence
    }


def test_unreadable_file_is_an_issue_and_analysis_continues(tmp_path, monkeypatch):
    _write(tmp_path, "unreadable.py", "value = 1\n")
    _write(tmp_path, "readable.py", "def ready(): pass\n")
    open_file = static_analysis.tokenize.open

    def fail_selected(path):
        if path.name == "unreadable.py":
            raise OSError("fixture read failure")
        return open_file(path)

    monkeypatch.setattr(static_analysis.tokenize, "open", fail_selected)
    result = PythonStaticAnalyzer().analyze(tmp_path)

    assert result.issues[0].issue_type.value == "read_error"
    assert result.issues[0].path == "unreadable.py"
    assert any(
        symbol.qualified_name == "readable.ready"
        for module in result.modules
        for symbol in module.symbols
    )


def test_nested_functions_and_classes_are_not_misreported_as_top_level(tmp_path):
    _write(
        tmp_path,
        "nested.py",
        """def outer():
    def inner(): pass

class Outer:
    class Inner:
        def hidden(self): pass
""",
    )

    symbols = PythonStaticAnalyzer().analyze(tmp_path).modules[0].symbols

    assert [item.qualified_name for item in symbols] == ["nested.outer", "nested.Outer"]


def test_empty_repository_and_json_serialization(tmp_path, capsys):
    result = PythonStaticAnalyzer().analyze(tmp_path)

    assert result.modules == ()
    assert result.issues == ()
    assert json.loads(json.dumps(result.to_dict()))["modules"] == []
    assert main(["static", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Modules: 0" in output
    assert "Imports: 0" in output


def test_static_cli_json_includes_structured_facts(tmp_path, capsys):
    _write(tmp_path, "café.py", "from math import sqrt\nclass Café: pass\n")

    assert main(["static", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["modules"][0]["path"] == "café.py"
    assert payload["modules"][0]["symbols"][0]["qualified_name"] == "café.Café"
    assert payload["modules"][0]["imports"][0]["resolution"] == "external"
