"""Read-only, deterministic Python AST analysis for one repository."""

from __future__ import annotations

import ast
import hashlib
import os
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from skillforge.config import AnalyzerConfig
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.domain.static_analysis import (
    ImportRelation,
    ImportResolution,
    PythonModuleAnalysis,
    PythonParseStatus,
    PythonSymbol,
    PythonSymbolKind,
    StaticAnalysisIssue,
    StaticAnalysisIssueType,
    StaticAnalysisResult,
)
from skillforge.errors import InvalidRepositoryPath, RepositoryReadError

_FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class _ModuleFile:
    path: str
    module: str | None
    is_package: bool


class PythonStaticAnalyzer:
    """Extract Python modules, symbols, and imports without executing repository code."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()

    def analyze(self, repo_path: Path | str) -> StaticAnalysisResult:
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")

        repository_id = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:20]
        module_files = self._discover(root)
        known_modules = {item.module for item in module_files if item.module}
        package_names = self._package_names(module_files)
        evidence_by_id: dict[str, Evidence] = {}
        modules: list[PythonModuleAnalysis] = []
        issues: list[StaticAnalysisIssue] = []
        trees: dict[str, ast.Module] = {}
        sources: dict[str, str] = {}

        for item in module_files:
            module_evidence = self._record_evidence(
                repository_id,
                item.path,
                None,
                None,
                f"module:{item.module}:{item.is_package}",
                f"Python module file for {item.module or item.path}",
                evidence_by_id,
            )
            if item.module is None:
                issues.append(
                    self._issue(
                        repository_id,
                        item.path,
                        StaticAnalysisIssueType.MODULE_NAME_ERROR,
                        "Path components do not form a valid dotted Python module name",
                        None,
                        None,
                        evidence_by_id,
                    )
                )
            try:
                with tokenize.open(root / item.path) as stream:
                    source = stream.read()
            except (UnicodeError, LookupError, SyntaxError) as error:
                issue = self._issue(
                    repository_id,
                    item.path,
                    StaticAnalysisIssueType.ENCODING_ERROR,
                    self._encoding_message(error),
                    None,
                    None,
                    evidence_by_id,
                )
                issues.append(issue)
                modules.append(self._failed_module(item, issue, module_evidence))
                continue
            except OSError as error:
                issue = self._issue(
                    repository_id,
                    item.path,
                    StaticAnalysisIssueType.READ_ERROR,
                    str(error),
                    None,
                    None,
                    evidence_by_id,
                )
                issues.append(issue)
                modules.append(self._failed_module(item, issue, module_evidence))
                continue

            try:
                tree = ast.parse(source, filename=item.path)
            except SyntaxError as error:
                issue = self._issue(
                    repository_id,
                    item.path,
                    StaticAnalysisIssueType.SYNTAX_ERROR,
                    error.msg,
                    error.lineno,
                    error.offset,
                    evidence_by_id,
                )
                issues.append(issue)
                modules.append(self._failed_module(item, issue, module_evidence))
                continue

            symbols = self._symbols(repository_id, item, tree, evidence_by_id)
            trees[item.path] = tree
            sources[item.path] = source
            imports = self._imports(
                repository_id, item, tree, known_modules, package_names, evidence_by_id
            )
            modules.append(
                PythonModuleAnalysis(
                    item.module,
                    item.path,
                    item.is_package,
                    symbols,
                    imports,
                    PythonParseStatus.PARSED,
                    None,
                    (module_evidence,),
                )
            )

        from skillforge.repository.static_calls import extract_calls

        calls = extract_calls(repository_id, tuple(modules), trees, evidence_by_id)
        from skillforge.repository.static_testing import extract_test_linkages

        test_linkages = extract_test_linkages(
            repository_id, tuple(modules), trees, evidence_by_id, sources
        )
        return StaticAnalysisResult(
            repository_id,
            tuple(modules),
            tuple(sorted(evidence_by_id.values(), key=lambda value: value.evidence_id)),
            tuple(
                sorted(
                    issues,
                    key=lambda value: (value.path, value.line or 0, value.issue_type),
                )
            ),
            calls,
            test_linkages,
        )

    def _discover(self, root: Path) -> tuple[_ModuleFile, ...]:
        paths: list[str] = []

        def on_error(error: OSError) -> None:
            raise RepositoryReadError(f"Cannot walk repository: {error}") from error

        for current, names, filenames in os.walk(root, followlinks=False, onerror=on_error):
            current_path = Path(current)
            names[:] = sorted(
                name
                for name in names
                if name not in self.config.excluded_directories
                and not name.endswith(".egg-info")
                and not (current_path / name).is_symlink()
            )
            for name in sorted(filenames):
                path = current_path / name
                if path.suffix not in {".py", ".pyi"} or path.is_symlink():
                    continue
                paths.append(path.relative_to(root).as_posix())

        return tuple(self._module_file(path) for path in sorted(paths))

    @staticmethod
    def _module_file(path: str) -> _ModuleFile:
        parts = list(Path(path).parts)
        if parts and parts[0] == "src":
            parts.pop(0)
        filename = parts.pop()
        is_package = filename in {"__init__.py", "__init__.pyi"}
        if not is_package:
            parts.append(filename.removesuffix(".pyi").removesuffix(".py"))
        module = ".".join(parts) if parts and all(part.isidentifier() for part in parts) else None
        return _ModuleFile(path, module, is_package)

    @staticmethod
    def _package_names(module_files: tuple[_ModuleFile, ...]) -> set[str]:
        names: set[str] = set()
        for item in module_files:
            if item.module is None:
                continue
            parts = item.module.split(".")
            if item.is_package:
                names.add(item.module)
            for index in range(1, len(parts)):
                names.add(".".join(parts[:index]))
        return names

    def _symbols(
        self,
        repository_id: str,
        module_file: _ModuleFile,
        tree: ast.Module,
        evidence_by_id: dict[str, Evidence],
    ) -> tuple[PythonSymbol, ...]:
        from skillforge.repository.literal_dependencies import module_literals

        literals = module_literals(tree)
        symbols: list[PythonSymbol] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    self._make_symbol(
                        repository_id,
                        module_file,
                        node.name,
                        PythonSymbolKind.CLASS,
                        node,
                        node.decorator_list,
                        evidence_by_id,
                    )
                )
                for member in node.body:
                    if isinstance(member, _FunctionNode):
                        kind = (
                            PythonSymbolKind.ASYNC_METHOD
                            if isinstance(member, ast.AsyncFunctionDef)
                            else PythonSymbolKind.METHOD
                        )
                        symbols.append(
                            self._make_symbol(
                                repository_id,
                                module_file,
                                f"{node.name}.{member.name}",
                                kind,
                                member,
                                member.decorator_list,
                                evidence_by_id,
                            )
                        )
            elif isinstance(node, _FunctionNode):
                kind = (
                    PythonSymbolKind.ASYNC_FUNCTION
                    if isinstance(node, ast.AsyncFunctionDef)
                    else PythonSymbolKind.FUNCTION
                )
                symbols.append(
                    self._make_symbol(
                        repository_id,
                        module_file,
                        node.name,
                        kind,
                        node,
                        node.decorator_list,
                        evidence_by_id,
                        literals=literals,
                    )
                )
        return tuple(symbols)

    def _make_symbol(
        self,
        repository_id: str,
        module_file: _ModuleFile,
        local_name: str,
        kind: PythonSymbolKind,
        node: ast.ClassDef | _FunctionNode,
        decorator_nodes: list[ast.expr],
        evidence_by_id: dict[str, Evidence],
        *,
        literals=None,
    ) -> PythonSymbol:
        qualified_name = ".".join(filter(None, (module_file.module, local_name)))
        line = node.lineno
        end_line = node.end_lineno or line
        column = node.col_offset + 1
        decorators = tuple(ast.unparse(decorator) for decorator in decorator_nodes)
        subject = f"symbol:{module_file.path}:{qualified_name}:{kind.value}:{line}:{column}"
        evidence_id = self._record_evidence(
            repository_id,
            module_file.path,
            line,
            column,
            subject,
            f"Python {kind.value} {qualified_name}",
            evidence_by_id,
        )
        symbol_id = self._stable_id(repository_id, subject)
        family = ()
        refs = (evidence_id,)
        if isinstance(node, _FunctionNode):
            from skillforge.repository.literal_dependencies import literal_context

            literal_body = literal_context(node, literals or {})
            parameters = [arg for arg in (*node.args.posonlyargs, *node.args.args)
                          if arg.arg not in {"self", "cls"}]
            if parameters and parameters[0].annotation is not None and node.returns is not None:
                family = (parameters[0].arg, ast.unparse(parameters[0].annotation),
                          ast.unparse(node.returns))
            body_id = self._record_evidence(
                repository_id, module_file.path, line, column,
                f"body:{qualified_name}:{ast.dump(node)}" + literal_body,
                f"Python implementation {qualified_name}: {ast.unparse(node)}" + literal_body,
                evidence_by_id,
            )
            refs += (body_id,)
        return PythonSymbol(
            symbol_id,
            repository_id,
            module_file.module,
            qualified_name,
            kind,
            module_file.path,
            line,
            end_line,
            decorators,
            refs,
            family,
        )

    def _imports(
        self,
        repository_id: str,
        module_file: _ModuleFile,
        tree: ast.Module,
        known_modules: set[str],
        package_names: set[str],
        evidence_by_id: dict[str, Evidence],
    ) -> tuple[ImportRelation, ...]:
        imports: list[ImportRelation] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    relation = self._make_import(
                        repository_id,
                        module_file,
                        node,
                        target,
                        target,
                        (alias.name,),
                        False,
                        known_modules,
                        package_names,
                        evidence_by_id,
                    )
                    imports.append(relation)
            elif isinstance(node, ast.ImportFrom):
                target = self._relative_target(module_file, node.level, node.module)
                raw_module = ("." * node.level) + (node.module or "")
                names = tuple(alias.name for alias in node.names)
                imports.append(
                    self._make_import(
                        repository_id,
                        module_file,
                        node,
                        target,
                        raw_module,
                        names,
                        node.level > 0,
                        known_modules,
                        package_names,
                        evidence_by_id,
                    )
                )
        imports.sort(key=lambda value: (value.line, value.raw_module, value.imported_names))
        return tuple(imports)

    def _make_import(
        self,
        repository_id: str,
        module_file: _ModuleFile,
        node: ast.Import | ast.ImportFrom,
        target: str | None,
        raw_module: str,
        imported_names: tuple[str, ...],
        is_relative: bool,
        known_modules: set[str],
        package_names: set[str],
        evidence_by_id: dict[str, Evidence],
    ) -> ImportRelation:
        resolution = self._resolve_import(target, is_relative, known_modules, package_names)
        line = node.lineno
        column = node.col_offset + 1
        subject = self._import_evidence_subject(
            raw_module, target, imported_names, line, column, resolution
        )
        evidence_id = self._record_evidence(
            repository_id,
            module_file.path,
            line,
            column,
            subject,
            f"Python import {raw_module}",
            evidence_by_id,
        )
        return ImportRelation(
            repository_id,
            module_file.module,
            target,
            raw_module,
            imported_names,
            is_relative,
            resolution,
            module_file.path,
            line,
            (evidence_id,),
        )

    @staticmethod
    def _import_evidence_subject(
        raw_module: str,
        target_module: str | None,
        imported_names: tuple[str, ...],
        line: int,
        column: int,
        resolution: ImportResolution,
    ) -> str:
        return (
            f"import:{raw_module}:{target_module}:{','.join(imported_names)}:{line}:{column}:"
            f"{resolution.value}"
        )

    @classmethod
    def expected_import_evidence_id(cls, relation: ImportRelation, column: int) -> str:
        """Reconstruct the analyzer's canonical Evidence ID for an import relation."""
        subject = cls._import_evidence_subject(
            relation.raw_module,
            relation.target_module,
            relation.imported_names,
            relation.line,
            column,
            relation.resolution,
        )
        return cls._stable_id(relation.repository_id, f"{relation.path}:{subject}")

    @staticmethod
    def _relative_target(
        module_file: _ModuleFile, level: int, imported_module: str | None
    ) -> str | None:
        if level == 0:
            return imported_module
        if module_file.module is None:
            return None
        package = (
            module_file.module
            if module_file.is_package
            else module_file.module.rpartition(".")[0]
        )
        if not package:
            return None
        parts = package.split(".") if package else []
        ascend = level - 1
        if ascend >= len(parts):
            return None
        base = parts[: len(parts) - ascend] if ascend else parts
        if imported_module:
            base.extend(imported_module.split("."))
        return ".".join(base) or None

    @staticmethod
    def _resolve_import(
        target: str | None,
        is_relative: bool,
        known_modules: set[str],
        package_names: set[str],
    ) -> ImportResolution:
        if target and (target in known_modules or target in package_names):
            return ImportResolution.INTERNAL
        if target and any(name.startswith(target + ".") for name in known_modules):
            return ImportResolution.INTERNAL
        if is_relative or target is None:
            return ImportResolution.UNRESOLVED
        root = target.split(".", 1)[0]
        if root in known_modules or root in package_names:
            return ImportResolution.UNRESOLVED
        return ImportResolution.EXTERNAL

    def _issue(
        self,
        repository_id: str,
        path: str,
        issue_type: StaticAnalysisIssueType,
        message: str,
        line: int | None,
        column: int | None,
        evidence_by_id: dict[str, Evidence],
    ) -> StaticAnalysisIssue:
        subject = f"issue:{issue_type.value}:{line}:{column}:{message}"
        evidence_id = self._record_evidence(
            repository_id,
            path,
            line,
            column,
            subject,
            f"Python static analysis issue: {message}",
            evidence_by_id,
        )
        return StaticAnalysisIssue(path, issue_type, message, line, column, (evidence_id,))

    @staticmethod
    def _failed_module(
        item: _ModuleFile, issue: StaticAnalysisIssue, module_evidence: str
    ) -> PythonModuleAnalysis:
        return PythonModuleAnalysis(
            item.module,
            item.path,
            item.is_package,
            (),
            (),
            PythonParseStatus.FAILED,
            issue,
            (module_evidence,),
        )

    @staticmethod
    def _encoding_message(error: Exception) -> str:
        if isinstance(error, UnicodeError):
            return str(error)
        if isinstance(error, SyntaxError):
            return error.msg
        return str(error)

    @staticmethod
    def _stable_id(repository_id: str, subject: str) -> str:
        return hashlib.sha256(f"{repository_id}:{subject}".encode()).hexdigest()[:24]

    @classmethod
    def _record_evidence(
        cls,
        repository_id: str,
        path: str,
        line: int | None,
        column: int | None,
        subject: str,
        description: str,
        evidence_by_id: dict[str, Evidence],
    ) -> str:
        evidence_id = cls._stable_id(repository_id, f"{path}:{subject}")
        evidence_by_id[evidence_id] = Evidence(
            evidence_id,
            repository_id,
            EvidenceType.SOURCE,
            SourceLocation(path, line=line, column=column),
            description,
            "python_ast",
        )
        return evidence_id
