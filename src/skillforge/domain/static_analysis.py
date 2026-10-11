"""Contracts for deterministic, evidence-linked Python static facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillforge.domain.evidence import Evidence
from skillforge.domain.serialization import to_json_value


class PythonSymbolKind(StrEnum):
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"


class ImportResolution(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNRESOLVED = "unresolved"


class PythonParseStatus(StrEnum):
    PARSED = "parsed"
    FAILED = "failed"


class StaticAnalysisIssueType(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    ENCODING_ERROR = "encoding_error"
    READ_ERROR = "read_error"
    MODULE_NAME_ERROR = "module_name_error"


@dataclass(frozen=True, slots=True)
class PythonSymbol:
    symbol_id: str
    repository_id: str
    module: str | None
    qualified_name: str
    kind: PythonSymbolKind
    path: str
    line: int
    end_line: int
    decorators: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    operation_family: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportRelation:
    repository_id: str
    source_module: str | None
    target_module: str | None
    raw_module: str
    imported_names: tuple[str, ...]
    is_relative: bool
    resolution: ImportResolution
    path: str
    line: int
    evidence_ids: tuple[str, ...]

    @property
    def is_internal(self) -> bool:
        return self.resolution == ImportResolution.INTERNAL


@dataclass(frozen=True, slots=True)
class StaticAnalysisIssue:
    path: str
    issue_type: StaticAnalysisIssueType
    message: str
    line: int | None = None
    column: int | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PythonModuleAnalysis:
    module: str | None
    path: str
    is_package: bool
    symbols: tuple[PythonSymbol, ...]
    imports: tuple[ImportRelation, ...]
    parse_status: PythonParseStatus
    parse_issue: StaticAnalysisIssue | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaticAnalysisResult:
    repository_id: str
    modules: tuple[PythonModuleAnalysis, ...]
    evidence: tuple[Evidence, ...]
    issues: tuple[StaticAnalysisIssue, ...]
    calls: tuple[PythonCallEdge, ...] = ()
    test_linkages: tuple[PythonTestLinkage, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class PythonCallEdge:
    source_symbol: str
    target_symbol: str
    path: str
    line: int
    returns_value: bool
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PythonTestLinkage:
    """Resolved public invocation whose result reaches a real behavior assertion."""

    operation_symbol: str
    test_symbol: str
    path: str
    call_line: int
    assertion_line: int
    evidence_ids: tuple[str, ...]
    assertion_kind: str = "result"
