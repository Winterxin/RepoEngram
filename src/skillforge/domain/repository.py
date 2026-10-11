"""Repository inventory contracts. Paths in findings are relative to the root."""

from dataclasses import dataclass
from pathlib import Path

from skillforge.domain.evidence import Evidence
from skillforge.domain.serialization import to_json_value


@dataclass(frozen=True, slots=True)
class RepositoryRef:
    repository_id: str
    root: Path


@dataclass(frozen=True, slots=True)
class GitMetadata:
    is_git_repository: bool
    branch: str | None = None
    head_commit: str | None = None


@dataclass(frozen=True, slots=True)
class LanguageStat:
    language: str
    file_count: int
    example_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PythonModule:
    module: str
    path: str
    is_package: bool


@dataclass(frozen=True, slots=True)
class RepositorySignal:
    category: str
    paths: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    repository: RepositoryRef
    file_count: int
    languages: tuple[LanguageStat, ...]
    frameworks: tuple[str, ...]
    top_level_directories: tuple[str, ...]
    directory_summary: tuple[str, ...]
    python_modules: tuple[PythonModule, ...]
    signals: tuple[RepositorySignal, ...]
    git: GitMetadata
    evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
