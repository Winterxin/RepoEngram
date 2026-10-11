"""Evidence and confidence are explicit so derived rules remain auditable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceType(StrEnum):
    SOURCE = "source"
    CONFIG = "config"
    TEST = "test"
    GIT = "git"
    DOCUMENT = "document"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A repo-relative file locator, optionally at a revision and line."""

    path: str
    commit: str | None = None
    line: int | None = None
    column: int | None = None

    def __post_init__(self) -> None:
        if (
            self.path.startswith("/")
            or "\\" in self.path
            or ":" in self.path.split("/")[0]
            or ".." in self.path.split("/")
        ):
            raise ValueError("SourceLocation.path must be a repo-relative POSIX path")
        if self.line is not None and self.line < 1:
            raise ValueError("line must be positive")
        if self.column is not None and self.column < 1:
            raise ValueError("column must be positive")


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    repository_id: str
    kind: EvidenceType
    location: SourceLocation
    description: str
    provenance: str  # e.g. filesystem_inventory, pyproject_parser, git_rev_parse
    confidence: Confidence | None = None  # direct observations can leave this unset


@dataclass(frozen=True, slots=True)
class Support:
    supporting: int
    examined: int

    def __post_init__(self) -> None:
        if self.supporting < 0 or self.examined < self.supporting:
            raise ValueError("support must satisfy 0 <= supporting <= examined")


@dataclass(frozen=True, slots=True)
class Confidence:
    """A method-specific score, never an unqualified probability of truth."""

    score: float
    method: str
    support: Support | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("confidence score must be between 0 and 1")
        if not self.method:
            raise ValueError("confidence method is required")
