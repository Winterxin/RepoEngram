"""Configuration for local, read-only repository analysis."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    # Exclude transient/vendor trees; the analyzer still includes ordinary hidden files.
    excluded_directories: frozenset[str] = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".artifacts",
            ".skillforge",
            "dist",
            "build",
        }
    )
    directory_summary_limit: int = 40

    def __post_init__(self) -> None:
        if self.directory_summary_limit < 0:
            raise ValueError("directory_summary_limit must be non-negative")
