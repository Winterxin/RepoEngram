"""Orchestrate read-only inventory, config discovery, and Git metadata probes."""

import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

from skillforge.config import AnalyzerConfig
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.domain.repository import (
    LanguageStat,
    PythonModule,
    RepositoryProfile,
    RepositoryRef,
    RepositorySignal,
)
from skillforge.errors import InvalidRepositoryPath, RepositoryReadError
from skillforge.repository.discovery import (
    SIGNAL_CATEGORIES,
    detect_frameworks,
    discover_signals,
    read_pyproject,
)
from skillforge.repository.git_metadata import read_git_metadata
from skillforge.repository.identity import repository_identity

LANGUAGES = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".sql": "SQL",
    ".sh": "Shell",
    ".ps1": "PowerShell",
    ".md": "Markdown",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".xml": "XML",
    ".txt": "Text",
    ".html": "HTML",
    ".css": "CSS",
    ".rst": "reStructuredText",
}


class RepositoryAnalyzer:
    """Inventory one local directory. Future AST analysis can consume this profile."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()

    def analyze(self, path: Path | str) -> RepositoryProfile:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")
        repository_id = repository_identity(root)
        reference = RepositoryRef(repository_id=repository_id, root=root)
        files, directories = self._inventory(root)
        file_names = {item.as_posix() for item in files}
        pyproject = read_pyproject(root, file_names)
        signals = discover_signals(file_names, directories, pyproject)
        evidence_by_id: dict[str, Evidence] = {}
        signal_models: list[RepositorySignal] = []
        for category in SIGNAL_CATEGORIES:
            paths = tuple(sorted(signals[category]))
            ids = tuple(
                self._record_evidence(reference, item, category, evidence_by_id) for item in paths
            )
            signal_models.append(RepositorySignal(category, paths, ids))
        git = read_git_metadata(root)
        if git.head_commit:
            self._record_evidence(reference, ".", "git_head", evidence_by_id, git.head_commit)
        return RepositoryProfile(
            repository=reference,
            file_count=len(files),
            languages=self._language_stats(files),
            frameworks=detect_frameworks(root, signals["dependency_config"], pyproject),
            top_level_directories=tuple(sorted(d for d in directories if "/" not in d)),
            directory_summary=self._directory_summary(files, directories),
            python_modules=self._python_modules(files),
            signals=tuple(signal_models),
            git=git,
            evidence=tuple(sorted(evidence_by_id.values(), key=lambda item: item.evidence_id)),
        )

    def _inventory(self, root: Path) -> tuple[list[Path], set[str]]:
        files: list[Path] = []
        directories: set[str] = set()

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
            for name in names:
                directories.add((current_path / name).relative_to(root).as_posix())
            for name in sorted(filenames):
                item = current_path / name
                if not item.is_symlink() and item.is_file():
                    files.append(item.relative_to(root))
        return sorted(files, key=lambda item: item.as_posix()), directories

    @staticmethod
    def _record_evidence(
        repo: RepositoryRef,
        path: str,
        category: str,
        store: dict[str, Evidence],
        commit: str | None = None,
    ) -> str:
        if category == "git_head":
            kind = EvidenceType.GIT
            provenance = "git_rev_parse"
        elif category == "test_directory":
            kind = EvidenceType.TEST
            provenance = "filesystem_inventory"
        elif category == "project_document":
            kind = EvidenceType.DOCUMENT
            provenance = "filesystem_inventory"
        else:
            kind = EvidenceType.CONFIG
            provenance = "filesystem_inventory"
        key = f"{repo.repository_id}:{kind.value}:{path}:{commit or ''}"
        evidence_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        store[evidence_id] = Evidence(
            evidence_id,
            repo.repository_id,
            kind,
            SourceLocation(path, commit),
            f"Observed {path}",
            provenance,
        )
        return evidence_id

    @staticmethod
    def _language_stats(files: list[Path]) -> tuple[LanguageStat, ...]:
        examples: dict[str, list[str]] = defaultdict(list)
        counts: Counter[str] = Counter()
        for path in files:
            language = LANGUAGES.get(path.suffix.lower(), "Other")
            counts[language] += 1
            if len(examples[language]) < 3:
                examples[language].append(path.as_posix())
        return tuple(
            LanguageStat(name, count, tuple(examples[name]))
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

    def _directory_summary(self, files: list[Path], directories: set[str]) -> tuple[str, ...]:
        counts: Counter[str] = Counter()
        for file in files:
            parts = file.parts[:-1]
            for depth in range(1, len(parts) + 1):
                counts[Path(*parts[:depth]).as_posix()] += 1
        selected = sorted(directories, key=lambda path: (-counts[path], path))
        return tuple(
            f"{path}/ ({counts[path]} files)"
            for path in selected[: self.config.directory_summary_limit]
        )

    @staticmethod
    def _python_modules(files: list[Path]) -> tuple[PythonModule, ...]:
        modules = []
        for path in files:
            if path.suffix != ".py":
                continue
            parts = list(path.parts)
            if parts and parts[0] == "src":
                parts.pop(0)
            is_package = parts[-1] == "__init__.py"
            parts[-1] = "" if is_package else path.stem
            name_parts = [part for part in parts if part]
            if name_parts and all(part.isidentifier() for part in name_parts):
                modules.append(PythonModule(".".join(name_parts), path.as_posix(), is_package))
        return tuple(modules)
