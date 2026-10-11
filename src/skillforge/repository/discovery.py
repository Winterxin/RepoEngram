"""Detect declared tooling and dependencies without inferring project conventions."""

import logging
import re
import tomllib
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SIGNAL_CATEGORIES = (
    "dependency_config",
    "test_directory",
    "pytest_config",
    "lint_config",
    "type_config",
    "ci_config",
    "container_config",
    "project_document",
)
FRAMEWORK_NAMES = {
    "fastapi": "FastAPI",
    "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy",
    "pytest": "pytest",
    "django": "Django",
    "flask": "Flask",
}


def read_pyproject(root: Path, files: set[str]) -> dict[str, object]:
    if "pyproject.toml" not in files:
        return {}
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        LOGGER.warning("Cannot parse pyproject.toml: %s", error)
        return {}


def discover_signals(
    files: set[str], directories: set[str], pyproject: dict[str, object]
) -> dict[str, set[str]]:
    signals = {category: set() for category in SIGNAL_CATEGORIES}
    tool = pyproject.get("tool", {})
    tool = tool if isinstance(tool, dict) else {}
    for path in files:
        name = path.rsplit("/", 1)[-1]
        lower = name.lower()
        if (
            path == "pyproject.toml"
            or lower
            in {"pipfile", "pipfile.lock", "poetry.lock", "uv.lock", "setup.py", "setup.cfg"}
            or re.fullmatch(r"requirements(?:[-_.][\w-]+)?\.txt", lower)
        ):
            signals["dependency_config"].add(path)
        if lower in {"pytest.ini", "tox.ini", "conftest.py"}:
            signals["pytest_config"].add(path)
        if lower in {"ruff.toml", ".ruff.toml", ".flake8", ".pylintrc", ".pre-commit-config.yaml"}:
            signals["lint_config"].add(path)
        if lower in {"mypy.ini", ".mypy.ini", "pyrightconfig.json"}:
            signals["type_config"].add(path)
        if (
            (path.startswith(".github/workflows/") and lower.endswith((".yml", ".yaml")))
            or path == ".circleci/config.yml"
            or lower in {".gitlab-ci.yml", "azure-pipelines.yml", "jenkinsfile"}
        ):
            signals["ci_config"].add(path)
        if (
            lower == "dockerfile"
            or lower.startswith("dockerfile.")
            or lower in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}
        ):
            signals["container_config"].add(path)
        if lower.startswith("readme") or lower in {
            "agents.md",
            "claude.md",
            "contributing.md",
            "architecture.md",
        }:
            signals["project_document"].add(path)
    if "pyproject.toml" in files:
        for key, category in (
            ("pytest", "pytest_config"),
            ("ruff", "lint_config"),
            ("black", "lint_config"),
            ("mypy", "type_config"),
            ("pyright", "type_config"),
        ):
            if key in tool:
                signals[category].add("pyproject.toml")
    signals["test_directory"].update(
        path for path in directories if path.rsplit("/", 1)[-1] in {"test", "tests"}
    )
    return signals


def detect_frameworks(
    root: Path, configs: set[str], pyproject: dict[str, object]
) -> tuple[str, ...]:
    dependencies: set[str] = set()
    project = pyproject.get("project", {})
    if isinstance(project, dict):
        for value in project.get("dependencies", []):
            if isinstance(value, str):
                dependencies.add(value)
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for group in optional.values():
                if isinstance(group, list):
                    dependencies.update(value for value in group if isinstance(value, str))
    for path in configs:
        if not path.rsplit("/", 1)[-1].startswith("requirements"):
            continue
        try:
            dependencies.update((root / path).read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeError) as error:
            LOGGER.warning("Cannot read %s: %s", path, error)
    names = set()
    for dependency in dependencies:
        match = re.match(r"^\s*([A-Za-z][\w.-]*)", dependency)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return tuple(FRAMEWORK_NAMES[name] for name in sorted(names & FRAMEWORK_NAMES.keys()))
