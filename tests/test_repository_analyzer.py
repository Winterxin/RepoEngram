import json
import subprocess

import pytest

from skillforge.domain.evidence import EvidenceType
from skillforge.errors import InvalidRepositoryPath
from skillforge.repository.analyzer import RepositoryAnalyzer


def _write(root, path, content=""):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _signals(profile):
    return {signal.category: signal for signal in profile.signals}


def test_invalid_path(tmp_path):
    with pytest.raises(InvalidRepositoryPath):
        RepositoryAnalyzer().analyze(tmp_path / "missing")
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    with pytest.raises(InvalidRepositoryPath):
        RepositoryAnalyzer().analyze(file_path)


def test_empty_non_git_directory(tmp_path):
    profile = RepositoryAnalyzer().analyze(tmp_path)
    assert profile.file_count == 0
    assert profile.languages == ()
    assert profile.python_modules == ()
    assert profile.git.is_git_repository is False
    assert all(not signal.paths for signal in profile.signals)
    assert json.loads(json.dumps(profile.to_dict()))["repository"]["root"] == str(tmp_path)


def test_repository_discovery_and_evidence(tmp_path):
    _write(
        tmp_path,
        "pyproject.toml",
        """[project]
dependencies = ["fastapi>=0.1", "pydantic>=2"]
[tool.pytest.ini_options]
testpaths = ["tests"]
[tool.ruff]
line-length = 100
[tool.mypy]
strict = true
""",
    )
    _write(tmp_path, "src/app/__init__.py")
    _write(tmp_path, "src/app/api.py", "def run(): pass\n")
    _write(tmp_path, "tests/test_api.py", "def test_api(): pass\n")
    _write(tmp_path, ".github/workflows/ci.yml", "name: CI\n")
    _write(tmp_path, "Dockerfile", "FROM python:3.12\n")
    _write(tmp_path, "README.md", "Project\n")
    profile = RepositoryAnalyzer().analyze(tmp_path)
    signals = _signals(profile)
    assert profile.file_count == 7
    assert profile.frameworks == ("FastAPI", "Pydantic")
    assert ("app", "src/app/__init__.py", True) in {
        (module.module, module.path, module.is_package) for module in profile.python_modules
    }
    assert "tests" in signals["test_directory"].paths
    assert "pyproject.toml" in signals["pytest_config"].paths
    assert "pyproject.toml" in signals["lint_config"].paths
    assert "pyproject.toml" in signals["type_config"].paths
    assert signals["ci_config"].paths == (".github/workflows/ci.yml",)
    assert signals["container_config"].paths == ("Dockerfile",)
    assert signals["project_document"].paths == ("README.md",)
    assert signals["dependency_config"].evidence_ids
    assert all(
        ref in {item.evidence_id for item in profile.evidence}
        for signal in profile.signals
        for ref in signal.evidence_ids
    )
    assert {item.kind for item in profile.evidence} >= {
        EvidenceType.CONFIG,
        EvidenceType.TEST,
        EvidenceType.DOCUMENT,
    }


def test_git_metadata_from_temporary_repository(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    _write(tmp_path, "README.md", "hello\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Initial",
        ],
        check=True,
    )
    profile = RepositoryAnalyzer().analyze(tmp_path)
    assert profile.git.is_git_repository is True
    assert profile.git.branch == "main"
    assert len(profile.git.head_commit) == 40
    assert any(
        item.kind == EvidenceType.GIT and item.location.commit == profile.git.head_commit
        for item in profile.evidence
    )


def test_non_git_subdirectory_inside_git_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    child = tmp_path / "child"
    child.mkdir()
    assert RepositoryAnalyzer().analyze(child).git.is_git_repository is False


def test_transient_directories_are_excluded(tmp_path):
    _write(tmp_path, "src/app.py", "print('ok')\n")
    _write(tmp_path, ".artifacts/install/generated.py", "x = 1\n")
    _write(tmp_path, "src/skillforge.egg-info/PKG-INFO", "generated\n")
    profile = RepositoryAnalyzer().analyze(tmp_path)
    assert profile.file_count == 1
    assert profile.top_level_directories == ("src",)
