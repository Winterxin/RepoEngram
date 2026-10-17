import json
import pickle
import subprocess

import pytest

import evals.coherent_history_request as request_builder
from evals.coherent_history_request import build_current_source_request
from skillforge.repository.bounded_git_patch_evidence import validate_git_patch_evidence


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
    )


def _write(repo, path, content):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=SkillForge Test", "-c",
         "user.email=skillforge@example.invalid", "commit", "--quiet", "-m", message)


def _repository(repo):
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/a.py", "value = 1\n")
    _write(repo, "pkg/b.py", "value = 1\n")
    _commit(repo, "Initial modules")
    for value in (2, 3):
        _write(repo, "pkg/a.py", f"value = {value}\n")
        _write(repo, "pkg/b.py", f"value = {value}\n")
        _commit(repo, f"Update paired modules {value}")


def test_production_path_builds_patch_enriched_request_and_respects_module_scope(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repository(repo)

    output = tmp_path / "matched"
    result = build_current_source_request(
        repo, output, history_limit=50, current_module="pkg.a", workflow_limit=1,
    )
    assert result["status"] == "ready"
    assert result["head"]
    request = json.loads((output / "request.json").read_text("utf-8"))
    assert request["current_modules"] == ["pkg.a", "pkg.b"]
    patch_ids = {
        item["evidence_id"] for item in request["evidence"]
        if item["provenance"] == "git_patch"
    }
    assert patch_ids
    history = pickle.loads((output / "history.pkl").read_bytes())
    evidence = {item.evidence_id: item for item in history.evidence}
    changes = {
        (commit.commit_id, change.path): (commit, change)
        for commit in history.changesets for change in commit.files
    }
    for evidence_id in patch_ids:
        item = evidence[evidence_id]
        commit, change = changes[(item.location.commit, item.location.path)]
        assert validate_git_patch_evidence(item, commit, change)
    assert result["request_git_patch_evidence_count"] == len(patch_ids)

    no_match = build_current_source_request(
        repo, tmp_path / "unmatched", current_module="pkg.unrelated", workflow_limit=1,
    )
    assert no_match["status"] == "no_match"
    assert no_match["request"] is None


def test_false_patch_validation_prevents_ready_request(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repository(repo)
    output = tmp_path / "rejected"
    monkeypatch.setattr(request_builder, "validate_git_patch_evidence", lambda *args: False)

    with pytest.raises(ValueError, match="git_patch Evidence failed validation"):
        build_current_source_request(
            repo, output, current_module="pkg.a", workflow_limit=1,
        )

    assert not (output / "request.json").exists()
    assert not (output / "summary.json").exists()
