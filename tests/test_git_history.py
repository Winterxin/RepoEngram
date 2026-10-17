"""Git history extraction contracts backed by isolated temporary repositories."""

import json
import subprocess
from datetime import datetime

import pytest

from skillforge.domain.change import FileChangeKind
from skillforge.domain.serialization import to_json_value
from skillforge.errors import GitHistoryError
from skillforge.repository.git_history import GitHistoryMiner


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=check,
    )


def _init_repo(repo):
    _git(repo, "init", "-q", "-b", "main")


def _write(repo, path, content):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _commit(repo, subject, body=None):
    _git(repo, "add", "-A")
    args = [
        "-c",
        "user.name=SkillForge Test",
        "-c",
        "user.email=skillforge@example.invalid",
        "commit",
        "--quiet",
        "-m",
        subject,
    ]
    if body is not None:
        args.extend(["-m", body])
    _git(repo, *args)
    return _git(repo, "rev-parse", "HEAD").stdout.decode().strip()


def _latest(repo):
    return GitHistoryMiner().mine(repo, limit=1).changesets[0]


def _rewrite_numstat_output(miner, monkeypatch, rewrite):
    original_run_git = miner._run_git

    def run_git(root, *args, **kwargs):
        result = original_run_git(root, *args, **kwargs)
        if "--numstat" in args:
            result.stdout = rewrite(result.stdout)
        return result

    monkeypatch.setattr(miner, "_run_git", run_git)


def test_root_commit_has_no_parents_and_lists_added_files(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "README.md", "root\n")
    commit_id = _commit(tmp_path, "Initial snapshot")

    change = _latest(tmp_path)

    assert change.commit_id == commit_id
    assert change.parent_ids == ()
    assert change.is_merge is False
    assert [(item.path, item.kind) for item in change.files] == [
        ("README.md", FileChangeKind.ADDED)
    ]


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_shallow_boundary_preserves_real_parents_without_inventing_added_files(tmp_path,
                                                                            linked_worktree):
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    _write(origin, "base.py", "value = 1\n")
    root_commit = _commit(origin, "Initial")
    _write(origin, "base.py", "value = 2\n")
    boundary_commit = _commit(origin, "Change base")
    _write(origin, "new.py", "other = 3\n")
    head_commit = _commit(origin, "Add new")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "--quiet", "--depth", "2", origin.as_uri(), str(shallow)],
                   check=True, capture_output=True)
    target = shallow
    if linked_worktree:
        target = tmp_path / "linked"
        _git(shallow, "worktree", "add", "--quiet", "--detach", str(target), head_commit)
    history = GitHistoryMiner().mine(target, limit=50)
    head, boundary = history.changesets
    assert head.commit_id == head_commit
    assert [(change.path, change.kind) for change in head.files] == [
        ("new.py", FileChangeKind.ADDED)
    ]
    assert boundary.commit_id == boundary_commit
    assert boundary.parent_ids == (root_commit,)
    assert boundary.files == ()
    assert any("parent diff unavailable" in evidence.description for evidence in history.evidence)


def test_added_file_has_line_counts(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "base.txt", "base\n")
    _commit(tmp_path, "Initial")
    _write(tmp_path, "added.txt", "one\ntwo\n")
    _commit(tmp_path, "Add notes")

    change = _latest(tmp_path)

    assert change.files[0].kind == FileChangeKind.ADDED
    assert change.files[0].additions == 2
    assert change.files[0].deletions == 0


def test_modified_file_reports_additions_and_deletions(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "before\n")
    _commit(tmp_path, "Initial")
    _write(tmp_path, "file.txt", "after\n")
    _commit(tmp_path, "Replace content")

    file_change = _latest(tmp_path).files[0]

    assert file_change.kind == FileChangeKind.MODIFIED
    assert file_change.additions == 1
    assert file_change.deletions == 1


def test_deleted_file_is_reported(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "gone.txt", "content\n")
    _commit(tmp_path, "Initial")
    (tmp_path / "gone.txt").unlink()
    _commit(tmp_path, "Delete file")

    file_change = _latest(tmp_path).files[0]

    assert file_change.path == "gone.txt"
    assert file_change.kind == FileChangeKind.DELETED
    assert file_change.additions == 0
    assert file_change.deletions == 1


def test_renamed_file_keeps_both_paths(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "old name.txt", "same content\n")
    _commit(tmp_path, "Initial")
    (tmp_path / "old name.txt").rename(tmp_path / "new name.txt")
    _commit(tmp_path, "Rename file")

    file_change = _latest(tmp_path).files[0]

    assert file_change.kind == FileChangeKind.RENAMED
    assert file_change.path == "new name.txt"
    assert file_change.previous_path == "old name.txt"
    assert file_change.additions == 0
    assert file_change.deletions == 0


def test_one_commit_can_change_multiple_files(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "existing.txt", "before\n")
    _commit(tmp_path, "Initial")
    _write(tmp_path, "existing.txt", "after\n")
    _write(tmp_path, "new.txt", "new\n")
    _commit(tmp_path, "Update two files")

    change = _latest(tmp_path)

    assert {item.path for item in change.files} == {"existing.txt", "new.txt"}
    assert len(change.files) == 2


def test_commit_message_and_metadata_are_read(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "hello\n")
    commit_id = _commit(tmp_path, "Add file", "Second line of message")

    change = _latest(tmp_path)

    assert change.commit_id == commit_id
    assert change.message == "Add file\n\nSecond line of message\n"
    assert datetime.fromisoformat(change.committed_at).utcoffset() is not None


def test_unicode_path_and_commit_message_are_preserved(tmp_path):
    _init_repo(tmp_path)
    path = "文档/配置 文件.txt"
    _write(tmp_path, path, "第一行\n")
    _commit(tmp_path, "增加说明", "包含中文内容")

    change = _latest(tmp_path)

    assert change.message == "增加说明\n\n包含中文内容\n"
    assert change.files[0].path == path
    assert any(item.location.path == path for item in GitHistoryMiner().mine(tmp_path).evidence)


def test_parent_commit_is_reported(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "one\n")
    first = _commit(tmp_path, "First")
    _write(tmp_path, "file.txt", "two\n")
    second = _commit(tmp_path, "Second")

    history = GitHistoryMiner().mine(tmp_path, limit=2)

    assert [item.commit_id for item in history.changesets] == [second, first]
    assert history.changesets[0].parent_ids == (first,)


def test_limit_restricts_number_of_changesets(tmp_path):
    _init_repo(tmp_path)
    for index in range(4):
        _write(tmp_path, "file.txt", f"{index}\n")
        _commit(tmp_path, f"Commit {index}")

    assert len(GitHistoryMiner().mine(tmp_path, limit=2).changesets) == 2
    assert GitHistoryMiner().mine(tmp_path, limit=0).changesets == ()


def test_non_git_directory_raises_project_error(tmp_path):
    with pytest.raises(GitHistoryError, match="Git repository"):
        GitHistoryMiner().mine(tmp_path)


def test_empty_git_repository_returns_empty_history(tmp_path):
    _init_repo(tmp_path)

    history = GitHistoryMiner().mine(tmp_path)

    assert history.changesets == ()
    assert history.evidence == ()


def test_binary_file_has_unknown_line_counts(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "binary.bin").write_bytes(bytes([0, 1, 2, 3]))
    _commit(tmp_path, "Add binary file")

    history = GitHistoryMiner().mine(tmp_path)
    file_change = history.changesets[0].files[0]

    assert file_change.additions is None
    assert file_change.deletions is None
    numstat_evidence = [item for item in history.evidence if item.provenance == "git_numstat"]
    assert len(numstat_evidence) == 1
    assert numstat_evidence[0].evidence_id in history.changesets[0].evidence_ids
    evidence_by_id = {item.evidence_id: item for item in history.evidence}
    assert {evidence_by_id[item].provenance for item in file_change.evidence_ids} == {
        "git_diff", "git_numstat"
    }
    assert all(item in history.changesets[0].evidence_ids for item in file_change.evidence_ids)
    assert all(evidence_by_id[item].location.path == file_change.path
               for item in file_change.evidence_ids)
    assert numstat_evidence[0].confidence is None


def test_missing_numstat_entry_fails_extraction(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "before\n")
    _commit(tmp_path, "Initial")
    _write(tmp_path, "file.txt", "after\n")
    _commit(tmp_path, "Modify file")
    miner = GitHistoryMiner()
    _rewrite_numstat_output(miner, monkeypatch, lambda _: b"")

    with pytest.raises(GitHistoryError) as caught:
        miner.mine(tmp_path, limit=1)

    assert "missing numstat" in str(caught.value)
    assert "file.txt" in str(caught.value)


def test_unexpected_numstat_entry_fails_extraction(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "before\n")
    _commit(tmp_path, "Initial")
    _write(tmp_path, "file.txt", "after\n")
    _commit(tmp_path, "Modify file")
    miner = GitHistoryMiner()
    _rewrite_numstat_output(
        miner, monkeypatch, lambda output: output + b"1\t0\tunexpected.txt\0"
    )

    with pytest.raises(GitHistoryError) as caught:
        miner.mine(tmp_path, limit=1)

    assert "unexpected numstat" in str(caught.value)
    assert "unexpected.txt" in str(caught.value)


def test_history_result_serializes_to_json(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "hello\n")
    _commit(tmp_path, "Initial")

    payload = to_json_value(GitHistoryMiner().mine(tmp_path))

    assert json.loads(json.dumps(payload))["changesets"][0]["files"][0]["path"] == "file.txt"


def test_changeset_evidence_references_resolve_and_include_provenance(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "old.txt", "one\n")
    _commit(tmp_path, "Initial")
    (tmp_path / "old.txt").rename(tmp_path / "new.txt")
    _commit(tmp_path, "Rename")

    history = GitHistoryMiner().mine(tmp_path, limit=1)
    evidence_by_id = {item.evidence_id: item for item in history.evidence}
    change = history.changesets[0]

    assert change.evidence_ids
    assert all(evidence_id in evidence_by_id for evidence_id in change.evidence_ids)
    linked = [evidence_by_id[evidence_id] for evidence_id in change.evidence_ids]
    assert {item.provenance for item in linked} >= {"git_log", "git_diff", "git_numstat"}
    assert all(item.location.commit == change.commit_id for item in linked)
    assert any("old.txt" in item.description and "new.txt" in item.description for item in linked)


def test_detached_head_does_not_prevent_history_reading(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "file.txt", "one\n")
    commit_id = _commit(tmp_path, "Initial")
    _git(tmp_path, "checkout", "--quiet", "--detach", commit_id)

    assert _latest(tmp_path).commit_id == commit_id


def test_merge_commit_is_identified_and_uses_first_parent_diff(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "base.txt", "base\n")
    _commit(tmp_path, "Initial")
    _git(tmp_path, "checkout", "-q", "-b", "feature")
    _write(tmp_path, "feature.txt", "feature\n")
    _commit(tmp_path, "Feature")
    _git(tmp_path, "checkout", "-q", "main")
    _write(tmp_path, "main.txt", "main\n")
    _commit(tmp_path, "Main")
    _git(
        tmp_path,
        "-c",
        "user.name=SkillForge Test",
        "-c",
        "user.email=skillforge@example.invalid",
        "merge",
        "--no-ff",
        "--quiet",
        "feature",
        "-m",
        "Merge feature",
    )

    change = _latest(tmp_path)

    assert change.is_merge is True
    assert len(change.parent_ids) == 2
    assert [item.path for item in change.files] == ["feature.txt"]
