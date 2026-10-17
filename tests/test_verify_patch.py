"""Real Git worktree facts and content-sensitive snapshot identity."""

import os
import subprocess

import pytest

from skillforge.domain.verification import PatchChangeKind
from skillforge.repository.analyzer import RepositoryAnalyzer
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.identity import repository_identity
from skillforge.verify import GitPatchCollector


def git(root, *args):
    return subprocess.check_output(
        ("git", "-c", f"safe.directory={root}", *args), cwd=root
    ).decode().strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "a.txt").write_text("base", encoding="utf-8")
    (root / "rename.txt").write_text("rename me", encoding="utf-8")
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    return root


def snap(root, base="HEAD", excluded_paths=()):
    return GitPatchCollector().snapshot(root, repository_identity(root), base,
                                        excluded_paths=excluded_paths)


def test_identity_matches_existing_analyzer_and_history(repo):
    expected = repository_identity(repo)
    assert RepositoryAnalyzer().analyze(repo).repository.repository_id == expected
    assert GitHistoryMiner().mine(repo).repository_id == expected


def test_clean_and_tracked_staged_unstaged_final_state(repo):
    clean = snap(repo)
    assert clean.files == ()
    (repo / "a.txt").write_text("staged", encoding="utf-8")
    git(repo, "add", "a.txt")
    staged = snap(repo)
    assert staged.files[0].kind == PatchChangeKind.MODIFIED
    (repo / "a.txt").write_text("final", encoding="utf-8")
    final = snap(repo)
    assert final.files == staged.files
    assert final.snapshot_id != staged.snapshot_id
    assert snap(repo).to_dict() == final.to_dict()
    (repo / "a.txt").write_text("again", encoding="utf-8")
    assert snap(repo).snapshot_id != final.snapshot_id


def test_add_delete_rename_untracked_ignore_and_content(repo):
    (repo / "new.txt").write_text("added", encoding="utf-8")
    git(repo, "add", "new.txt")
    (repo / "a.txt").unlink()
    git(repo, "mv", "rename.txt", "renamed.txt")
    (repo / "loose.txt").write_text("one", encoding="utf-8")
    (repo / "ignored.txt").write_text("ignored", encoding="utf-8")
    first = snap(repo)
    kinds = {item.path: item.kind for item in first.files}
    assert kinds == {"a.txt": PatchChangeKind.DELETED, "new.txt": PatchChangeKind.ADDED,
                     "renamed.txt": PatchChangeKind.RENAMED,
                     "loose.txt": PatchChangeKind.UNTRACKED}
    renamed = next(item for item in first.files if item.path == "renamed.txt")
    assert renamed.previous_path == "rename.txt"
    (repo / "loose.txt").write_text("two", encoding="utf-8")
    assert snap(repo).snapshot_id != first.snapshot_id


def test_base_not_head_and_invalid_inputs(repo, tmp_path):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("committed", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "second")
    assert snap(repo, base).files[0].kind == PatchChangeKind.MODIFIED
    with pytest.raises(ValueError):
        snap(repo, "bad-ref")
    with pytest.raises(ValueError):
        GitPatchCollector().snapshot(repo, "wrong", base)
    with pytest.raises(ValueError):
        GitPatchCollector().snapshot(repo / ".git", repository_identity(repo / ".git"), base)
    child = repo / "child"
    child.mkdir()
    with pytest.raises(ValueError):
        snap(child, base)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError):
        snap(other)


def test_untracked_symlink_is_hashed_without_following(repo, tmp_path):
    target = tmp_path / "outside"
    target.write_text("private", encoding="utf-8")
    link = repo / "link"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    result = snap(repo)
    record = next(item for item in result.files if item.path == "link")
    assert record.kind == PatchChangeKind.UNTRACKED
    assert record.content_sha256 is not None
    assert os.readlink(link) == str(target)


def test_copy_and_index_deleted_file_reintroduced_as_untracked(repo):
    (repo / "copied.txt").write_text("rename me", encoding="utf-8")
    git(repo, "add", "copied.txt")
    copied = next(item for item in snap(repo).files if item.path == "copied.txt")
    assert copied.kind == PatchChangeKind.COPIED
    assert copied.previous_path == "rename.txt"
    git(repo, "rm", "--cached", "a.txt")
    records = tuple(item for item in snap(repo).files if item.path == "a.txt")
    assert {item.kind for item in records} == {
        PatchChangeKind.DELETED, PatchChangeKind.UNTRACKED
    }


def test_exact_exclusions_filter_tracked_diff_and_untracked_files(repo):
    managed = ".codex/skills/skillforge-" + "a" * 32
    tracked_managed = f"{managed}/SKILL.md"
    untracked_managed = f"{managed}/references/evidence.md"
    tracked_internal = (".skillforge/productization/prepare/old.json",
                        ".skillforge/productization/verify/old.json",
                        ".skillforge/productization/feedback/old.json")
    untracked_internal = (".skillforge/productization/prepare/new.json",
                          ".skillforge/productization/verify/new.json",
                          ".skillforge/productization/feedback/new.json")
    user_owned = ".codex/skills/my-own-skill/SKILL.md"
    for relative in (tracked_managed, *tracked_internal, user_owned, "checks.json", "delete.txt"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "tracked inputs")

    for relative in (tracked_managed, *tracked_internal, user_owned, "checks.json", "a.txt"):
        (repo / relative).write_text("changed", encoding="utf-8")
    (repo / tracked_internal[-1]).write_bytes(b"\x00binary feedback\xff")
    (repo / "delete.txt").unlink()
    git(repo, "mv", "rename.txt", "renamed.txt")
    for relative in (untracked_managed, *untracked_internal,
                     ".claude/skills/my-own-skill/SKILL.md", "agent-new.py"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new", encoding="utf-8")

    exclusions = (tracked_managed, untracked_managed, "checks.json")
    first = snap(repo, excluded_paths=exclusions)
    assert {item.path: item.kind for item in first.files} == {
        "a.txt": PatchChangeKind.MODIFIED,
        user_owned: PatchChangeKind.MODIFIED,
        ".claude/skills/my-own-skill/SKILL.md": PatchChangeKind.UNTRACKED,
        "agent-new.py": PatchChangeKind.UNTRACKED,
        "delete.txt": PatchChangeKind.DELETED,
        "renamed.txt": PatchChangeKind.RENAMED,
    }
    assert next(item for item in first.files if item.path == "renamed.txt").previous_path == (
        "rename.txt"
    )
    for relative in (tracked_managed, *tracked_internal, "checks.json", untracked_managed,
                     *untracked_internal):
        (repo / relative).write_text("changed again", encoding="utf-8")
    assert snap(repo, excluded_paths=exclusions).snapshot_id == first.snapshot_id


@pytest.mark.parametrize("excluded_paths", [
    ("../escape",), ("/absolute",), ("C:/absolute",), ("a\\b",),
    ("a//b",), ("a/./b",), ("a", "a"), ("z", "a"),
])
def test_exclusions_require_canonical_unique_paths(repo, excluded_paths):
    with pytest.raises(ValueError):
        snap(repo, excluded_paths=excluded_paths)
