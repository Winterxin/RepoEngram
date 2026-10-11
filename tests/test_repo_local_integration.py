"""Filesystem sync fixtures, ownership, traversal and symlink safety."""

import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_skill_rendering import route
from test_skill_registry import skill

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.integration import AgentSkillExporter, RepoLocalSkillSync
from skillforge.integration.repo_local import _is_link


def plan(*keys, target=CodeAgentTarget.CODEX):
    return AgentSkillExporter().plan(route(tuple(skill(key) for key in keys)), target)


@pytest.mark.parametrize("target,prefix", [
    (CodeAgentTarget.CODEX, ".codex"),
    (CodeAgentTarget.CLAUDE_CODE, ".claude"),
])
def test_sync_layout_idempotence_stale_and_empty_route(tmp_path, target, prefix):
    sync = RepoLocalSkillSync()
    both = plan("a", "b", target=target)
    root = tmp_path / prefix / "skills"
    user = root / "user-skill"
    user.mkdir(parents=True)
    (user / "SKILL.md").write_text("user", encoding="utf-8")
    first = sync.sync(tmp_path, both)
    assert len(first.written_files) == 6
    for package in both.packages:
        directory = root / package.agent_skill_name
        assert (directory / "SKILL.md").is_file()
        assert (directory / "references" / "evidence.md").is_file()
        assert (directory / ".skillforge.json").is_file()
    second = sync.sync(tmp_path, both)
    assert second.written_files == second.removed_packages == ()
    assert len(second.unchanged_files) == 6
    one = plan("a", target=target)
    third = sync.sync(tmp_path, one)
    assert third.removed_packages == (next(p.agent_skill_name for p in both.packages
                                              if p.source_skill_id == "b"),)
    empty = plan(target=target)
    last = sync.sync(tmp_path, empty)
    assert len(last.removed_packages) == 1
    assert (user / "SKILL.md").read_text(encoding="utf-8") == "user"


def test_collision_malformed_marker_and_foreign_repo_are_safe(tmp_path):
    selected = plan("a")
    root = tmp_path / ".codex" / "skills"
    directory = root / selected.packages[0].agent_skill_name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("user", encoding="utf-8")
    with pytest.raises(ValueError, match="ownership marker"):
        RepoLocalSkillSync().sync(tmp_path, selected)
    assert (directory / "SKILL.md").read_text(encoding="utf-8") == "user"
    (directory / ".skillforge.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="ownership marker"):
        RepoLocalSkillSync().sync(tmp_path, selected)
    assert (directory / "SKILL.md").read_text(encoding="utf-8") == "user"
    # A valid-looking marker with another repository cannot authorize deletion.
    other_root = tmp_path / "other"
    other_root.mkdir()
    directory = other_root / ".codex" / "skills" / plan("b").packages[0].agent_skill_name
    RepoLocalSkillSync().sync(other_root, plan("b"))
    marker_path = directory / ".skillforge.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["repository_id"] = "foreign"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="ownership marker"):
        RepoLocalSkillSync().sync(other_root, plan())
    assert directory.exists()


def test_modified_managed_file_and_user_added_file_block_cleanup(tmp_path):
    selected = plan("a")
    RepoLocalSkillSync().sync(tmp_path, selected)
    directory = tmp_path / ".codex" / "skills" / selected.packages[0].agent_skill_name
    body = directory / "SKILL.md"
    body.write_text("user edit", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        RepoLocalSkillSync().sync(tmp_path, plan())
    assert body.read_text(encoding="utf-8") == "user edit"
    body.write_text(next(item.content for item in selected.packages[0].files
                         if item.relative_path == "SKILL.md"), encoding="utf-8")
    (directory / "notes.txt").write_text("user", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        RepoLocalSkillSync().sync(tmp_path, plan())
    assert (directory / "notes.txt").exists()
    (directory / "notes.txt").unlink()
    (directory / "user-dir").mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        RepoLocalSkillSync().sync(tmp_path, plan())
    assert (directory / "user-dir").exists()


def test_symlink_collision_and_root_escape_fail(tmp_path):
    outside = tmp_path.parent / "outside-skillforge-test"
    outside.mkdir(exist_ok=True)
    root = tmp_path / ".codex" / "skills"
    root.mkdir(parents=True)
    selected = plan("a")
    link = root / selected.packages[0].agent_skill_name
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="link or reparse"):
        RepoLocalSkillSync().sync(tmp_path, selected)
    assert list(outside.iterdir()) == []
    link.unlink()
    with pytest.raises(ValueError):
        RepoLocalSkillSync().sync(tmp_path, replace(selected, plan_id="fake"))
    assert not link.exists()


def test_python311_reparse_detection_without_path_is_junction():
    class FakePath:
        def __init__(self, mode, attributes=0, missing=False):
            self.mode = mode
            self.attributes = attributes
            self.missing = missing

        def lstat(self):
            if self.missing:
                raise FileNotFoundError
            return SimpleNamespace(st_mode=self.mode, st_file_attributes=self.attributes)

    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    assert _is_link(FakePath(stat.S_IFDIR, reparse))
    assert _is_link(FakePath(stat.S_IFREG, reparse))
    assert _is_link(FakePath(stat.S_IFLNK))  # A broken symlink still has an lstat mode.
    assert not _is_link(FakePath(stat.S_IFDIR))
    assert not _is_link(FakePath(stat.S_IFDIR, missing=True))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction integration")
@pytest.mark.parametrize("location", ["root", ".codex", "skills", "package", "references"])
def test_windows_junction_blocks_all_generated_path_layers(tmp_path, location):
    selected = plan("a")
    real_root = tmp_path / "real"
    real_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    name = selected.packages[0].agent_skill_name
    if location == "root":
        link = tmp_path / "linked-root"
        target = real_root
        sync_root = link
    elif location == ".codex":
        link = real_root / ".codex"
        target = outside
        sync_root = real_root
    elif location == "skills":
        (real_root / ".codex").mkdir()
        link = real_root / ".codex" / "skills"
        target = outside
        sync_root = real_root
    elif location == "package":
        (real_root / ".codex" / "skills").mkdir(parents=True)
        link = real_root / ".codex" / "skills" / name
        target = outside
        sync_root = real_root
    else:
        RepoLocalSkillSync().sync(real_root, selected)
        link = real_root / ".codex" / "skills" / name / "references"
        (link / "evidence.md").unlink()
        link.rmdir()
        target = outside
        sync_root = real_root
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip("Windows junction creation unavailable")
    try:
        assert _is_link(link)
        with pytest.raises(ValueError, match="link|reparse|real directory"):
            RepoLocalSkillSync().sync(sync_root, selected)
        assert sentinel.read_text(encoding="utf-8") == "untouched"
        assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]
    finally:
        os.rmdir(link)
