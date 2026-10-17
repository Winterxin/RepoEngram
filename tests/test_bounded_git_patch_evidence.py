from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from skillforge.domain.change import FileChangeKind
from skillforge.repository.bounded_git_patch_evidence import (
    MAX_TOTAL_REFERENCE_BYTES,
    _first_small_complete_hunk,
    add_bounded_git_patch_evidence,
    validate_git_patch_evidence,
)
from skillforge.repository.git_history import GitHistoryMiner


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git_init(root)
    _git(root, "config", "user.name", "Patch Evidence Tests")
    _git(root, "config", "user.email", "patch-evidence@example.invalid")
    return root


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", message], check=True,
    )
    return _git(root, "rev-parse", "HEAD")


def _change(history, commit_id: str, path: str):
    changeset = next(item for item in history.changesets if item.commit_id == commit_id)
    change = next(item for item in changeset.files if item.path == path)
    return changeset, change


def _new_refs(result):
    return [item for item in result.evidence if item.provenance == "git_patch"]


def test_modified_unicode_path_is_bound_to_parent_commit_and_not_worktree(tmp_path):
    root = _repo(tmp_path)
    path = "src/功能.py"
    target = root / path
    target.parent.mkdir()
    target.write_text('message = "old\n"\n', encoding="utf-8")
    _commit(root, "initial")
    target.write_text('message = "新内容\n"\n', encoding="utf-8")
    commit_id = _commit(root, "modify unicode path")
    target.write_text('message = "WORKTREE CONTAMINATION\n"\n', encoding="utf-8")

    original = GitHistoryMiner().mine(root, limit=20)
    changeset, change = _change(original, commit_id, path)
    result = add_bounded_git_patch_evidence(root, original, [(commit_id, path)])

    refs = _new_refs(result)
    assert len(refs) == 1
    patch = refs[0]
    metadata_text, hunk = patch.description.split("\n", 1)
    assert json.loads(metadata_text) == {
        "parent": changeset.parent_ids[0],
        "commit": commit_id,
        "path": path,
        "previous_path": None,
    }
    assert '+message = "新内容' in hunk
    assert "WORKTREE CONTAMINATION" not in hunk
    assert patch.location.commit == commit_id
    assert validate_git_patch_evidence(patch, changeset, change) is True

    enriched_set, enriched_change = _change(result.history, commit_id, path)
    assert set(change.evidence_ids) <= set(enriched_change.evidence_ids)
    assert set(changeset.evidence_ids) <= set(enriched_set.evidence_ids)
    assert changeset.parent_ids == enriched_set.parent_ids
    assert change.additions == enriched_change.additions
    assert change.deletions == enriched_change.deletions
    assert len(patch.description.encode("utf-8")) > 0


def test_rename_patch_records_both_names_and_validates(tmp_path):
    root = _repo(tmp_path)
    old_path = root / "pkg" / "old.py"
    old_path.parent.mkdir()
    old_path.write_text("alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\n", encoding="utf-8")
    _commit(root, "initial")
    new_path = root / "pkg" / "new.py"
    old_path.rename(new_path)
    new_path.write_text("alpha = 1\nbeta = 20\ngamma = 3\ndelta = 4\n", encoding="utf-8")
    commit_id = _commit(root, "rename and edit")
    old_path.write_text("later recreation = True\n", encoding="utf-8")
    _commit(root, "recreate old path")
    history = GitHistoryMiner().mine(root, limit=20)
    changeset, change = _change(history, commit_id, "pkg/new.py")
    assert change.kind == FileChangeKind.RENAMED
    assert change.previous_path == "pkg/old.py"

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, change.path)])
    patch = _new_refs(result)[0]
    metadata_text, hunk = patch.description.split("\n", 1)
    assert json.loads(metadata_text)["previous_path"] == "pkg/old.py"
    assert "+beta = 20" in hunk
    assert "later recreation" not in hunk
    validate_git_patch_evidence(patch, changeset, change)


def test_hunk_selection_requires_matching_diff_file_headers():
    patch = (
        b"diff --git a/pkg/old.py b/pkg/old.py\n"
        b"--- a/pkg/old.py\n+++ b/pkg/old.py\n"
        b"@@ -1 +1 @@\n-old content\n+wrong file content\n"
        b"diff --git a/pkg/old.py b/pkg/new.py\n"
        b"similarity index 80%\nrename from pkg/old.py\nrename to pkg/new.py\n"
        b"--- a/pkg/old.py\n+++ b/pkg/new.py\n"
        b"@@ -1 +1 @@\n-old content\n+new content\n"
    )
    selected = _first_small_complete_hunk(
        patch,
        repository_id="repository",
        commit_id="a" * 40,
        parent_id="b" * 40,
        path="pkg/new.py",
        expected_old_path="pkg/old.py",
        previous_path="pkg/old.py",
    )
    assert selected is not None
    assert "+new content" in selected[0].description
    assert "+wrong file content" not in selected[0].description


def test_later_executable_change_beats_first_import_hunk(tmp_path):
    root = _repo(tmp_path)
    source = root / "service.py"
    filler = "\n".join(f"def stable_{n}(): return {n}" for n in range(8))
    source.write_text(
        "import os\n\n"
        "def transform(value):\n"
        "    if value > 0:\n"
        "        return value\n"
        "    return 0\n\n"
        f"{filler}\n\n"
        "def scale(value):\n"
        "    return value * 2\n",
        encoding="utf-8",
    )
    _commit(root, "initial implementation")
    source.write_text(
        "import pathlib\n\n"
        "def transform(value):\n"
        "    if value > 0:\n"
        "        return value\n"
        "    return 0\n\n"
        f"{filler}\n\n"
        "def scale(value):\n"
        "    return value * 3\n",
        encoding="utf-8",
    )
    commit_id = _commit(root, "change import and calculation")
    history = GitHistoryMiner().mine(root, limit=10)
    changeset, change = _change(history, commit_id, "service.py")

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "service.py")])
    patch = _new_refs(result)[0]
    hunk = patch.description.split("\n", 1)[1]

    assert "+    return value * 3" in hunk
    assert "import os" not in hunk
    assert "import pathlib" not in hunk
    validate_git_patch_evidence(patch, changeset, change)


def test_git_patch_keeps_the_complete_native_unified_three_hunk(tmp_path):
    root = _repo(tmp_path)
    source = root / "module.py"
    source.write_text(
        "def measure():\n"
        "    left = 1\n"
        "    top = 2\n"
        "    value = 3\n"
        "    bottom = 4\n"
        "    right = 5\n"
        "    return value\n",
        encoding="utf-8",
    )
    _commit(root, "initial implementation")
    source.write_text(source.read_text(encoding="utf-8").replace(
        "    value = 3\n", "    value = 30\n",
    ), encoding="utf-8")
    commit_id = _commit(root, "change calculation")
    history = GitHistoryMiner().mine(root, limit=10)
    changeset, change = _change(history, commit_id, "module.py")

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "module.py")])
    patch = _new_refs(result)[0]
    hunk = patch.description.split("\n", 1)[1]
    header = hunk.splitlines()[0]
    assert header.startswith("@@ -")
    assert ",7 " in header and " +" in header
    assert " def measure" in hunk
    assert sum(line.startswith(" ") for line in hunk.splitlines()[1:]) == 6
    validate_git_patch_evidence(patch, changeset, change)


def test_oversized_later_same_function_retries_with_compact_context(tmp_path):
    root = _repo(tmp_path)
    source = root / "module.py"
    long_a = "    a = '" + "a" * 400 + "'\n"
    long_b = "    b = '" + "b" * 400 + "'\n"
    long_c = "    c = '" + "c" * 400 + "'\n"
    long_d = "    d = '" + "d" * 400 + "'\n"
    content = (
        "def transform():\n"
        "    guard = 1\n"
        "    pad_one = 1\n"
        "    pad_two = 2\n"
        + long_a + long_b + "    summary = 1\n" + long_c + long_d
        + "    return summary\n"
    )
    source.write_text(content, encoding="utf-8")
    _commit(root, "initial implementation")
    source.write_text(content.replace("guard = 1", "guard = 2"), encoding="utf-8")
    older = _commit(root, "update guard")
    source.write_text(source.read_text(encoding="utf-8").replace(
        "guard = 2", "guard = 3",
    ).replace("summary = 1", "summary = 2"), encoding="utf-8")
    newer = _commit(root, "update guard and later summary")
    history = GitHistoryMiner().mine(root, limit=10)

    result = add_bounded_git_patch_evidence(
        root, history, [(older, "module.py"), (newer, "module.py")],
    )

    refs = _new_refs(result)
    assert result.git_reads == 6  # one compact-context retry; the 16-read cap is unchanged
    assert {item.location.commit for item in refs} == {older, newer}
    by_commit = {item.location.commit: item for item in refs}
    assert "guard = 2" in by_commit[older].description
    assert "guard = 3" in by_commit[newer].description
    for item in refs:
        changeset, change = _change(result.history, item.location.commit, item.location.path)
        validate_git_patch_evidence(item, changeset, change)


def test_import_and_comment_only_hunks_use_stable_first_fallback(tmp_path):
    root = _repo(tmp_path)
    source = root / "service.py"
    filler = "\n".join(f"def stable_{n}(): return {n}" for n in range(8))
    source.write_text(
        "import os\n\n"
        "def run():\n"
        "    return True\n\n"
        f"{filler}\n\n"
        "# original note\n",
        encoding="utf-8",
    )
    _commit(root, "initial")
    source.write_text(
        "import pathlib\n\n"
        "def run():\n"
        "    return True\n\n"
        f"{filler}\n\n"
        "# updated note\n",
        encoding="utf-8",
    )
    commit_id = _commit(root, "update import and note")
    history = GitHistoryMiner().mine(root, limit=10)

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "service.py")])
    hunk = _new_refs(result)[0].description.split("\n", 1)[1]

    assert "import os" in hunk
    assert "import pathlib" in hunk
    assert "updated note" not in hunk


@pytest.mark.parametrize("before,after", [
    ("    if value >= 0:", "    if value > 0:"),
    ("    except ValueError:", "    except (ValueError, TypeError):"),
    ("    raise ValueError('invalid')", "    raise TypeError('invalid')"),
])
def test_changed_boundary_beats_earlier_small_implementation(before, after):
    first = "@@ -1 +1 @@\n-limit = 1\n+limit = 2\n"
    boundary = f"@@ -20 +20 @@\n-{before}\n+{after}\n"
    patch = ("diff --git a/service.py b/service.py\n"
             "--- a/service.py\n+++ b/service.py\n" + first + boundary).encode()
    selected = _first_small_complete_hunk(
        patch, repository_id="repository", commit_id="a" * 40,
        parent_id="b" * 40, path="service.py", expected_old_path="service.py",
        previous_path=None,
    )
    assert selected is not None
    evidence, size = selected
    assert evidence.description.split("\n", 1)[1] == boundary
    assert evidence.location.line == 20
    assert size <= 1536


def test_stable_guard_context_does_not_displace_first_code_change():
    first = "@@ -1 +1 @@\n-limit = 1\n+limit = 2\n"
    later = ("@@ -20,2 +20,2 @@\n if value > 0:\n"
             "-    return value * 2\n+    return value * 3\n")
    patch = ("diff --git a/service.py b/service.py\n"
             "--- a/service.py\n+++ b/service.py\n" + first + later).encode()
    selected = _first_small_complete_hunk(
        patch, repository_id="repository", commit_id="a" * 40,
        parent_id="b" * 40, path="service.py", expected_old_path="service.py",
        previous_path=None,
    )
    assert selected is not None
    assert selected[0].description.split("\n", 1)[1] == first


def test_oversized_boundary_does_not_expand_budget_or_get_cut():
    first = "@@ -1 +1 @@\n-limit = 1\n+limit = 2\n"
    boundary = "@@ -20 +20 @@\n-if old:\n+if " + "condition and " * 160 + "value:\n"
    patch = ("diff --git a/service.py b/service.py\n"
             "--- a/service.py\n+++ b/service.py\n" + first + boundary).encode()
    selected = _first_small_complete_hunk(
        patch, repository_id="repository", commit_id="a" * 40,
        parent_id="b" * 40, path="service.py", expected_old_path="service.py",
        previous_path=None,
    )
    assert selected is not None
    assert selected[0].description.split("\n", 1)[1] == first
    assert selected[1] <= 1536


def test_concrete_transformation_call_is_not_displaced_by_later_branch():
    first = ("@@ -1 +1 @@\n-rationale = text[:140]\n"
             "+rationale = shorten_rationale(text)\n")
    later = "@@ -20 +20 @@\n-if selected:\n+if selected and available:\n"
    patch = ("diff --git a/service.py b/service.py\n"
             "--- a/service.py\n+++ b/service.py\n" + first + later).encode()
    selected = _first_small_complete_hunk(
        patch, repository_id="repository", commit_id="a" * 40,
        parent_id="b" * 40, path="service.py", expected_old_path="service.py",
        previous_path=None,
    )
    assert selected is not None
    assert selected[0].description.split("\n", 1)[1] == first
    assert selected[1] <= 1536


def test_literal_spacing_change_is_not_downranked_as_formatting(tmp_path):
    root = _repo(tmp_path)
    source = root / "service.py"
    filler = "\n".join(f"def stable_{n}(): return {n}" for n in range(8))
    source.write_text(
        f'import os\n\n{filler}\n\ndef label():\n    return "two  spaces"\n',
        encoding="utf-8",
    )
    _commit(root, "initial text")
    source.write_text(
        f'import pathlib\n\n{filler}\n\ndef label():\n    return "two spaces"\n',
        encoding="utf-8",
    )
    commit_id = _commit(root, "change import and literal content")
    history = GitHistoryMiner().mine(root, limit=10)
    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "service.py")])
    hunk = _new_refs(result)[0].description.split("\n", 1)[1]
    assert '+    return "two spaces"' in hunk
    assert "import pathlib" not in hunk


def test_whitespace_only_hunk_remains_eligible_as_fallback(tmp_path):
    root = _repo(tmp_path)
    source = root / "formatting.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    _commit(root, "initial formatting")
    source.write_text("def value():\n  return 1\n", encoding="utf-8")
    commit_id = _commit(root, "adjust indentation")
    history = GitHistoryMiner().mine(root, limit=10)
    changeset, change = _change(history, commit_id, "formatting.py")

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "formatting.py")])
    patch = _new_refs(result)[0]

    assert "return 1" in patch.description
    assert validate_git_patch_evidence(patch, changeset, change)


def test_validator_rejects_parent_path_line_and_hunk_tampering(tmp_path):
    root = _repo(tmp_path)
    path = root / "a.py"
    path.write_text("value = 1\n", encoding="utf-8")
    _commit(root, "initial")
    path.write_text("value = 2\n", encoding="utf-8")
    commit_id = _commit(root, "change")
    history = GitHistoryMiner().mine(root, limit=10)
    changeset, change = _change(history, commit_id, "a.py")
    patch = _new_refs(add_bounded_git_patch_evidence(
        root, history, [(commit_id, "a.py")],
    ))[0]
    variants = [
        replace(patch, description=patch.description.replace(changeset.parent_ids[0], "0" * 40)),
        replace(patch, location=replace(patch.location, path="other.py")),
        replace(patch, location=replace(patch.location, line=patch.location.line + 1)),
        replace(patch, description=patch.description.replace("+value = 2", "+forged = 2")),
    ]
    for tampered in variants:
        with pytest.raises(ValueError):
            validate_git_patch_evidence(tampered, changeset, change)


def test_invalid_or_forged_requests_are_rejected(tmp_path):
    root = _repo(tmp_path)
    path = root / "a.py"
    path.write_text("value = 1\n", encoding="utf-8")
    commit_id = _commit(root, "initial")
    history = GitHistoryMiner().mine(root, limit=10)
    for request in [("0" * 40, "a.py"), (commit_id, "../outside.py"),
                    (commit_id, "missing.py")]:
        with pytest.raises(ValueError):
            add_bounded_git_patch_evidence(root, history, [request])


def test_root_and_merge_changes_are_skipped_without_patches(tmp_path):
    root = _repo(tmp_path)
    (root / "root.py").write_text("root = 1\n", encoding="utf-8")
    root_commit = _commit(root, "root")
    default_branch = _git(root, "symbolic-ref", "--short", "HEAD")
    _git(root, "branch", "feature")
    _git(root, "checkout", "-q", "feature")
    (root / "feature.py").write_text("feature = 1\n", encoding="utf-8")
    feature_commit = _commit(root, "feature change")
    _git(root, "checkout", "-q", default_branch)
    (root / "main.py").write_text("main = 1\n", encoding="utf-8")
    _commit(root, "main change")
    subprocess.run(
        ["git", "-C", str(root), "merge", "--no-ff", "-q", "feature", "-m", "merge"],
        check=True,
    )
    merge_commit = _git(root, "rev-parse", "HEAD")
    history = GitHistoryMiner().mine(root, limit=20)
    requests = [(root_commit, "root.py"), (merge_commit, "feature.py")]
    result = add_bounded_git_patch_evidence(root, history, requests)
    assert not _new_refs(result)
    assert {(item.commit_id, item.reason) for item in result.skipped} == {
        (root_commit, "root_commit"),
        (merge_commit, "merge_commit"),
    }
    assert feature_commit in {item.commit_id for item in history.changesets}


def test_binary_and_oversize_hunks_are_skipped_without_truncation(tmp_path):
    root = _repo(tmp_path)
    binary = root / "data.bin"
    binary.write_bytes(b"\x00old\n")
    large = root / "large.txt"
    large.write_text("old\n", encoding="utf-8")
    huge = root / "huge.txt"
    huge.write_text("old\n", encoding="utf-8")
    _commit(root, "initial")
    binary.write_bytes(b"\x00new\n")
    large.write_text("x" * 1800 + "\n", encoding="utf-8")
    huge.write_text("y" * 70_000 + "\n", encoding="utf-8")
    commit_id = _commit(root, "binary and large changes")
    history = GitHistoryMiner().mine(root, limit=10)
    result = add_bounded_git_patch_evidence(
        root, history, [
            (commit_id, "data.bin"), (commit_id, "large.txt"), (commit_id, "huge.txt"),
        ],
    )
    assert not _new_refs(result)
    assert len(result.skipped) == 3
    assert {item.reason for item in result.skipped} == {
        "no_short_complete_text_hunk", "stdout_budget",
    }


def test_external_diff_and_textconv_configuration_cannot_change_evidence(tmp_path):
    root = _repo(tmp_path)
    attributes = root / ".gitattributes"
    attributes.write_text("*.txt diff=unavailable\n", encoding="utf-8")
    target = root / "a.txt"
    target.write_text("before\n", encoding="utf-8")
    _commit(root, "initial")
    _git(root, "config", "diff.external", "does-not-exist-external-diff")
    _git(root, "config", "diff.unavailable.textconv", "does-not-exist-textconv")
    target.write_text("after\n", encoding="utf-8")
    commit_id = _commit(root, "modify text")
    history = GitHistoryMiner().mine(root, limit=10)

    result = add_bounded_git_patch_evidence(root, history, [(commit_id, "a.txt")])
    refs = _new_refs(result)
    assert len(refs) == 1
    assert "+after" in refs[0].description


def test_parent_validation_is_reused_for_files_from_the_same_commit(tmp_path):
    root = _repo(tmp_path)
    paths = [f"pkg/file_{index}.py" for index in range(12)]
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = 1\n", encoding="utf-8")
    _commit(root, "initial files")
    for path in paths:
        (root / path).write_text("value = 2\n", encoding="utf-8")
    commit_id = _commit(root, "modify all files")
    history = GitHistoryMiner().mine(root, limit=10)

    result = add_bounded_git_patch_evidence(
        root, history, [(commit_id, path) for path in paths],
    )
    assert len(_new_refs(result)) == len(paths)
    assert result.git_reads == 1 + 1 + len(paths)
    assert not result.skipped


def test_read_and_reference_budgets_and_incremental_idempotence(tmp_path):
    root = _repo(tmp_path)
    path = root / "counter.py"
    path.write_text("count = 0\n", encoding="utf-8")
    _commit(root, "initial")
    commits = []
    for value in range(1, 10):
        path.write_text(f"count = {value}\n", encoding="utf-8")
        commits.append(_commit(root, f"count {value}"))
    history = GitHistoryMiner().mine(root, limit=20)
    requests = [(commit_id, "counter.py") for commit_id in commits]
    result = add_bounded_git_patch_evidence(root, history, requests)
    refs = _new_refs(result)
    assert result.git_reads <= 16
    assert result.added_reference_bytes <= MAX_TOTAL_REFERENCE_BYTES
    assert len(refs) == 7
    assert {item.location.commit for item in refs} == set(commits[-7:])
    assert sum(item.reason == "git_read_budget" for item in result.skipped) == 2
    for item in refs:
        changeset = next(cs for cs in result.history.changesets
                         if cs.commit_id == item.location.commit)
        change = next(fc for fc in changeset.files if fc.path == item.location.path)
        validate_git_patch_evidence(item, changeset, change)

    again = add_bounded_git_patch_evidence(root, result.history, requests)
    assert again.evidence == ()
    assert again.added_reference_bytes == 0
    assert again.history == result.history


def test_incomplete_hunk_cannot_relabel_a_later_file():
    patch = (
        b"diff --git a/target.py b/target.py\n--- a/target.py\n+++ b/target.py\n"
        b"@@ -1,3 +1,3 @@\n-old\n+new\n"
        b"diff --git a/other.py b/other.py\n--- a/other.py\n+++ b/other.py\n"
        b"@@ -1 +1 @@\n-other_old\n+other_new\n"
    )
    assert _first_small_complete_hunk(
        patch, repository_id="example", commit_id="a" * 40, parent_id="b" * 40,
        path="target.py", expected_old_path="target.py", previous_path=None,
    ) is None
