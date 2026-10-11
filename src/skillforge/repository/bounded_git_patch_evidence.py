"""Read a bounded amount of complete historical Git patch evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from skillforge.domain.change import (
    ChangeSet,
    FileChange,
    FileChangeKind,
    GitHistoryResult,
)
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.repository.identity import repository_identity

MAX_GIT_READS = 16
MAX_REFERENCE_BYTES = 1536
MAX_TOTAL_REFERENCE_BYTES = 16 * 1024
MAX_COMMAND_STDOUT_BYTES = 64 * 1024
GIT_TIMEOUT_SECONDS = 3.0
_COMMIT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?:.*?)(?:\r?\n)?$"
)


@dataclass(frozen=True, slots=True)
class GitPatchSkip:
    commit_id: str
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class BoundedGitPatchEvidenceResult:
    history: GitHistoryResult
    evidence: tuple[Evidence, ...]
    skipped: tuple[GitPatchSkip, ...]
    git_reads: int
    added_reference_bytes: int


@dataclass(frozen=True, slots=True)
class _GitRead:
    stdout: bytes
    returncode: int
    truncated: bool


def add_bounded_git_patch_evidence(
    root: Path | str,
    history: GitHistoryResult,
    requests: Iterable[tuple[str, str]],
) -> BoundedGitPatchEvidenceResult:
    """Attach at most one small, complete patch hunk for each requested history file.

    Requests must be exact ``(commit_id, FileChange.path)`` pairs already present in
    ``history``. Only MODIFIED and RENAMED file changes are eligible in this version.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError("Git repository root must be an existing directory")
    if history.repository_id != repository_identity(root_path):
        raise ValueError("Git history repository_id does not match root")

    changesets: dict[str, ChangeSet] = {}
    for changeset in history.changesets:
        if changeset.commit_id in changesets:
            raise ValueError("Git history contains duplicate commit IDs")
        if changeset.repository_id != history.repository_id:
            raise ValueError("Git history changeset repository_id mismatch")
        changesets[changeset.commit_id] = changeset

    unique_requests: list[tuple[str, str, ChangeSet, FileChange]] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        if (not isinstance(request, tuple) or len(request) != 2
                or not all(isinstance(part, str) for part in request)):
            raise ValueError("Git patch request must be a (commit_id, path) pair")
        commit_id, path = request
        _validate_commit_id(commit_id)
        _validate_repo_path(path)
        if request in seen:
            continue
        seen.add(request)
        changeset = changesets.get(commit_id)
        if changeset is None:
            raise ValueError("Git patch request commit is not present in history")
        matches = [change for change in changeset.files if change.path == path]
        if len(matches) != 1:
            raise ValueError("Git patch request path is not an exact FileChange path")
        change = matches[0]
        if change.previous_path is not None:
            _validate_repo_path(change.previous_path)
        unique_requests.append((commit_id, path, changeset, change))

    skipped: list[GitPatchSkip] = []
    eligible: list[tuple[str, str, ChangeSet, FileChange]] = []
    for item in unique_requests:
        commit_id, path, changeset, change = item
        if changeset.is_merge:
            skipped.append(GitPatchSkip(commit_id, path, "merge_commit"))
        elif not changeset.parent_ids:
            skipped.append(GitPatchSkip(commit_id, path, "root_commit"))
        elif len(changeset.parent_ids) != 1:
            raise ValueError("Non-merge history entry must have exactly one parent")
        elif change.kind not in {FileChangeKind.MODIFIED, FileChangeKind.RENAMED}:
            skipped.append(GitPatchSkip(commit_id, path, "unsupported_file_change_kind"))
        else:
            _validate_commit_id(changeset.parent_ids[0])
            eligible.append(item)

    if not eligible:
        return BoundedGitPatchEvidenceResult(history, (), tuple(skipped), 0, 0)

    # Reserve one root read, one parent read per distinct commit, and one patch read
    # per requested file. Reusing a verified parent lets same-commit requests fit.
    bounded_eligible: list[tuple[str, str, ChangeSet, FileChange]] = []
    verified_commits: set[str] = set()
    reserved_reads = 1
    for item in eligible:
        commit_id, path, _, _ = item
        needed = 1 + (commit_id not in verified_commits)
        if reserved_reads + needed > MAX_GIT_READS:
            skipped.append(GitPatchSkip(commit_id, path, "git_read_budget"))
            continue
        bounded_eligible.append(item)
        reserved_reads += needed
        verified_commits.add(commit_id)
    eligible = bounded_eligible

    reads = 0
    root_read = _run_git_limited(root_path, ("rev-parse", "--show-toplevel"), 4096)
    reads += 1
    if root_read.truncated or root_read.returncode != 0:
        raise ValueError("Could not verify Git repository root")
    try:
        actual_root = Path(root_read.stdout.decode("utf-8", errors="strict").strip()).resolve()
    except (UnicodeDecodeError, OSError) as error:
        raise ValueError("Git returned an invalid repository root") from error
    if os.path.normcase(str(actual_root)) != os.path.normcase(str(root_path)):
        raise ValueError("Pass the exact Git repository root")

    existing_by_id: dict[str, Evidence] = {}
    for item in history.evidence:
        previous = existing_by_id.setdefault(item.evidence_id, item)
        if previous != item:
            raise ValueError("Git history has conflicting Evidence IDs")

    additions_by_file: dict[tuple[str, str], list[Evidence]] = {}
    new_evidence_by_id: dict[str, Evidence] = {}
    total_new_bytes = 0
    verified_parent_ids: dict[str, str] = {}
    for commit_id, path, changeset, change in eligible:
        parent_id = changeset.parent_ids[0]
        if commit_id not in verified_parent_ids:
            parent_read = _run_git_limited(
                root_path, ("rev-list", "--parents", "-n", "1", commit_id), 4096,
            )
            reads += 1
            if parent_read.truncated or parent_read.returncode != 0:
                raise ValueError(f"Cannot verify commit parent for {commit_id}")
            try:
                commit_record = parent_read.stdout.decode(
                    "ascii", errors="strict",
                ).strip().split()
            except UnicodeDecodeError as error:
                raise ValueError("Git returned a malformed commit-parent record") from error
            if commit_record != [commit_id, parent_id]:
                raise ValueError(f"History parent does not match Git for {commit_id}")
            verified_parent_ids[commit_id] = parent_id
        elif verified_parent_ids[commit_id] != parent_id:
            raise ValueError(f"History parent is inconsistent for {commit_id}")

        pathspecs = [path]
        if change.previous_path is not None and change.previous_path != path:
            pathspecs.append(change.previous_path)
        diff_args = (
            "--literal-pathspecs", "diff", "--no-ext-diff", "--no-textconv",
            "--no-color", "--no-indent-heuristic", "--find-renames=50%",
            "--unified=3", parent_id, commit_id, "--", *pathspecs,
        )
        patch_read = _run_git_limited(root_path, diff_args, MAX_COMMAND_STDOUT_BYTES)
        reads += 1
        if patch_read.returncode != 0 and not patch_read.truncated:
            raise ValueError(f"Cannot read historical patch for {commit_id}:{path}")
        hunk = _first_small_complete_hunk(
            patch_read.stdout, repository_id=history.repository_id,
            commit_id=commit_id, parent_id=parent_id,
            path=path,
            expected_old_path=change.previous_path or path,
            previous_path=change.previous_path,
        )
        if hunk is None:
            reason = "stdout_budget" if patch_read.truncated else "no_short_complete_text_hunk"
            skipped.append(GitPatchSkip(commit_id, path, reason))
            continue
        evidence, serialized_size = hunk
        existing = existing_by_id.get(evidence.evidence_id)
        if existing is not None:
            if existing != evidence:
                raise ValueError("Existing git_patch Evidence conflicts with Git content")
            evidence = existing
        else:
            if total_new_bytes + serialized_size > MAX_TOTAL_REFERENCE_BYTES:
                skipped.append(GitPatchSkip(commit_id, path, "reference_byte_budget"))
                continue
            new_evidence_by_id[evidence.evidence_id] = evidence
            existing_by_id[evidence.evidence_id] = evidence
            total_new_bytes += serialized_size
        additions_by_file.setdefault((commit_id, path), []).append(evidence)

    if reads > MAX_GIT_READS:
        raise AssertionError("Git read budget implementation error")
    if not additions_by_file:
        return BoundedGitPatchEvidenceResult(history, (), tuple(skipped), reads, 0)

    enriched_changesets: list[ChangeSet] = []
    for changeset in history.changesets:
        updated_files = []
        change_ids: list[str] = []
        for change in changeset.files:
            additions = additions_by_file.get((changeset.commit_id, change.path), ())
            new_ids = tuple(sorted({item.evidence_id for item in additions}))
            updated = replace(
                change,
                evidence_ids=tuple(dict.fromkeys((*change.evidence_ids, *new_ids))),
            )
            updated_files.append(updated)
            change_ids.extend(new_ids)
        updated_change_ids = tuple(
            dict.fromkeys((*changeset.evidence_ids, *sorted(set(change_ids))))
        )
        enriched_changesets.append(replace(
            changeset, files=tuple(updated_files), evidence_ids=updated_change_ids,
        ))

    all_evidence = dict(existing_by_id)
    enriched_history = GitHistoryResult(
        history.repository_id,
        tuple(enriched_changesets),
        tuple(sorted(all_evidence.values(), key=lambda item: item.evidence_id)),
    )
    return BoundedGitPatchEvidenceResult(
        enriched_history,
        tuple(sorted(new_evidence_by_id.values(), key=lambda item: item.evidence_id)),
        tuple(skipped),
        reads,
        total_new_bytes,
    )


def validate_git_patch_evidence(
    item: Evidence, changeset: ChangeSet, change: FileChange
) -> bool:
    """Validate patch metadata, object binding, complete hunk counts, and location."""
    if (item.kind != EvidenceType.GIT or item.provenance != "git_patch"
            or item.repository_id != changeset.repository_id
            or item.location.path != change.path):
        raise ValueError("git_patch Evidence provenance or path is invalid")
    if (changeset.is_merge or len(changeset.parent_ids) != 1
            or change.kind not in {FileChangeKind.MODIFIED, FileChangeKind.RENAMED}):
        raise ValueError("git_patch Evidence is not attached to an eligible change")
    try:
        metadata_text, hunk_text = item.description.split("\n", 1)
        metadata = json.loads(metadata_text)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError("git_patch Evidence metadata is malformed") from error
    parent_id = changeset.parent_ids[0]
    expected = {
        "parent": parent_id,
        "commit": changeset.commit_id,
        "path": change.path,
        "previous_path": change.previous_path,
    }
    if metadata != expected:
        raise ValueError("git_patch Evidence commit/path/parent metadata mismatch")
    parsed = _parse_single_hunk(hunk_text)
    if parsed is None:
        raise ValueError("git_patch Evidence does not contain one complete hunk")
    old_start, _, new_start, new_count = parsed
    if new_count < 1 or item.location.commit != changeset.commit_id:
        raise ValueError("git_patch Evidence location is not on the target commit")
    if item.location.line != max(1, new_start):
        raise ValueError("git_patch Evidence line does not match its hunk")
    expected_id = _evidence_id(
        item.repository_id, parent_id, changeset.commit_id, change.path, hunk_text.encode("utf-8"),
    )
    if item.evidence_id != expected_id:
        raise ValueError("git_patch Evidence ID does not bind the original hunk bytes")
    if _serialized_evidence_size(item) > MAX_REFERENCE_BYTES:
        raise ValueError("git_patch Evidence exceeds the per-reference byte limit")
    return True


def _first_small_complete_hunk(
    patch: bytes,
    *,
    repository_id: str,
    commit_id: str,
    parent_id: str,
    path: str,
    expected_old_path: str,
    previous_path: str | None,
) -> tuple[Evidence, int] | None:
    try:
        text = patch.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines(keepends=True)
    index = 0
    hunk_order = 0
    selected: tuple[int, int, Evidence, int] | None = None
    old_header_path: str | None = None
    new_header_path: str | None = None
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git ") or line.startswith("diff --cc "):
            old_header_path = None
            new_header_path = None
            index += 1
            continue
        if line.startswith("--- "):
            old_header_path = _parse_patch_file_header(line, "a/")
            index += 1
            continue
        if line.startswith("+++ "):
            new_header_path = _parse_patch_file_header(line, "b/")
            index += 1
            continue
        if not lines[index].startswith("@@ "):
            index += 1
            continue
        parsed = _parse_hunk_lines(lines, index)
        if parsed is None:
            # Keep visiting file headers; a later hunk may belong to another file.
            index += 1
            continue
        hunk_text, end, _, _, new_start, new_count = parsed
        this_hunk_order = hunk_order
        hunk_order += 1
        index = end
        if (old_header_path != expected_old_path or new_header_path != path):
            continue
        if new_count < 1:
            continue
        metadata = {
            "parent": parent_id,
            "commit": commit_id,
            "path": path,
            "previous_path": previous_path,
        }
        description = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n" + hunk_text
        item = Evidence(
            evidence_id=_evidence_id(
                repository_id, parent_id, commit_id, path, hunk_text.encode("utf-8"),
            ),
            repository_id=repository_id,
            kind=EvidenceType.GIT,
            location=SourceLocation(path, commit_id, max(1, new_start)),
            description=description,
            provenance="git_patch",
        )
        serialized_size = _serialized_evidence_size(item)
        if serialized_size <= MAX_REFERENCE_BYTES:
            candidate = (_hunk_selection_priority(hunk_text), this_hunk_order,
                         item, serialized_size)
            if selected is None or candidate[:2] < selected[:2]:
                selected = candidate
    if selected is None:
        return None
    return selected[2], selected[3]


def _hunk_selection_priority(hunk_text: str) -> int:
    """Prefer changed boundaries/calls, then other code, then imports/comments.

    This is a retrieval hint, not a claim that the selected change is correct or
    mandatory. Only changed lines count; stable surrounding guards do not.
    """
    removed: list[str] = []
    added: list[str] = []
    for line in hunk_text.splitlines():
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    if not added and not removed:
        return 2

    # Do not normalize code whitespace: indentation and spaces inside literals
    # can change behavior. Only empty/comment/import lines are obvious fallback
    # hints; incomplete hunks cannot establish cosmetic equivalence otherwise.
    changed_lines = [*removed, *added]
    if all(_is_import_or_comment_line(line) for line in changed_lines):
        return 2
    if any(not _is_import_or_comment_line(line)
           and _BOUNDARY_LINE.match(line.lstrip()) for line in changed_lines):
        return 0
    # A complete changed call site may point to a concrete transformation whose
    # whole implementation is supplied downstream. Preserve its chance against
    # a later branch; source order still breaks equal-priority ties. Exclude
    # declarations, comments and bare string contents from this lexical hint.
    if any(not _is_import_or_comment_line(line)
           and _CALL_SITE_LINE.match(line.lstrip()) for line in changed_lines):
        return 0
    return 1


_BOUNDARY_LINE = re.compile(
    r"^(?:if\b|elif\b|else\b|except\b|raise\b|assert\b|match\b|case\b|"
    r"catch\b|throw\b|}\s*(?:else\b|catch\b))",
)
_CALL_SITE_LINE = re.compile(
    r"^(?:(?:return|await)\s+|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*=\s*)?"
    r"(?:await\s+)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(",
)


_IMPORT_LINE = re.compile(
    r"^(?:import\b|from\s+\S+\s+import\b|using\b|#\s*include\b|"
    r"use\s+\S|require\s*\(|(?:const|let|var)\s+\S+\s*=\s*require\s*\()",
    re.IGNORECASE,
)
_COMMENT_LINE_PREFIXES = (
    "#", "//", "/*", "* ", "*/", "<!--", "-->", "-- ", ";",
)


def _is_import_or_comment_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _IMPORT_LINE.match(stripped):
        return True
    return stripped.startswith(_COMMENT_LINE_PREFIXES)


def _parse_patch_file_header(line: str, prefix: str) -> str | None:
    """Parse only unquoted, unambiguous `--- a/path` / `+++ b/path` headers."""
    value = line[4:]
    if value.endswith("\n"):
        value = value[:-1]
    if (not value.startswith(prefix) or value.startswith('"')
            or any(character in value for character in ('"', "\\", "\t", "\r", "\n"))):
        return None
    path = value[len(prefix):]
    try:
        _validate_repo_path(path)
    except ValueError:
        return None
    return path


def _parse_single_hunk(text: str) -> tuple[int, int, int, int] | None:
    parsed = _parse_hunk_lines(text.splitlines(keepends=True), 0)
    if parsed is None or parsed[1] != len(text.splitlines(keepends=True)):
        return None
    _, _, old_start, old_count, new_start, new_count = parsed
    return old_start, old_count, new_start, new_count


def _parse_hunk_lines(
    lines: list[str], start: int,
) -> tuple[str, int, int, int, int, int] | None:
    if start >= len(lines):
        return None
    match = _HUNK_HEADER.match(lines[start])
    if match is None:
        return None
    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count") or 1)
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count") or 1)
    if old_count == 0 and new_count == 0:
        return None
    old_seen = new_seen = 0
    end = start + 1
    while end < len(lines) and (old_seen < old_count or new_seen < new_count):
        line = lines[end]
        if line.startswith("\\ No newline at end of file"):
            end += 1
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            return None
        if line[0] in {" ", "-"}:
            old_seen += 1
        if line[0] in {" ", "+"}:
            new_seen += 1
        if old_seen > old_count or new_seen > new_count:
            return None
        end += 1
    if old_seen != old_count or new_seen != new_count:
        return None
    while end < len(lines) and lines[end].startswith("\\ No newline at end of file"):
        end += 1
    hunk_text = "".join(lines[start:end])
    return hunk_text, end, old_start, old_count, new_start, new_count


def _evidence_id(
    repository_id: str, parent_id: str, commit_id: str, path: str, hunk_bytes: bytes,
) -> str:
    digest = hashlib.sha256()
    for value in ("git_patch", repository_id, parent_id, commit_id, path):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(hunk_bytes)
    return digest.hexdigest()[:24]


def _serialized_evidence_size(item: Evidence) -> int:
    value = {
        "evidence_id": item.evidence_id,
        "repository_id": item.repository_id,
        "kind": item.kind.value,
        "location": {
            "path": item.location.path,
            "commit": item.location.commit,
            "line": item.location.line,
            "column": item.location.column,
        },
        "description": item.description,
        "provenance": item.provenance,
        "confidence": None,
    }
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def _validate_commit_id(commit_id: str) -> None:
    if not _COMMIT_ID.fullmatch(commit_id):
        raise ValueError("Git patch commit IDs must be full lowercase object IDs")


def _validate_repo_path(path: str) -> None:
    if (not path or "\\" in path or "\x00" in path or path.startswith("/")):
        raise ValueError("Git patch path must be a non-empty repository-relative POSIX path")
    if ":" in path.split("/", 1)[0]:
        raise ValueError("Git patch path cannot contain a drive or revision prefix")
    parts = PurePosixPath(path).parts
    if (not parts or PurePosixPath(path).as_posix() != path
            or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError("Git patch path is not canonical")


def _run_git_limited(root: Path, args: tuple[str, ...], output_limit: int, *,
                     inherit_system_config: bool = False) -> _GitRead:
    command = [
        "git", "--literal-pathspecs", "-C", str(root),
        "-c", f"safe.directory={root}", "-c", "diff.external=",
        "-c", "core.quotePath=false", *args,
    ]
    environment = {
        **os.environ,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
    }
    if not inherit_system_config:
        environment['GIT_CONFIG_NOSYSTEM'] = '1'
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            shell=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git executable was not found on PATH") from error
    assert process.stdout is not None
    chunks: list[bytes] = []
    total = 0
    read_errors: list[OSError] = []

    def consume_stdout() -> None:
        nonlocal total
        try:
            while total <= output_limit:
                chunk = process.stdout.read(min(8192, output_limit + 1 - total))
                if not chunk:
                    return
                chunks.append(chunk)
                total += len(chunk)
        except OSError as error:
            read_errors.append(error)

    reader = threading.Thread(target=consume_stdout, daemon=True)
    reader.start()
    reader.join(GIT_TIMEOUT_SECONDS)
    timed_out = reader.is_alive()
    truncated = total > output_limit
    if timed_out or truncated:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    reader.join(1)
    if process.stdout:
        process.stdout.close()
    if timed_out:
        raise RuntimeError(f"Git command exceeded {GIT_TIMEOUT_SECONDS:g} seconds")
    if read_errors:
        raise RuntimeError("Could not read bounded Git output") from read_errors[0]
    return _GitRead(b"".join(chunks)[:output_limit], process.returncode or 0, truncated)
