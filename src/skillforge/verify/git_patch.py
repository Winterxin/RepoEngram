"""Read-only Git working-tree patch collection against an explicit base commit."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

from skillforge.domain.verification import (
    GitPatchSnapshot,
    PatchChangeKind,
    VerifyPatchFile,
    patch_fingerprint,
    validate_patch_exclusions,
)
from skillforge.filesystem_safety import no_symlink
from skillforge.repository.identity import repository_identity

_INTERNAL_ARTIFACT_PREFIXES = (
    ".skillforge/productization/prepare/",
    ".skillforge/productization/verify/",
    ".skillforge/productization/feedback/",
    ".skillforge/productization/evolution/",
)


class GitPatchCollector:
    """Observe final tracked state plus non-ignored untracked files."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        if not 1 <= timeout_seconds <= 3600:
            raise ValueError("invalid Git timeout")
        self.timeout_seconds = timeout_seconds

    def _git(self, root: Path, *args: str) -> bytes:
        try:
            result = subprocess.run(
                ("git", "-c", f"safe.directory={root}", *args), cwd=root,
                capture_output=True, timeout=self.timeout_seconds, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Git command failed: {args[0]}") from exc
        if result.returncode:
            raise ValueError(f"Git {args[0]} failed: " +
                             result.stderr.decode("utf-8", "replace").strip())
        return result.stdout

    def snapshot(self, repo_root: Path | str, repository_id: str,
                 base_commit: str, *,
                 excluded_paths: tuple[str, ...] = ()) -> GitPatchSnapshot:
        if (not isinstance(base_commit, str) or not base_commit.strip()
                or base_commit.startswith("-")):
            raise ValueError("base commit must be an explicit ref")
        validate_patch_exclusions(excluded_paths)
        supplied = Path(repo_root).expanduser().absolute()
        if not supplied.is_dir():
            raise ValueError("repo_root must be a Git worktree root")
        no_symlink(supplied, supplied)
        root = supplied.resolve()
        if repository_identity(root) != repository_id:
            raise ValueError("repository_id mismatch")
        for relative_path in excluded_paths:
            no_symlink(root.joinpath(*relative_path.split("/")), root)
        pathspecs = (
            ".",
            *(f":(glob,exclude){prefix}**" for prefix in _INTERNAL_ARTIFACT_PREFIXES),
            *(f":(literal,exclude){path}" for path in excluded_paths),
        )
        top = Path(os.fsdecode(self._git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
        if os.path.normcase(str(top)) != os.path.normcase(str(root)):
            raise ValueError("repo_root is not the Git worktree root")
        base = self._git(root, "rev-parse", "--verify", "--end-of-options",
                         f"{base_commit}^{{commit}}").decode("ascii").strip()
        head = self._git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        raw = self._git(root, "diff", "--name-status", "-z", "-M", "-C",
                        "--find-copies-harder", base, "--", *pathspecs)
        parts = raw.split(b"\0")
        if parts[-1] != b"":
            raise ValueError("malformed Git name-status output")
        files: list[VerifyPatchFile] = []
        position = 0
        while position < len(parts) - 1:
            status = parts[position].decode("ascii")
            position += 1
            code = status[:1]
            kinds = {"A": PatchChangeKind.ADDED, "M": PatchChangeKind.MODIFIED,
                     "D": PatchChangeKind.DELETED, "R": PatchChangeKind.RENAMED,
                     "C": PatchChangeKind.COPIED, "T": PatchChangeKind.TYPE_CHANGED}
            if code not in kinds or (code in "RC" and not status[1:].isdigit()) or (
                code not in "RC" and len(status) != 1
            ):
                raise ValueError(f"unsupported Git name-status: {status}")
            if code in "RC":
                if position + 1 >= len(parts) - 1:
                    raise ValueError("incomplete Git rename/copy")
                previous = os.fsdecode(parts[position])
                path = os.fsdecode(parts[position + 1])
                position += 2
            else:
                if position >= len(parts) - 1:
                    raise ValueError("incomplete Git name-status")
                previous = None
                path = os.fsdecode(parts[position])
                position += 1
            files.append(VerifyPatchFile(path, kinds[code], previous))
        tracked_diff = self._git(root, "diff", "--binary", "--full-index",
                                 "--no-ext-diff", base, "--", *pathspecs)
        untracked = self._git(root, "ls-files", "--others", "--exclude-standard", "-z",
                              "--", *pathspecs)
        if not untracked.endswith(b"\0") and untracked:
            raise ValueError("malformed Git untracked output")
        for raw_path in filter(None, untracked.split(b"\0")):
            path = os.fsdecode(raw_path)
            if path.startswith(_INTERNAL_ARTIFACT_PREFIXES) or path in excluded_paths:
                continue
            full_path = root / path
            no_symlink(full_path.parent, root)
            metadata = full_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                digest_value = hashlib.sha256(os.fsencode(os.readlink(full_path))).hexdigest()
            elif stat.S_ISREG(metadata.st_mode):
                with full_path.open("rb") as stream:
                    digest_value = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                raise ValueError(f"unsupported untracked file type: {path}")
            files.append(VerifyPatchFile(path, PatchChangeKind.UNTRACKED,
                                         content_sha256=digest_value))
        records = tuple(sorted(files, key=lambda item: (item.path, item.kind.value)))
        digest = hashlib.sha256(tracked_diff).hexdigest()
        provisional = object.__new__(GitPatchSnapshot)
        for key, value in (("repository_id", repository_id), ("resolved_base_commit", base),
                           ("head_commit", head), ("tracked_diff_sha256", digest),
                           ("files", records)):
            object.__setattr__(provisional, key, value)
        return GitPatchSnapshot(repository_id, base_commit, base, head, records, digest,
                                patch_fingerprint(provisional))
