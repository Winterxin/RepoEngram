"""Read-only extraction of deterministic facts from a local Git history."""

import hashlib
import logging
import os
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind, GitHistoryResult
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.domain.evolution import GitHistoryDelta
from skillforge.errors import GitHistoryError, InvalidRepositoryPath
from skillforge.repository.git_history_parser import (
    ParsedCommit,
    ParsedFileChange,
    parse_log_output,
    parse_name_status,
    parse_numstat,
)
from skillforge.repository.identity import repository_identity

LOGGER = logging.getLogger(__name__)


class GitHistoryMiner:
    """Mine commit and file-change observations without changing the target repository."""

    def __init__(self, timeout: float = 10) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def mine(self, repo_path: Path | str, limit: int = 20) -> GitHistoryResult:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")

        top_result = self._run_git(root, "rev-parse", "--show-toplevel", check=False)
        if top_result.returncode != 0:
            detail = self._stderr(top_result) or "directory is not a Git worktree"
            raise GitHistoryError(f"Git repository could not be opened at {root}: {detail}")
        git_root = Path(self._stdout_text(top_result).strip()).resolve()
        if os.path.normcase(str(git_root)) != os.path.normcase(str(root)):
            raise GitHistoryError(f"Pass the Git repository root, not a subdirectory: {root}")

        repository_id = repository_identity(root)
        head = self._run_git(root, "rev-parse", "--verify", "HEAD", check=False)
        if head.returncode != 0:
            symbolic_head = self._run_git(
                root, "symbolic-ref", "--quiet", "HEAD", check=False
            )
            if symbolic_head.returncode == 0:
                return GitHistoryResult(repository_id, (), ())
            detail = self._stderr(head) or "HEAD could not be resolved"
            raise GitHistoryError(f"Cannot read Git history at {root}: {detail}")
        if limit == 0:
            return GitHistoryResult(repository_id, (), ())

        log_result = self._run_git(
            root,
            "log",
            f"--max-count={limit}",
            "--encoding=UTF-8",
            "--format=%H%x00%P%x00%cI%x00%B%x00",
            "HEAD",
        )
        try:
            commits = parse_log_output(log_result.stdout)
        except (UnicodeDecodeError, ValueError) as error:
            raise GitHistoryError(f"Cannot parse Git commit history at {root}: {error}") from error

        return self._mine_commits(root, repository_id, commits)

    def mine_delta(
        self, repo_path: Path | str, base_commit: str, head_commit: str = "HEAD"
    ) -> GitHistoryDelta:
        """Read only commits in (base, head], rejecting rewritten ancestry."""
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")
        top = self._run_git(root, "rev-parse", "--show-toplevel", check=False)
        if (top.returncode != 0 or os.path.normcase(str(
                Path(self._stdout_text(top).strip()).resolve()
        )) != os.path.normcase(str(root))):
            raise GitHistoryError("Pass the Git repository root")
        resolved = []
        for ref in (base_commit, head_commit):
            if not isinstance(ref, str) or not ref or ref.startswith("-"):
                raise GitHistoryError("Git delta commit reference is invalid")
            result = self._run_git(root, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
            if result.returncode != 0:
                raise GitHistoryError(f"Git delta commit does not exist: {ref}")
            resolved.append(self._stdout_text(result).strip())
        base, head = resolved
        ancestor = self._run_git(root, "merge-base", "--is-ancestor", base, head, check=False)
        if ancestor.returncode != 0:
            raise GitHistoryError("Git delta base is not an ancestor of head")
        repository_id = repository_identity(root)
        if base == head:
            history = GitHistoryResult(repository_id, (), ())
        else:
            log = self._run_git(
                root, "log", "--encoding=UTF-8", "--format=%H%x00%P%x00%cI%x00%B%x00",
                f"{base}..{head}",
            )
            try:
                commits = parse_log_output(log.stdout)
            except (UnicodeDecodeError, ValueError) as error:
                raise GitHistoryError(f"Cannot parse Git delta: {error}") from error
            history = self._mine_commits(root, repository_id, commits)
        return GitHistoryDelta(repository_id, base, head, history.changesets, history.evidence)

    def commit_window_order(
        self, repo_path: Path | str, repository_id: str, head_commit: str, limit: int
    ) -> tuple[str, ...]:
        """Read only commit IDs in the current HEAD's bounded Git log order."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")
        top = self._run_git(root, "rev-parse", "--show-toplevel", check=False)
        if (top.returncode != 0 or os.path.normcase(str(
                Path(self._stdout_text(top).strip()).resolve()
        )) != os.path.normcase(str(root))):
            raise GitHistoryError("Pass the Git repository root")
        if repository_identity(root) != repository_id:
            raise GitHistoryError("Git repository identity changed")
        current = self._run_git(root, "rev-parse", "--verify", "HEAD^{commit}", check=False)
        if current.returncode != 0 or self._stdout_text(current).strip() != head_commit:
            raise GitHistoryError("Git HEAD does not match delta head")
        result = self._run_git(
            root, "log", f"--max-count={limit}", "--format=%H", "HEAD"
        )
        commits = tuple(self._stdout_text(result).splitlines())
        if len(commits) != len(set(commits)) or not commits or commits[0] != head_commit:
            raise GitHistoryError("Cannot read canonical bounded Git order")
        return commits

    def _mine_commits(
        self, root: Path, repository_id: str, commits: tuple[ParsedCommit, ...]
    ) -> GitHistoryResult:
        changesets = []
        all_evidence: dict[str, Evidence] = {}
        for commit in commits:
            boundary = False
            if not commit.parent_ids:
                # Git's shallow traversal reports the boundary as parentless even
                # though its commit object retains the real, unavailable parents.
                raw = self._run_git(root, "cat-file", "-p", commit.commit_id)
                header = self._stdout_text(raw).split("\n\n", 1)[0]
                parents = tuple(line.removeprefix("parent ") for line in header.splitlines()
                                if line.startswith("parent "))
                if parents:
                    commit = replace(commit, parent_ids=parents)
                    boundary = True
            commit_evidence = self._make_evidence(
                repository_id,
                commit.commit_id,
                ".",
                "git_log",
                f"Observed Git commit {commit.commit_id}" + (
                    "; history boundary: parent diff unavailable; no file changes inferred"
                    if boundary else ""
                ),
            )
            all_evidence[commit_evidence.evidence_id] = commit_evidence
            try:
                parsed_files, numstat = (
                    ((), {}) if boundary else self._read_file_changes(root, commit)
                )
            except (UnicodeDecodeError, ValueError) as error:
                raise GitHistoryError(
                    f"Cannot parse Git diff for commit {commit.commit_id}: {error}"
                ) from error

            package_paths = sorted({
                str(PurePosixPath(parsed.path).parent / "__init__.py")
                for parsed in parsed_files
                if parsed.kind == FileChangeKind.ADDED and parsed.path.endswith(".py")
                and PurePosixPath(parsed.path).name != "__init__.py"
                and PurePosixPath(parsed.path).parent != PurePosixPath(".")
            })
            existing_packages = set()
            if commit.parent_ids and package_paths:
                tree = self._run_git(
                    root, "ls-tree", "-r", "--name-only", "-z",
                    commit.parent_ids[0], "--", *package_paths,
                )
                existing_packages = set(self._stdout_text(tree).split("\0")) & set(package_paths)
            file_changes = []
            evidence_ids = [commit_evidence.evidence_id]
            for parsed in sorted(
                parsed_files,
                key=lambda item: (item.path, item.kind.value, item.previous_path or ""),
            ):
                additions, deletions = numstat[(parsed.path, parsed.previous_path)]
                if parsed.previous_path is None:
                    description = f"Git reports {parsed.kind.value} file {parsed.path!r}"
                else:
                    description = (
                        f"Git reports {parsed.kind.value} from {parsed.previous_path!r} "
                        f"to {parsed.path!r}"
                    )
                diff_evidence = self._make_evidence(
                    repository_id,
                    commit.commit_id,
                    parsed.path,
                    "git_diff",
                    description,
                    parsed.previous_path,
                )
                numstat_description = (
                    f"Git line counts unavailable for {parsed.path!r} "
                    "(binary or non-text change)"
                    if additions is None or deletions is None
                    else f"Git reports {additions} additions and {deletions} deletions "
                    f"for {parsed.path!r}"
                )
                numstat_evidence = self._make_evidence(
                    repository_id,
                    commit.commit_id,
                    parsed.path,
                    "git_numstat",
                    numstat_description,
                    parsed.previous_path,
                )
                package_ids = ()
                package_path = str(PurePosixPath(parsed.path).parent / "__init__.py")
                if (parsed.kind == FileChangeKind.ADDED and parsed.path.endswith(".py")
                        and PurePosixPath(parsed.path).name != "__init__.py"
                        and package_path in existing_packages):
                    package_evidence = self._make_evidence(
                        repository_id, commit.commit_id, package_path, "git_parent_package",
                        f"Git parent {commit.parent_ids[0]} contains package initializer "
                        f"{package_path!r} before this change",
                    )
                    all_evidence[package_evidence.evidence_id] = package_evidence
                    package_ids = (package_evidence.evidence_id,)
                    if package_evidence.evidence_id not in evidence_ids:
                        evidence_ids.append(package_evidence.evidence_id)
                file_changes.append(
                    FileChange(
                        parsed.path,
                        parsed.kind,
                        parsed.previous_path,
                        additions,
                        deletions,
                        (diff_evidence.evidence_id, numstat_evidence.evidence_id),
                        package_ids,
                    )
                )
                for evidence in (diff_evidence, numstat_evidence):
                    all_evidence[evidence.evidence_id] = evidence
                    evidence_ids.append(evidence.evidence_id)

            changesets.append(
                ChangeSet(
                    repository_id=repository_id,
                    commit_id=commit.commit_id,
                    committed_at=commit.committed_at,
                    files=tuple(file_changes),
                    evidence_ids=tuple(evidence_ids),
                    parent_ids=commit.parent_ids,
                    message=commit.message,
                    is_merge=len(commit.parent_ids) > 1,
                )
            )

        return GitHistoryResult(
            repository_id,
            tuple(changesets),
            tuple(sorted(all_evidence.values(), key=lambda item: item.evidence_id)),
        )

    def _read_file_changes(
        self, root: Path, commit: ParsedCommit
    ) -> tuple[
        tuple[ParsedFileChange, ...],
        dict[tuple[str, str | None], tuple[int | None, int | None]],
    ]:
        if commit.parent_ids:
            base_args = (commit.parent_ids[0], commit.commit_id)
            name_result = self._run_git(
                root,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--find-renames",
                "--name-status",
                "-z",
                *base_args,
            )
            numstat_result = self._run_git(
                root,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--find-renames",
                "--numstat",
                "-z",
                *base_args,
            )
        else:
            base_args = (
                "--root",
                "--no-commit-id",
                "--no-ext-diff",
                "--no-textconv",
                "--find-renames",
                "-r",
            )
            name_result = self._run_git(
                root, "diff-tree", *base_args, "--name-status", "-z", commit.commit_id
            )
            numstat_result = self._run_git(
                root, "diff-tree", *base_args, "--numstat", "-z", commit.commit_id
            )
        parsed_files = parse_name_status(name_result.stdout)
        numstat = parse_numstat(numstat_result.stdout)
        change_keys = {(item.path, item.previous_path) for item in parsed_files}
        numstat_keys = set(numstat)
        missing = change_keys - numstat_keys
        unexpected = numstat_keys - change_keys
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    "missing numstat entries "
                    f"{sorted(missing, key=lambda item: (item[0], item[1] or ''))!r}"
                )
            if unexpected:
                details.append(
                    "unexpected numstat entries "
                    f"{sorted(unexpected, key=lambda item: (item[0], item[1] or ''))!r}"
                )
            raise ValueError("Git name-status/numstat file mismatch: " + "; ".join(details))
        return parsed_files, numstat

    def _run_git(
        self, root: Path, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        command = [
            "git",
            "-C",
            str(root),
            "-c",
            f"safe.directory={root}",
            *args,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=self.timeout,
                check=False,
                env={
                    **os.environ,
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_PAGER": "cat",
                },
            )
        except FileNotFoundError as error:
            raise GitHistoryError("Git executable was not found on PATH") from error
        except subprocess.TimeoutExpired as error:
            raise GitHistoryError(
                f"Git command timed out after {self.timeout:g} seconds"
            ) from error
        except OSError as error:
            LOGGER.warning("Git history command failed: %s", error)
            raise GitHistoryError(f"Could not run Git command: {error}") from error
        if check and result.returncode != 0:
            detail = self._stderr(result) or f"exit status {result.returncode}"
            raise GitHistoryError(f"Git command failed: {detail}")
        return result

    @staticmethod
    def _stdout_text(result: subprocess.CompletedProcess[bytes]) -> str:
        return result.stdout.decode("utf-8", errors="replace")

    @staticmethod
    def _stderr(result: subprocess.CompletedProcess[bytes]) -> str:
        return result.stderr.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _make_evidence(
        repository_id: str,
        commit_id: str,
        path: str,
        provenance: str,
        description: str,
        previous_path: str | None = None,
    ) -> Evidence:
        identity = "\0".join((repository_id, commit_id, provenance, path, previous_path or ""))
        evidence_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return Evidence(
            evidence_id=evidence_id,
            repository_id=repository_id,
            kind=EvidenceType.GIT,
            location=SourceLocation(path, commit_id),
            description=description,
            provenance=provenance,
        )
