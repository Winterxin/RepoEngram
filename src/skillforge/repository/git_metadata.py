"""Read-only Git identity probe for a worktree root."""

import logging
import os
import subprocess
from pathlib import Path

from skillforge.domain.repository import GitMetadata

LOGGER = logging.getLogger(__name__)


def read_git_metadata(root: Path) -> GitMetadata:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            LOGGER.debug("Git metadata unavailable: %s", error)
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != root:
        return GitMetadata(False)
    return GitMetadata(
        True,
        git("symbolic-ref", "--quiet", "--short", "HEAD"),
        git("rev-parse", "--verify", "HEAD"),
    )
