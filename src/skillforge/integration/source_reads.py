"""Request-local source reads shared by deterministic context renderers."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


class SourceReadSession:
    """Reuse bytes/ASTs within one export, never across Agent calls or edits."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._sources = {}
        self._tracked = None

    def read_python(self, path: Path) -> tuple[bytes, ast.Module]:
        stat = path.stat()
        identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        prior = self._sources.get(path)
        if prior is not None and prior[0] == identity:
            return prior[1], prior[2]
        content = path.read_bytes()
        tree = ast.parse(content)
        self._sources[path] = (identity, content, tree)
        return content, tree

    def tracked_python_files(self) -> tuple[str, ...]:
        if self._tracked is None:
            self._tracked = tuple(subprocess.run(
                ["git", "ls-files", "-z", "--", "*.py"], cwd=self.root,
                capture_output=True, check=True, timeout=10,
            ).stdout.decode("utf8").split("\0"))
        return self._tracked
