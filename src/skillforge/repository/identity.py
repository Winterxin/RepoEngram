"""The repository identity policy shared by inventory, history, and Verify."""

import hashlib
from pathlib import Path


def repository_identity(root: Path) -> str:
    return hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:20]
