"""Conservative link and Windows reparse-point protection."""

import stat
from pathlib import Path


def is_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & flag
    )


def no_symlink(path: Path, boundary: Path) -> None:
    current = path
    while current != boundary:
        if is_link(current):
            raise ValueError(f"link or reparse point is not allowed: {current}")
        current = current.parent
    if is_link(boundary):
        raise ValueError("repository root cannot be a link or reparse point")
