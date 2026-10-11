"""Parsers for Git's NUL-delimited history and diff output."""

from dataclasses import dataclass

from skillforge.domain.change import FileChangeKind


@dataclass(frozen=True, slots=True)
class ParsedCommit:
    commit_id: str
    parent_ids: tuple[str, ...]
    committed_at: str
    message: str


@dataclass(frozen=True, slots=True)
class ParsedFileChange:
    path: str
    kind: FileChangeKind
    previous_path: str | None = None


def parse_log_output(output: bytes) -> tuple[ParsedCommit, ...]:
    if not output:
        return ()
    fields = output.split(b"\0")
    if len(fields) % 4 != 1:
        raise ValueError("Malformed Git log output")
    commits = []
    for index in range(0, len(fields) - 1, 4):
        commit_id = fields[index].lstrip(b"\n").decode("ascii")
        parents = fields[index + 1].decode("ascii").split()
        committed_at = fields[index + 2].decode("ascii")
        message = fields[index + 3].decode("utf-8", errors="replace")
        commits.append(ParsedCommit(commit_id, tuple(parents), committed_at, message))
    return tuple(commits)


def parse_name_status(output: bytes) -> tuple[ParsedFileChange, ...]:
    fields = output.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    changes = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        code = status[:1]
        if code in {b"R", b"C"}:
            if index + 1 >= len(fields):
                raise ValueError("Malformed Git rename output")
            previous_path = _decode_path(fields[index])
            path = _decode_path(fields[index + 1])
            index += 2
            kind = FileChangeKind.RENAMED if code == b"R" else FileChangeKind.ADDED
            changes.append(ParsedFileChange(path, kind, previous_path))
            continue
        if index >= len(fields):
            raise ValueError("Malformed Git name-status output")
        path = _decode_path(fields[index])
        index += 1
        if code == b"A":
            kind = FileChangeKind.ADDED
        elif code == b"D":
            kind = FileChangeKind.DELETED
        else:
            kind = FileChangeKind.MODIFIED
        changes.append(ParsedFileChange(path, kind))
    return tuple(changes)


def parse_numstat(output: bytes) -> dict[tuple[str, str | None], tuple[int | None, int | None]]:
    fields = output.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    counts = {}
    index = 0
    while index < len(fields):
        parts = fields[index].split(b"\t", 2)
        index += 1
        if len(parts) != 3:
            raise ValueError("Malformed Git numstat output")
        additions_raw, deletions_raw, path_raw = parts
        previous_path = None
        if path_raw:
            path = _decode_path(path_raw)
        else:
            if index + 1 >= len(fields):
                raise ValueError("Malformed Git rename numstat output")
            previous_path = _decode_path(fields[index])
            path = _decode_path(fields[index + 1])
            index += 2
        if additions_raw == b"-" or deletions_raw == b"-":
            additions, deletions = None, None
        else:
            additions = int(additions_raw)
            deletions = int(deletions_raw)
        counts[(path, previous_path)] = (additions, deletions)
    return counts


def _decode_path(path: bytes) -> str:
    return path.decode("utf-8", errors="replace")
