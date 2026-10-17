"""Request-local reuse must never conceal source edits."""

from pathlib import Path

from skillforge.integration.source_reads import SourceReadSession


def test_session_reuses_ast_but_invalidates_changed_file(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_bytes(b"value = 1\n")
    original = Path.read_bytes
    observed = []

    def read(path):
        observed.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    session = SourceReadSession(tmp_path)
    first = session.read_python(source)
    assert session.read_python(source)[1] is first[1]
    assert observed == [source]
    source.write_bytes(b"value = 200\n")
    changed = session.read_python(source)
    assert changed[0] == b"value = 200\n"
    assert changed[1] is not first[1]
    assert observed == [source, source]
