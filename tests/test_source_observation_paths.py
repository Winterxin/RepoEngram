"""Error paths keep their context while competing for the same bounded package."""

from test_source_constraints import _export, _files

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_observation_selection import (
    COLD_PACKAGE_BYTES,
    MAX_PACKAGE_BYTES,
    package_sizes,
)


def test_nested_loop_try_handler_else_and_finally_stay_explicit(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text(
        "def process(items, connection):\n"
        "    for item in items:\n"
        "        with connection:\n"
        "            try:\n"
        "                result = run(item)\n"
        "            except ValueError as error:\n"
        "                result = recover(error)\n"
        "            else:\n"
        "                publish(result)\n"
        "            finally:\n"
        "                cleanup(item)\n",
        encoding="utf-8",
    )
    plan = _export(tmp_path)
    text = _files(plan)["SKILL.md"]
    assert "for item in items (iteration unknown)" in text
    assert "with connection (entry/exit unknown)" in text
    assert "except ValueError as error (selection unknown)" in text
    assert "try else (requires normal try completion)" in text
    assert "finally (unwinding context, execution unknown)" in text
    assert "result = recover(error)" in text
    assert "publish(result)" in text and "cleanup(item)" in text
    assert all(size <= COLD_PACKAGE_BYTES for size in package_sizes(plan))


def test_later_file_error_path_is_not_starved_by_early_file_rows(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text(
        "def normalize(value):\n" + "".join(
            f"    if value == {i}:\n        return {i}\n" for i in range(120)
        ), encoding="utf-8",
    )
    (src / "z.py").write_text(
        "def execute(value):\n    try:\n        return run(value)\n"
        "    except LookupError:\n        return fallback(value)\n", encoding="utf-8",
    )
    plan = _export(tmp_path, scope=ProjectScope(ProjectScopeKind.DIRECTORY, ("src",)))
    text = _files(plan)["SKILL.md"]
    assert "z.py" in text and "except LookupError" in text
    assert "a.py" in text
    assert len([line for line in text.splitlines() if line.startswith("- ")]) <= 48
    assert all(size <= MAX_PACKAGE_BYTES for size in package_sizes(plan))


def test_distinct_handlers_and_finally_keep_distinct_source_contexts(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text(
        "def execute():\n    try:\n        return run()\n"
        "    except ValueError:\n        return recover_value()\n"
        "    except TypeError:\n        return recover_type()\n"
        "    finally:\n        close()\n", encoding="utf-8",
    )
    text = _files(_export(tmp_path))["SKILL.md"]
    value = next(line for line in text.splitlines() if "return recover_value()" in line)
    typed = next(line for line in text.splitlines() if "return recover_type()" in line)
    assert "except ValueError" in value and "except TypeError" not in value
    assert "except TypeError" in typed and "except ValueError" not in typed
    cleanup = next(line for line in text.splitlines() if "close()" in line)
    assert "finally" in cleanup and "except" not in cleanup


def test_unsupported_match_and_while_remain_omitted(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text(
        "def execute(value):\n    while value:\n        hidden_while()\n"
        "    match value:\n        case 1:\n            hidden_match()\n"
        "    return value\n", encoding="utf-8",
    )
    text = _files(_export(tmp_path))["SKILL.md"]
    assert "hidden_while" not in text and "hidden_match" not in text
    assert "return value" in text


def test_many_handlers_cannot_starve_another_files_module_iteration(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("".join(
        f"def execute_{i}():\n    try:\n        return run()\n"
        "    except RuntimeError:\n        return recover()\n" for i in range(70)
    ), encoding="utf-8")
    (src / "z.py").write_text(
        "def execute(modules):\n    for module in modules:\n        run(module)\n",
        encoding="utf-8",
    )
    plan = _export(tmp_path, scope=ProjectScope(ProjectScopeKind.DIRECTORY, ("src",)))
    text = _files(plan)["SKILL.md"]
    assert "except RuntimeError" in text
    assert "for module in modules (iteration unknown)" in text
    assert "run(module)" in text
    assert all(size <= COLD_PACKAGE_BYTES for size in package_sizes(plan))
