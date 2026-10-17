"""Function-unit span selection stays distributed and within caller budgets."""

import ast
import json

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_rule_distillation import _chunks
from skillforge.integration.source_span_selection import select_distributed_spans
from skillforge.integration.source_structure_priority import explicit_constraint_benefits


def _files(prefixes, unit_count=8):
    return [
        (f"src/{prefix}.py", f"hash-{prefix}",
         ([
             (0, [(f"symbol_{index}", f"value_{index} = '值{index}'\n")])
             for index in range(unit_count)
          ],))
        for prefix in prefixes
    ]


def test_selection_spreads_across_files_and_function_positions():
    selected, skipped = select_distributed_spans(
        _files(("alpha", "beta"), 12), max_spans=12, raw_byte_budget=4096,
        serialized_fits=lambda rows: True,
    )
    by_file = {}
    for path, _, symbol, _ in selected:
        by_file.setdefault(path, set()).add(symbol)
    assert len(selected) == 12
    assert skipped
    assert set(by_file) == {"src/alpha.py", "src/beta.py"}
    for symbols in by_file.values():
        assert {"symbol_0", "symbol_6", "symbol_11"} <= symbols


def test_large_function_unit_is_skipped_whole_when_span_limit_is_too_small():
    units = [
        (0, [("large", "a" * 1800), ("large", "b" * 1800), ("large", "c" * 1800)]),
        (0, [("middle", "return value\n")]),
        (0, [("last", "return fallback\n")]),
    ]
    selected, skipped = select_distributed_spans(
        [("src/module.py", "sha", (units,))], max_spans=2, raw_byte_budget=4096,
        serialized_fits=lambda rows: True,
    )
    assert all(symbol != "large" for _, _, symbol, _ in selected)
    assert {symbol for _, _, symbol, _ in selected} == {"middle", "last"}
    assert ("src/module.py", "large", "unit_exceeds_span_limit") in skipped


def test_utf8_raw_and_escaped_serialization_budgets_are_both_enforced():
    units = [(0, [("first", "中文😀\\\"" * 40)]), (0, [("last", "x" * 100)])]

    def within_wire_budget(rows):
        raw = sum(len(text.encode("utf-8")) for _, _, _, text in rows)
        serialized = len(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
        escaped = len(json.dumps(rows, ensure_ascii=True).encode("utf-8"))
        return raw <= 120 and serialized <= 500 and escaped <= 500

    selected, skipped = select_distributed_spans(
        [("src/flags.py", "sha", (units,))], max_spans=2, raw_byte_budget=120,
        serialized_fits=within_wire_budget,
    )
    assert sum(len(text.encode("utf-8")) for _, _, _, text in selected) <= 120
    assert len(json.dumps(selected, ensure_ascii=True).encode("utf-8")) <= 500
    assert skipped


def test_cost_aware_selection_prefers_compact_boundary_evidence_to_long_implementation():
    units = [
        (8, [("large_transform", "x" * 9000)]),
        (3, [("compact_boundary", "if value is None: raise ValueError()\n")]),
        (0, [("ordinary", "return value\n")]),
    ]
    selected, _ = select_distributed_spans(
        [("src/flow.py", "sha", (units,))], max_spans=2, raw_byte_budget=9000,
        serialized_fits=lambda rows: len(rows) <= 2,
    )
    symbols = {symbol for _, _, symbol, _ in selected}
    assert "compact_boundary" in symbols
    assert "large_transform" not in symbols


def test_ast_signal_is_name_independent_and_ignores_loop_and_boolean_complexity():
    source = (
        "def first_name(values):\n"
        "    for item in values:\n"
        "        consume(item and valid(item))\n"
        "    return None\n"
        "def renamed(value):\n"
        "    if value is None:\n"
        "        raise ValueError()\n"
        "    return value\n"
        "def invoke(value):\n"
        "    return renamed(value)\n"
    )
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    units = [(node.name, node, ast.get_source_segment(source, node)) for node in functions]
    benefits = explicit_constraint_benefits(units)
    assert benefits["renamed"] > benefits["first_name"]
    assert benefits["invoke"] > benefits["first_name"]

    renamed_source = source.replace("renamed", "neutral_name")
    renamed_module = ast.parse(renamed_source)
    renamed_units = [
        (node.name, node, ast.get_source_segment(renamed_source, node))
        for node in renamed_module.body if isinstance(node, ast.FunctionDef)
    ]
    assert explicit_constraint_benefits(renamed_units)["neutral_name"] == benefits["renamed"]


def test_compact_constraint_beats_large_template_using_complete_unit_cost(tmp_path):
    source = (
        "def render(value):\n"
        "    template = '" + "x" * 10_000 + "'\n"
        "    if value:\n"
        "        return template\n"
        "    return template\n"
        "def validate(value):\n"
        "    if value is None:\n"
        "        raise ValueError()\n"
        "    return value\n"
    )
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    ranked = explicit_constraint_benefits([
        (node.name, node, ast.get_source_segment(source, node)) for node in functions
    ])
    source_hash = "sha"
    units = []
    for node in functions:
        text = ast.get_source_segment(source, node)
        encoded = text.encode("utf-8")
        spans = [
            (node.name, encoded[offset:offset + 1800].decode("utf-8"))
            for offset in range(0, len(encoded), 1800)
        ]
        units.append((ranked[node.name], spans))

    selected, _ = select_distributed_spans(
        [("src/view.py", source_hash, (units,))],
        max_spans=6,
        raw_byte_budget=sum(len(text.encode("utf-8")) for _, spans in units for _, text in spans),
        serialized_fits=lambda rows: len(rows) <= 6,
    )
    symbols = {symbol for _, _, symbol, _ in selected}
    assert "validate" in symbols
    assert "render" not in symbols


def test_source_chunking_covers_late_definitions_and_keeps_long_function_complete(tmp_path):
    first = tmp_path / "src" / "first.py"
    second = tmp_path / "src" / "second.py"
    first.parent.mkdir()
    definitions = [f"def transform_{i:02d}(value):\n    return value + {i}\n"
                   for i in range(36)]
    first.write_text("\n".join(definitions), encoding="utf-8")
    second.write_text("\n".join(definitions), encoding="utf-8")

    rows, hashes = _chunks(
        tmp_path,
        ProjectScope(ProjectScopeKind.FILE, ("src/first.py", "src/second.py")),
    )
    assert len(rows) <= 64
    assert set(hashes) == {"src/first.py", "src/second.py"}
    for relative in hashes:
        symbols = {row["symbol"] for row in rows if row["path"] == relative}
        assert {"transform_00", "transform_18", "transform_35"} <= symbols

    long_source = tmp_path / "src" / "long.py"
    long_source.write_text(
        "def before(value):\n    return value\n\n"
        "def complete_unit(value):\n" + "    value = value + 1\n" * 350
        + "    return value\n\n"
        "def after(value):\n    return value\n",
        encoding="utf-8",
    )
    long_rows, _ = _chunks(
        tmp_path, ProjectScope(ProjectScopeKind.FILE, ("src/long.py",)),
    )
    body_parts = [row for row in long_rows if row["symbol"] == "complete_unit"]
    assert len(body_parts) > 1
    source_text = long_source.read_bytes().decode("utf-8")
    module = ast.parse(source_text)
    expected_segment = ast.get_source_segment(
        source_text,
        next(node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "complete_unit"),
    )
    assert "".join(row["text"] for row in body_parts) == expected_segment


def test_ast_guard_density_and_direct_local_callee_get_priority_without_names(tmp_path):
    source = tmp_path / "src" / "flow.py"
    source.parent.mkdir()
    simple = [f"def convert_{index:02d}(value):\n    return value + {index}\n"
              for index in range(40)]
    source.write_text(
        "\n".join(simple)
        + "\ndef apply(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    return inspect_value(value)\n"
        "\ndef inspect_value(value):\n"
        "    try:\n"
        "        assert value\n"
        "    except ValueError:\n"
        "        raise\n"
        "    return value\n",
        encoding="utf-8",
    )
    rows, _ = _chunks(
        tmp_path, ProjectScope(ProjectScopeKind.FILE, ("src/flow.py",)),
    )
    symbols = {row["symbol"] for row in rows}
    assert len(rows) <= 32
    assert {"apply", "inspect_value"} <= symbols


def test_task_match_precedes_unrelated_units_across_files_without_expanding_budget():
    files = _files(("first", "second"), 3)
    chosen, _ = select_distributed_spans(
        files, max_spans=1, raw_byte_budget=100, serialized_fits=lambda rows: True,
        relevance={("src/second.py", "symbol_2"): 10},
    )
    assert len(chosen) == 1
    assert chosen[0][0:3:2] == ("src/second.py", "symbol_2")


def test_query_reorders_source_without_changing_unmatched_fallback_or_caps(tmp_path):
    from skillforge.integration.source_rule_sources import collect_source_units, legacy_source_rows

    path = tmp_path / "src" / "policy.py"
    path.parent.mkdir()
    path.write_text(
        "def round_amount(value):\n    if value is None: return 0\n    return round(value)\n\n"
        "def reserve_seat(status):\n    if status != 'available': return False\n    return True\n",
        encoding="utf8",
    )
    scope = ProjectScope(ProjectScopeKind.FILE, ("src/policy.py",))
    baseline, _ = _chunks(tmp_path, scope)
    related, _ = _chunks(tmp_path, scope, task_description="reserve seat available status")
    unmatched, _ = _chunks(tmp_path, scope, task_description="预订需要校验可用状态")
    assert related[0]["symbol"] == "reserve_seat"
    assert unmatched == baseline
    legacy = legacy_source_rows(collect_source_units(tmp_path, scope))
    assert len(related) <= len(legacy)
    assert sum(len(r["text"].encode()) for r in related) <= sum(
        len(r["text"].encode()) for r in legacy
    )


def test_whitespace_only_spans_remain_part_of_complete_source_unit():
    from skillforge.integration.source_rule_sources import _file_units

    text = 'def render():\n    return "' + ' ' * 4000 + '"'
    unit = _file_units("src/layout.py", text.encode(), ".py")
    spans = unit.tiers[0][0][1]
    assert any(not excerpt.strip() for _, excerpt in spans)
    assert "".join(excerpt for _, excerpt in spans) == text
