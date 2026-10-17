import json

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_rule_distillation import _chunks
from skillforge.integration.source_rule_sources import _file_units, legacy_source_rows
from skillforge.integration.source_span_selection import select_distributed_spans


def test_transitive_lookup_is_delivered_with_focused_function_under_original_caps(tmp_path):
    text = (
        "ALIASES = {'amount': {'net', 'gross'}}\n"
        "LOOKUP = {alias: role for role, names in ALIASES.items() for alias in names}\n"
        "def classify(value):\n    return LOOKUP.get(value)\n"
        "def unrelated(value):\n    return value + 1\n"
    )
    (tmp_path / 'policy.py').write_text(text, encoding='utf-8')
    units = _file_units('policy.py', text.encode(), '.py')
    assert units.dependencies['classify'] == ('module.LOOKUP', 'module.ALIASES')
    rows, _ = _chunks(tmp_path, ProjectScope(ProjectScopeKind.FILE, ('policy.py',)),
                      focus_task='Update classify()')
    assert {row['symbol'] for row in rows} == {'classify', 'module.LOOKUP', 'module.ALIASES'}
    old = legacy_source_rows((units,))
    assert len(rows) <= len(old)
    assert sum(len(r['text'].encode()) for r in rows) <= sum(len(r['text'].encode()) for r in old)
    assert 'amount' in json.dumps(rows)


def test_bundle_is_skipped_atomically_and_shared_constants_are_not_duplicated():
    files = [('f.py', 'hash', ([(1, [('first', 'A')]), (0, [('other', 'B')])],
                              [(0, [('module.TABLE', 'table')])]))]
    dependencies = {('f.py', 'first'): ('module.TABLE',),
                    ('f.py', 'other'): ('module.TABLE',)}
    rows, skipped = select_distributed_spans(
        files, max_spans=1, raw_byte_budget=100, serialized_fits=lambda _: True,
        allowed_symbols=frozenset({('f.py', 'first')}), dependencies=dependencies,
    )
    assert rows == []
    assert ('f.py', 'first', 'unit_exceeds_span_limit') in skipped
    rows, _ = select_distributed_spans(
        files, max_spans=3, raw_byte_budget=100, serialized_fits=lambda _: True,
        dependencies=dependencies,
    )
    assert [symbol for _, _, symbol, _ in rows].count('module.TABLE') == 1
    assert len(rows) == 3


def test_shadowed_ambiguous_and_sensitive_bindings_are_not_resolved():
    text = (
        "TABLE = {'ready': True}\n"
        "AMBIGUOUS = 1\nAMBIGUOUS = 2\n"
        "API_KEY = 'sensitive-canary-42'\n"
        "def shadow(TABLE): return TABLE\n"
        "def ambiguous(): return AMBIGUOUS\n"
        "def nested():\n    def inner(): return TABLE\n    return inner\n"
        "def normal(): return TABLE\n"
    )
    units = _file_units('f.py', text.encode(), '.py')
    assert not {'shadow', 'ambiguous', 'nested'} & units.dependencies.keys()
    assert units.dependencies['normal'] == ('module.TABLE',)
    assert 'API_KEY' not in json.dumps(units.dependencies)
    assert 'sensitive-canary-42' not in str(units.tiers)


def test_dependency_overflow_is_explicit_and_non_python_has_no_dependencies():
    text = '\n'.join(f'VALUE_{i} = {i}' for i in range(9))
    text += '\ndef consume(): return [' + ', '.join(f'VALUE_{i}' for i in range(9)) + ']\n'
    units = _file_units('f.py', text.encode(), '.py')
    assert units.dependencies['consume'] is None
    rows, skipped = select_distributed_spans(
        [units.selection_file()], max_spans=32, raw_byte_budget=4096,
        serialized_fits=lambda _: True,
        dependencies={('f.py', key): value for key, value in units.dependencies.items()},
    )
    assert not any(symbol == 'consume' for _, _, symbol, _ in rows)
    assert ('f.py', 'consume', 'unit_dependency_unavailable') in skipped
    assert _file_units('readme.md', b'hello', '.md').dependencies == {}


def test_unrelated_duplicate_function_units_retain_previous_selection_behavior():
    files = [('f.py', 'hash', ([(0, [('entry', 'first definition')]),
                               (0, [('entry', 'second definition')])],))]
    rows, _ = select_distributed_spans(
        files, max_spans=4, raw_byte_budget=100, serialized_fits=lambda _: True,
    )
    assert len(rows) == 2
    assert {text for _, _, _, text in rows} == {'first definition', 'second definition'}
