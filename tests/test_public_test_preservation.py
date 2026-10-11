"""Preservation checks do not reward changed expectations as added coverage."""
import subprocess
from functools import partial
from pathlib import Path

import pytest

from skillforge.verify.public_tests import (
    _test_expectations,
    preservation_check,
    preserved_test_expectations,
)
from skillforge.verify.public_tests import (
    changed_tests as configured_changed_tests,
)
from skillforge.verify.public_tests import (
    qualifying_tests as configured_qualifying_tests,
)

# The legacy fixture remains supported only through explicit repository configuration.
qualifying_tests = partial(configured_qualifying_tests, production_modules=('app.services',))
changed_tests = partial(configured_changed_tests, directory='apps/api/tests',
                        production_modules=('app.services',))

IMPORT = 'from app.services.example import build\n'
ORIGINAL = IMPORT + '''def test_report():
    result = build()
    assert result == "original"
'''


def test_replacing_expected_value_is_not_new_coverage():
    assert qualifying_tests(ORIGINAL, ORIGINAL.replace('"original"', '"changed"')) == []


def test_weakening_assertion_is_not_new_coverage():
    assert qualifying_tests(ORIGINAL, ORIGINAL.replace('result == "original"', 'True')) == []


def test_preserved_assertion_plus_added_behavior_check_qualifies():
    after = ORIGINAL + '    assert result.startswith("ori")\n'
    assert qualifying_tests(ORIGINAL, after) == ['test_report']


def test_added_test_with_production_call_and_assertion_qualifies():
    assert qualifying_tests('', ORIGINAL) == ['test_report']


def test_fixture_or_comment_edit_without_extra_assertion_does_not_qualify():
    assert qualifying_tests(ORIGINAL, ORIGINAL.replace('build()', 'build(mode="new")')) == []
    assert qualifying_tests(ORIGINAL, ORIGINAL + '    # updated\n') == []


def test_literal_and_nested_assertions_do_not_establish_coverage():
    source = IMPORT + 'def test_literal():\n    build()\n    assert True\n'
    assert qualifying_tests('', source) == []
    assert qualifying_tests('', IMPORT + '''def test_nested():
    build()
    def ignored():
        assert build() == "ok"
''') == []


def test_new_test_cannot_mask_removed_old_assertion():
    after = IMPORT + '''def test_other():
    assert build() == "new"
'''
    assert qualifying_tests(ORIGINAL, after) == []


def test_whitespace_and_assert_message_changes_do_not_remove_behavior():
    after = ORIGINAL.replace('assert result == "original"',
                             'assert (result == "original"), "message"')
    assert qualifying_tests(ORIGINAL, after + '    assert len(result) > 0\n') == ['test_report']


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, encoding='utf-8').strip()


def test_explicit_baseline_survives_agent_commit_and_weakening(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    folder = tmp_path / 'apps/api/tests'
    folder.mkdir(parents=True)
    target = folder / 'test_report.py'
    target.write_text(ORIGINAL, encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'baseline')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    target.write_text(ORIGINAL.replace('"original"', '"changed"'), encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'agent expectation edit')
    with pytest.raises(ValueError, match='assertions changed or removed'):
        changed_tests(tmp_path, base_commit=base)


def test_untracked_new_test_is_selected_but_unchanged_files_are_not_read(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    folder = tmp_path / 'apps/api/tests'
    folder.mkdir(parents=True)
    unchanged = folder / 'test_existing.py'
    unchanged.write_text(ORIGINAL, encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'baseline')
    target = folder / 'test_new.py'
    target.write_text(ORIGINAL.replace('test_report', 'test_new_report'), encoding='utf-8')
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path != unchanged, 'Unchanged test source must not be scanned'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert changed_tests(tmp_path) == ['apps/api/tests/test_new.py::test_new_report']


@pytest.mark.parametrize('budget,message,tracked', [
    ('MAX_CHANGED_TEST_FILES', 'file budget exceeded', False),
    ('MAX_TEST_SOURCE_BYTES', 'current public test source budget exceeded', False),
    ('MAX_TEST_SOURCE_BYTES', 'baseline public test source budget exceeded', True),
    ('MAX_TOTAL_TEST_SOURCE_BYTES', 'total public test source budget exceeded', False),
])
def test_read_budgets_fail_without_selecting_partial_coverage(
    tmp_path, monkeypatch, budget, message, tracked
):
    from skillforge.verify import public_tests
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    folder = tmp_path / 'apps/api/tests'
    folder.mkdir(parents=True)
    (tmp_path / 'README.md').write_text('Fixture', encoding='utf-8')
    target = folder / 'test_budget.py'
    if tracked:
        target.write_text(ORIGINAL, encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'baseline')
    target.write_text(ORIGINAL + '    assert len(result) > 0\n', encoding='utf-8')
    monkeypatch.setattr(public_tests, budget, 0 if budget == 'MAX_CHANGED_TEST_FILES' else 8)
    with pytest.raises(ValueError, match=message):
        changed_tests(tmp_path)


def test_generic_guard_distinguishes_classes_and_preserves_literal_assertions():
    source = '''class TestOne:
    def test_same(self):
        assert result == "original"
class TestTwo:
    def test_same(self):
        assert True
'''
    before = _test_expectations(source)
    after = _test_expectations(source.replace('assert True', 'assert False'))
    assert before[('TestOne', 'test_same')] == after[('TestOne', 'test_same')]
    assert before[('TestTwo', 'test_same')] - after[('TestTwo', 'test_same')]


def test_formal_preservation_does_not_require_app_imports_or_new_assertions(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    target = tmp_path / 'test_contract.py'
    source = 'def test_contract():\n    assert result == "original"\n'
    target.write_text(source, 'utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'baseline')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    target.write_text(source + '    # unchanged expectation\n', 'utf-8')
    assert preserved_test_expectations(tmp_path, base) == 1
    target.write_text(source.replace('"original"', '"changed"'), 'utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'actor commits changed expectation')
    with pytest.raises(ValueError, match='existing test assertions changed or removed'):
        preserved_test_expectations(tmp_path, base)
    target.unlink()
    with pytest.raises(ValueError, match='existing test assertions changed or removed'):
        preserved_test_expectations(tmp_path, base)


def test_preservation_check_never_claims_skill_obligation_coverage():
    check = preservation_check('fixed-before-agent')
    assert check.covers == ()
    assert '--preserve-only' in check.argv
    assert check.argv[-1] == 'fixed-before-agent'
    assert check.timeout_seconds == 30


def test_system_checkout_line_endings_do_not_consume_changed_file_budget(tmp_path, monkeypatch):
    config = tmp_path / 'system.gitconfig'
    config.write_text('[core]\n    autocrlf = true\n', 'utf-8')
    empty = tmp_path / 'global.gitconfig'
    empty.write_text('', 'utf-8')
    monkeypatch.setenv('GIT_CONFIG_SYSTEM', str(config))
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(empty))
    monkeypatch.delenv('GIT_CONFIG_NOSYSTEM', raising=False)
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    for index in range(25):
        (root / f'test_{index}.py').write_bytes(b'def test_ok():\r\n    assert value == 1\r\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'unchanged CRLF checkout')
    assert preserved_test_expectations(root, 'HEAD') == 0
