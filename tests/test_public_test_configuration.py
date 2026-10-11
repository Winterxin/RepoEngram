"""Explicit test selection works across layouts without project-specific defaults."""
import subprocess
import sys

import pytest

from skillforge.verify.public_tests import behavioral_nodes, changed_tests, main


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, encoding='utf-8').strip()


@pytest.mark.parametrize('module,directory', [
    ('warehouse.inventory', 'quality/contracts'),
    ('audio.transforms', 'packages/sound/checks'),
])
def test_independent_layout_selection_preserves_original_assertions(tmp_path, module, directory):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    (tmp_path / 'README.md').write_text('Selection fixture', 'utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'empty baseline')
    target = tmp_path / directory / 'test_contract.py'
    target.parent.mkdir(parents=True)
    source = f'''from {module} import execute as operation
def helper():
    return operation()
def test_contract():
    result = helper()
    assert result == "kept"
'''
    target.write_text(source, 'utf-8')
    expected = [f'{directory}/test_contract.py::test_contract']
    assert changed_tests(tmp_path, directory, production_modules=(module,)) == expected
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'original contract')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    target.write_text(source + '    assert len(result) == 4\n', 'utf-8')
    assert changed_tests(tmp_path, directory, base, production_modules=(module,)) == expected
    target.write_text(source.replace('"kept"', '"changed"')
                      + '    assert len(result) == 7\n', 'utf-8')
    with pytest.raises(ValueError, match='assertions changed or removed'):
        changed_tests(tmp_path, directory, base, production_modules=(module,))


@pytest.mark.parametrize('statement,call', [
    ('from domain import execute', 'execute()'),
    ('from domain.engine import execute as operation', 'operation()'),
    ('import domain.engine as engine', 'engine.execute()'),
])
def test_exact_prefix_children_and_aliases_link(statement, call):
    source = f'{statement}\ndef test_contract():\n    assert {call} == "ok"\n'
    assert set(behavioral_nodes(source, production_modules=('domain',))) == {'test_contract'}


def test_similar_prefix_other_project_and_relative_imports_do_not_link():
    for statement in ('from domain_extra import execute', 'from app.services import execute',
                      'from .domain import execute'):
        source = f'{statement}\ndef test_contract():\n    assert execute() == "ok"\n'
        assert behavioral_nodes(source, production_modules=('domain',)) == {}


def test_invalid_configuration_fails_before_repository_access(tmp_path, monkeypatch):
    def unexpected_git(*args, **kwargs):
        raise AssertionError('Invalid configuration must fail before Git access')
    monkeypatch.setattr('skillforge.verify.public_tests.subprocess.run', unexpected_git)
    for modules in ((), 'domain', ('domain..engine',), ('domain',) * 17):
        with pytest.raises(ValueError, match='module prefixes'):
            changed_tests(tmp_path, 'checks', production_modules=modules)


def test_cli_has_no_implicit_repository_layout(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['public-tests'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
