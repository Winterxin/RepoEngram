"""Run changed public behavioral tests; never load effect oracles or target patches."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from skillforge.filesystem_safety import no_symlink
from skillforge.repository.bounded_git_patch_evidence import _run_git_limited

MAX_CHANGED_TEST_FILES = 20
MAX_TEST_SOURCE_BYTES = 256 * 1024
MAX_TOTAL_TEST_SOURCE_BYTES = 1024 * 1024


def _own_nodes(node):
    """Assertions in nested definitions do not verify the enclosing test."""
    pending = list(node.body)
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield item
        pending.extend(ast.iter_child_nodes(item))


def _assertions(node):
    # Literal-only assertions cannot establish behavior of a production call.
    return Counter(ast.dump(item.test, include_attributes=False)
                   for item in _own_nodes(node) if isinstance(item, ast.Assert)
                   and any(isinstance(part, (ast.Name, ast.Call, ast.Attribute, ast.Subscript))
                           for part in ast.walk(item.test)))


def _module_prefixes(production_modules):
    """Explicit import linkage configuration, never inferred from a fixture layout."""
    if (not isinstance(production_modules, (tuple, list))
            or not 0 < len(production_modules) <= 16):
        raise ValueError('declare 1-16 production module prefixes')
    if any(not isinstance(prefix, str) or len(prefix) > 128
           or not all(part.isidentifier() for part in prefix.split('.'))
           for prefix in production_modules):
        raise ValueError('production module prefixes must be absolute dotted identifiers')
    return tuple(dict.fromkeys(production_modules))


def behavioral_nodes(source, *, production_modules):
    """Declared absolute import/call linkage; not proof of behavioral coverage."""
    prefixes = _module_prefixes(production_modules)
    def matches(module):
        return any(module == prefix or module.startswith(prefix + '.') for prefix in prefixes)
    tree = ast.parse(source)
    imports = {item.asname or item.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module and not node.level
               and matches(node.module)
               for item in node.names}
    imports.update(item.asname or item.name.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for item in node.names
                   if matches(item.name))
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    def linked(node, seen):
        for item in ast.walk(node):
            if not isinstance(item, ast.Call):
                continue
            func = item.func
            while isinstance(func, ast.Attribute):
                func = func.value
            if isinstance(func, ast.Name):
                if func.id in imports:
                    return True
                if (func.id in functions and func.id not in seen
                        and linked(functions[func.id], seen | {func.id})):
                    return True
        # Fixture-mediated calls in this public module have the same linkage rule.
        for arg in node.args.args:
            if (arg.arg in functions and arg.arg not in seen
                    and linked(functions[arg.arg], seen | {arg.arg})):
                return True
        return False
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            and any(isinstance(n, ast.Assert) for n in ast.walk(node))
            and linked(node, {node.name})}


def test_execution_covers(plan):
    """Explicit narrow mapping; never claim coverage of arbitrary Skill conditions."""
    actions = {"update corresponding tests", "add corresponding tests", "update regression tests",
               "add regression tests", "run regression tests", "run corresponding tests"}
    conditions = {None, "", "tests pass", "regression tests pass"}
    return [o.target_ref for o in plan.obligations
            if o.target_ref.kind.value == "WORKFLOW_STEP"
            and o.description.strip().rstrip(".").casefold() in actions
            and (o.condition.casefold() if o.condition else o.condition) in conditions]


def qualifying_tests(before_source, after_source, *, production_modules):
    """Conservative syntactic addition; never claim semantic strengthening.

    Existing assertions must survive verbatim as AST expressions. A modified
    test needs an additional behavior assertion; editing fixtures or expected
    values alone is not new coverage. Intentional contract changes require a
    different explicit check, rather than borrowing this preservation claim.
    """
    before = behavioral_nodes(before_source, production_modules=production_modules)
    after = behavioral_nodes(after_source, production_modules=production_modules)
    before_nodes = {node.name: node for node in ast.walk(ast.parse(before_source))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith('test_')}
    after_nodes = {node.name: node for node in ast.walk(ast.parse(after_source))
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name.startswith('test_')}
    # Deleting or weakening an existing behavioral test cannot be masked by
    # adding a different test in the same changed file.
    for name in before:
        if (name not in after_nodes
                or _assertions(before_nodes[name]) - _assertions(after_nodes[name])):
            return []
    selected = []
    for name in after:
        assertions = _assertions(after_nodes[name])
        if not assertions:
            continue
        if name not in before_nodes:
            selected.append(name)
        elif before.get(name) != after[name] and assertions - _assertions(before_nodes[name]):
            selected.append(name)
    return sorted(selected)


def changed_tests(root, directory, base_commit="HEAD", *, production_modules):
    prefixes = _module_prefixes(production_modules)
    root = Path(root).resolve()
    if not isinstance(directory, str) or not directory:
        raise ValueError('declare a repository-relative test directory')
    selector = Path(directory)
    if selector.is_absolute() or any(part in {'..', '.'} for part in selector.parts):
        raise ValueError('test directory must be repository-relative')
    no_symlink(root / selector, root)
    if not isinstance(base_commit, str) or not base_commit or base_commit.startswith('-'):
        raise ValueError('invalid test baseline')
    resolved = subprocess.run(['git', '-c', f'safe.directory={root}', 'rev-parse',
                               '--verify', '--end-of-options', f'{base_commit}^{{commit}}'],
                              cwd=root, capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
    def paths(argv):
        result = subprocess.run(['git', '-c', f'safe.directory={root}', *argv],
                                cwd=root, capture_output=True, check=True, timeout=10)
        return result.stdout.decode('utf-8').split('\0')
    changed = set(paths(['diff', '--name-only', '-z', resolved, '--', directory]))
    changed.update(paths(['ls-files', '--others', '--exclude-standard', '-z', '--', directory]))
    changed = sorted(name for name in changed if name and name.endswith('.py')
                     and Path(name).name.startswith('test_'))
    if len(changed) > MAX_CHANGED_TEST_FILES:
        raise ValueError('changed public test file budget exceeded; use an explicit focused check')
    selected = []
    used = 0
    for relative in changed:
        path = root / relative
        no_symlink(path, root)
        original = _run_git_limited(root, ('show', f'{resolved}:{relative}'), MAX_TEST_SOURCE_BYTES)
        if original.truncated:
            raise ValueError('baseline public test source budget exceeded')
        before = original.stdout.decode('utf-8-sig') if original.returncode == 0 else ''
        if path.is_file():
            with path.open('rb') as stream:
                current = stream.read(MAX_TEST_SOURCE_BYTES + 1)
            if len(current) > MAX_TEST_SOURCE_BYTES:
                raise ValueError('current public test source budget exceeded')
            after = current.decode('utf-8-sig')
        else:
            after = ''
        used += len(original.stdout) + len(after.encode('utf-8'))
        if used > MAX_TOTAL_TEST_SOURCE_BYTES:
            raise ValueError('total public test source budget exceeded')
        qualifying = qualifying_tests(before, after, production_modules=prefixes)
        # Missing baseline expectations make the preservation check fail,
        # even if another file added passing tests.
        old_nodes = {n.name: n for n in ast.walk(ast.parse(before))
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        new_nodes = {n.name: n for n in ast.walk(ast.parse(after))
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if any(name not in new_nodes or _assertions(old_nodes[name]) - _assertions(new_nodes[name])
               for name in behavioral_nodes(before, production_modules=prefixes)):
            raise ValueError(f'existing public test assertions changed or removed: {relative}')
        selected.extend(f'{relative}::{name}' for name in qualifying)
    return selected


def _test_expectations(source):
    """Qualified pytest function identities, with literal assertions preserved too."""
    result = {}

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith('test_'):
                    identity = (*prefix, node.name)
                    if identity in result:
                        raise ValueError('duplicate test identity cannot prove preservation')
                    result[identity] = Counter(ast.dump(item.test, include_attributes=False)
                                               for item in _own_nodes(node)
                                               if isinstance(item, ast.Assert))

    visit(ast.parse(source).body, ())
    return result


def preserved_test_expectations(root, base_commit):
    """Bounded changed-file syntax guard; independent of import linkage/new coverage.

    New files need no preservation proof. Renames are observed as deletion plus
    addition, conservatively rejecting removal of an existing asserted test.
    This does not prove fixture, helper, parameterization or runtime semantics.
    """
    root = Path(root).resolve()
    if not isinstance(base_commit, str) or not base_commit or base_commit.startswith('-'):
        raise ValueError('invalid test baseline')
    resolved = _run_git_limited(root, ('rev-parse', '--verify', '--end-of-options',
                                     f'{base_commit}^{{commit}}'), 1024,
                                inherit_system_config=True)
    if resolved.returncode or resolved.truncated:
        raise ValueError('cannot resolve test preservation baseline')
    base = resolved.stdout.decode('ascii').strip()
    changed = _run_git_limited(root, ('diff', '--name-status', '-z', '--no-renames',
                                     base, '--', '.'), 64 * 1024,
                               inherit_system_config=True)
    if changed.returncode or changed.truncated:
        raise ValueError('test preservation changed-path budget/read failed')
    parts = changed.stdout.decode('utf-8').split('\0')
    if parts.pop() != '' or len(parts) % 2:
        raise ValueError('invalid test preservation changed paths')
    paths = []
    for status, relative in zip(parts[::2], parts[1::2], strict=True):
        if status not in {'A', 'M', 'D', 'T'}:
            raise ValueError('unsupported test preservation change status')
        name = Path(relative).name
        if status != 'A' and name.endswith('.py') and (
            name.startswith('test_') or name.endswith('_test.py')
        ):
            paths.append(relative)
    if len(paths) > MAX_CHANGED_TEST_FILES:
        raise ValueError('changed public test file budget exceeded; use an explicit focused check')
    used = 0
    for relative in paths:
        path = root / relative
        no_symlink(path, root)
        original = _run_git_limited(root, ('show', f'{base}:{relative}'), MAX_TEST_SOURCE_BYTES,
                                    inherit_system_config=True)
        if original.returncode or original.truncated:
            raise ValueError('baseline test preservation source unreadable/over budget: '
                             + relative)
        current = b''
        if path.exists():
            if not path.is_file():
                raise ValueError(f'current test preservation source is not a file: {relative}')
            with path.open('rb') as stream:
                current = stream.read(MAX_TEST_SOURCE_BYTES + 1)
        if len(current) > MAX_TEST_SOURCE_BYTES:
            raise ValueError('current public test source budget exceeded')
        used += len(original.stdout) + len(current)
        if used > MAX_TOTAL_TEST_SOURCE_BYTES:
            raise ValueError('total public test source budget exceeded')
        before = _test_expectations(original.stdout.decode('utf-8-sig'))
        after = _test_expectations(current.decode('utf-8-sig'))
        for name, assertions in before.items():
            if assertions and (name not in after or assertions - after[name]):
                identity = f'{relative}::{"::".join(name)}'
                raise ValueError(f'existing test assertions changed or removed: {identity}')
    return len(paths)


def preservation_check(base_commit):
    """An existing formal Verify check; grants no natural-language coverage."""
    from skillforge.domain.verification import VerifyCheckKind, VerifyCommandSpec
    return VerifyCommandSpec('zz-skillforge-preserve-test-expectations', VerifyCheckKind.CUSTOM,
                             (sys.executable, '-m', 'skillforge.verify.public_tests',
                              '--preserve-only', '--base-commit', base_commit),
                             timeout_seconds=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--directory", help='explicit repository-relative test directory')
    parser.add_argument('--production-module', action='append',
                        help='absolute production import prefix; repeat for multiple modules')
    parser.add_argument('--base-commit', default='HEAD',
                        help='fixed pre-edit commit; use the Verify knowledge baseline')
    parser.add_argument('--preserve-only', action='store_true',
                        help='check changed pytest assertions only; do not execute pytest')
    args = parser.parse_args()
    if args.preserve_only:
        try:
            count = preserved_test_expectations(args.repo.resolve(), args.base_commit)
        except (ValueError, OSError, SyntaxError, UnicodeError,
                RecursionError, RuntimeError) as error:
            print(f'Test expectation preservation failed: {error}', file=sys.stderr)
            return 1
        print(json.dumps({'changed_test_files_checked': count,
                          'scope': 'syntactic assertions; not behavioral coverage'}))
        return 0
    if not args.directory or not args.production_module:
        parser.error('test selection requires --directory and --production-module; '
                     'no project layout is assumed')
    try:
        nodes = changed_tests(args.repo.resolve(), args.directory, args.base_commit,
                              production_modules=args.production_module)
    except (ValueError, OSError, SyntaxError, UnicodeError,
            RecursionError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Public test selection failed: {error}', file=sys.stderr)
        return 1
    if not nodes:
        print('No new public behavior assertion preserving existing test expectations.')
        return 1
    print(json.dumps({"public_behavioral_tests": nodes}, ensure_ascii=False), flush=True)
    # This proves execution, not full natural-language obligation fulfillment.
    # Only explicit test-execution obligations may declare this check as covers.
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *nodes], cwd=args.repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
