"""Cross-path role mining through real local Git and Python facts."""

import json
import subprocess
from dataclasses import replace

import pytest

from skillforge.domain.change import ChangePatternKind
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner

ROLE = ChangePatternKind.ADDED_PYTHON_MODULE_ROLE


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(root, paths, message):
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# local fixture\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def _repo(tmp_path, events, *, initial=("atlas/__init__.py",), min_support=2):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Local test")
    _git(root, "config", "user.email", "local@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, initial, "base")
    commits = [_commit(root, paths, f"feature {i}") for i, paths in enumerate(events)]
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner(min_support=min_support).mine(history, facts)
    candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    return root, commits, history, facts, patterns, candidates


@pytest.mark.parametrize("count", [3, 4])
def test_distinct_additions_form_neutral_role_with_closed_evidence(tmp_path, count):
    events = [(f"atlas/unit_{i}.py",) for i in range(count)]
    _, commits, history, _, patterns, result = _repo(tmp_path, events)
    roles = [pattern for pattern in patterns.patterns if pattern.kind == ROLE]
    assert len(roles) == 1
    pattern = roles[0]
    assert pattern.supporting_commits == tuple(sorted(commits))
    assert pattern.support.supporting == count
    assert pattern.support.examined == count + 1
    assert len(pattern.structural_events) == count
    assert pattern.items[0].value == "ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER(atlas)"
    candidate = next(item for item in result.candidates if item.kind == ROLE)
    assert candidate.structural_events == pattern.structural_events
    assert candidate.supporting_commits == pattern.supporting_commits
    assert candidate.pair_associations == ()
    assert candidate.observation.startswith("Observed tendency:")
    evidence = {item.evidence_id: item for item in result.evidence}
    for event in candidate.structural_events:
        refs = [evidence[key] for key in event.evidence_ids]
        assert any(item.provenance == "git_diff" and item.location.path == event.path
                   and item.location.commit == event.commit_id for item in refs)
        assert any(item.provenance == "git_parent_package" for item in refs)
        raw = next(item for item in history.changesets if item.commit_id == event.commit_id)
        assert event.path in {item.path for item in raw.files}
    text = json.dumps(result.to_dict()).lower()
    assert not any(word in text for word in (
        "supportdesk", "assignment", "reopening", "moderation", "mandatory", "must", "unseen",
    ))
    assert all(f"unit_{i}.py" not in candidate.observation for i in range(count))


@pytest.mark.parametrize("events,initial", [
    ([("atlas/a.py", "atlas/b.py", "atlas/c.py")], ("atlas/__init__.py",)),
    ([("foo/a.py",), ("bar/b.py",), ("baz/c.py",)],
     ("foo/__init__.py", "bar/__init__.py", "baz/__init__.py")),
    ([("atlas/a.md",), ("atlas/b.json",), ("atlas/cache.bin",)], ("atlas/__init__.py",)),
    ([("atlas/a.py",), ("atlas/b.py",)], ("atlas/__init__.py",)),
    ([("atlas/a.py",), ("atlas/b.py",), ("atlas/c.py",)], ("README.md",)),
    ([("atlas/__init__.py", "atlas/a.py", "atlas/b.py", "atlas/c.py")], ("README.md",)),
])
def test_non_independent_or_unproven_events_do_not_form_roles(tmp_path, events, initial):
    *_, patterns, candidates = _repo(tmp_path, events, initial=initial)
    assert not any(item.kind == ROLE for item in patterns.patterns)
    assert not any(item.kind == ROLE for item in candidates.candidates)


def test_test_package_role_stays_structural_and_higher_support_is_respected(tmp_path):
    events = [(f"tests/test_{i}.py",) for i in range(3)]
    *_, candidates = _repo(tmp_path, events, initial=("tests/__init__.py",))
    candidate = next(item for item in candidates.candidates if item.kind == ROLE)
    assert candidate.observation == (
        "Observed tendency: repeated changes added new direct child Python modules under tests."
    )
    other = tmp_path / "higher"
    other.mkdir()
    *_, patterns, _ = _repo(other, events, initial=("tests/__init__.py",), min_support=4)
    assert not any(item.kind == ROLE for item in patterns.patterns)


def test_exact_path_patterns_are_unchanged(tmp_path):
    root, _, _, _, _, _ = _repo(tmp_path, [("atlas/a.py", "atlas/b.py")])
    for i in range(3):
        for name in ("a", "b"):
            with (root / f"atlas/{name}.py").open("a", encoding="utf-8") as stream:
                stream.write(f"# modification {i}\n")
        _git(root, "commit", "-qam", f"modify {i}")
    history = GitHistoryMiner().mine(root)
    patterns = ChangePatternMiner().mine(history, PythonStaticAnalyzer().analyze(root))
    pair = next(item for item in patterns.patterns
                if item.kind == ChangePatternKind.FILE_CO_CHANGE
                and tuple(member.value for member in item.items) == ("atlas/a.py", "atlas/b.py"))
    assert pair.support.supporting == 4
    assert not any(item.kind == ROLE for item in patterns.patterns)


def test_multiple_files_per_commit_preserve_events_without_inflating_support(tmp_path):
    events = [(f"atlas/a{i}.py", f"atlas/b{i}.py") for i in range(3)]
    _, _, history, _, _, candidates = _repo(tmp_path, events)
    candidate = next(item for item in candidates.candidates if item.kind == ROLE)
    assert candidate.support.supporting == 3
    assert len(candidate.structural_events) == 6
    assert len(candidate.supporting_commits) == 3
    assert all(len(change.evidence_ids) == len(set(change.evidence_ids))
               for change in history.changesets)
    assert not NegativePatternMiner().mine(history).candidates


def test_readding_one_path_cannot_supply_distinct_path_support(tmp_path):
    root, _, _, _, _, _ = _repo(tmp_path, [("atlas/a.py",)])
    for i in range(3):
        _git(root, "rm", "atlas/a.py")
        _git(root, "commit", "-qm", f"remove {i}")
        _commit(root, ("atlas/a.py",), f"readd {i}")
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    assert not any(item.kind == ROLE for item in patterns.patterns)


def test_source_layout_and_nested_packages_use_static_module_mapping(tmp_path):
    events = [(f"src/nebula/plugins/item_{i}.py",) for i in range(3)]
    *_, candidates = _repo(tmp_path, events, initial=(
        "src/nebula/__init__.py", "src/nebula/plugins/__init__.py",
    ))
    candidate = next(item for item in candidates.candidates if item.kind == ROLE)
    assert candidate.members[0].value == (
        "ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER(nebula.plugins)"
    )


def test_parent_package_evidence_is_validated(tmp_path):
    events = [(f"atlas/unit_{i}.py",) for i in range(3)]
    _, _, history, facts, _, _ = _repo(tmp_path, events)
    evidence = tuple(
        replace(item, location=replace(item.location, path="wrong/__init__.py"))
        if item.provenance == "git_parent_package" else item
        for item in history.evidence
    )
    with pytest.raises(ValueError, match="historical parent package Evidence"):
        ChangePatternMiner().mine(replace(history, evidence=evidence), facts)
