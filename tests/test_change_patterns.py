import json
import subprocess
from dataclasses import replace
from itertools import combinations

import pytest

from skillforge.cli import main
from skillforge.domain import NoiseSignal, NoiseSignalKind, PairAssociation
from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind, GitHistoryResult
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.domain.static_analysis import (
    PythonModuleAnalysis,
    PythonParseStatus,
    StaticAnalysisResult,
)
from skillforge.repository.change_patterns import ChangePatternMiner


@pytest.mark.parametrize("size", [3, 4, 5])
def test_prefix_join_matches_exhaustive_candidate_generation(size):
    import random

    from skillforge.repository.change_patterns import _join_frequent_itemsets

    randomizer = random.Random(125 + size)
    all_previous = tuple(combinations("abcdefghij", size - 1))
    for _ in range(40):
        previous = set(randomizer.sample(all_previous, randomizer.randrange(len(all_previous) + 1)))
        values = sorted({value for item in previous for value in item})
        expected = {candidate for candidate in combinations(values, size)
                    if all(subset in previous for subset in combinations(candidate, size - 1))}
        assert _join_frequent_itemsets(previous) == expected


def test_sparse_unrelated_pairs_do_not_enumerate_all_global_triples():
    from skillforge.repository.change_patterns import _join_frequent_itemsets

    previous = {(f"a{index:04}", f"b{index:04}") for index in range(1000)}
    assert _join_frequent_itemsets(previous) == set()


def _changeset(commit_id, paths, *, is_merge=False, deleted=()):
    files = tuple(
        FileChange(
            path,
            FileChangeKind.DELETED if path in deleted else FileChangeKind.MODIFIED,
            evidence_ids=(f"{commit_id}:{path}:diff", f"{commit_id}:{path}:numstat"),
        )
        for path in paths
    )
    evidence_ids = (f"{commit_id}:log",) + tuple(
        evidence_id for change in files for evidence_id in change.evidence_ids
    )
    return ChangeSet(
        repository_id="repo",
        commit_id=commit_id,
        committed_at="2026-09-25T00:00:00+00:00",
        files=files,
        evidence_ids=evidence_ids,
        is_merge=is_merge,
    )


def _static_result(path_modules=()):
    modules = tuple(
        PythonModuleAnalysis(
            module=module,
            path=path,
            is_package=False,
            symbols=(),
            imports=(),
            parse_status=PythonParseStatus.PARSED,
            parse_issue=None,
            evidence_ids=(f"static:{path}",),
        )
        for path, module in path_modules
    )
    evidence = tuple(
        Evidence(
            evidence_id=f"static:{path}", repository_id="repo",
            kind=EvidenceType.SOURCE, location=SourceLocation(path),
            description=f"Current module {module}", provenance="python_ast",
        )
        for path, module in path_modules
    )
    return StaticAnalysisResult("repo", modules, evidence, ())


def _history(changesets):
    file_evidence = tuple(
        Evidence(
            evidence_id=f"{changeset.commit_id}:{change.path}:{provenance}",
            repository_id="repo",
            kind=EvidenceType.GIT,
            location=SourceLocation(change.path, changeset.commit_id),
            description=f"Git changed {change.path}",
            provenance=f"git_{provenance}",
        )
        for changeset in changesets
        for change in changeset.files
        for provenance in ("diff", "numstat")
    )
    log_evidence = tuple(
        Evidence(
            evidence_id=f"{changeset.commit_id}:log",
            repository_id="repo",
            kind=EvidenceType.GIT,
            location=SourceLocation(".", changeset.commit_id),
            description="Git commit",
            provenance="git_log",
        )
        for changeset in changesets
    )
    return GitHistoryResult("repo", tuple(changesets), file_evidence + log_evidence)


def _mine(changesets, *, min_support=1, max_pattern_size=5, path_modules=(), **kwargs):
    return ChangePatternMiner(
        min_support=min_support, max_pattern_size=max_pattern_size, **kwargs
    ).mine(
        _history(changesets), _static_result(path_modules)
    )


def _pattern(result, kind, values):
    return next(
        pattern
        for pattern in result.patterns
        if pattern.kind.value == kind
        and tuple(item.value for item in pattern.items) == tuple(values)
    )


def test_repeated_file_pair_counts_supporting_commits_and_evidence():
    first = _changeset("c1", ("src/a.py", "tests/test_a.py"))
    second = _changeset("c2", ("tests/test_a.py", "src/a.py"))
    unrelated = _changeset("c3", ("README.md", "docs/usage.md"))

    result = _mine((first, second, unrelated))
    pattern = _pattern(result, "file_co_change", ("src/a.py", "tests/test_a.py"))

    assert pattern.support.supporting == 2
    assert pattern.support.examined == 3
    assert pattern.supporting_commits == ("c1", "c2")
    assert pattern.evidence_ids == tuple(sorted(first.evidence_ids + second.evidence_ids))
    assert not set(pattern.evidence_ids) & set(unrelated.evidence_ids)
    assert {item.evidence_id for item in result.evidence} == {
        evidence_id for output in result.patterns for evidence_id in output.evidence_ids
    }


def test_shared_evidence_appears_once_across_file_and_module_patterns():
    changesets = (
        _changeset("c1", ("a.py", "b.py")),
        _changeset("c2", ("a.py", "b.py")),
    )

    result = _mine(
        changesets,
        min_support=2,
        path_modules=(("a.py", "a"), ("b.py", "b")),
    )

    assert len(result.patterns) == 2
    file_pattern = _pattern(result, "file_co_change", ("a.py", "b.py"))
    module_pattern = _pattern(result, "module_co_change", ("a", "b"))
    assert set(file_pattern.evidence_ids) < set(module_pattern.evidence_ids)
    assert set(module_pattern.evidence_ids) - set(file_pattern.evidence_ids) == {
        "static:a.py", "static:b.py"
    }
    assert tuple(item.evidence_id for item in result.evidence) == module_pattern.evidence_ids


def test_filtered_pattern_does_not_include_its_evidence_in_result():
    first = _changeset("c1", ("a.py", "b.py"))
    repeated = _changeset("c2", ("a.py", "b.py"))
    filtered = _changeset("c3", ("c.py", "d.py"))

    result = _mine((first, repeated, filtered), min_support=2)

    assert len(result.patterns) == 1
    assert tuple(item.evidence_id for item in result.evidence) == tuple(
        sorted(first.evidence_ids + repeated.evidence_ids)
    )
    assert not set(filtered.evidence_ids) & {item.evidence_id for item in result.evidence}


def test_min_support_filters_and_three_changed_files_make_three_pairs():
    triple = _changeset("c1", ("a.py", "b.py", "c.py"))
    repeated = _changeset("c2", ("b.py", "a.py"))

    filtered = _mine((triple, repeated), min_support=2)
    assert [tuple(item.value for item in p.items) for p in filtered.patterns] == [
        ("a.py", "b.py")
    ]

    all_pairs = _mine((triple,), min_support=1)
    assert {
        tuple(item.value for item in pattern.items) for pattern in all_pairs.patterns
    } == {("a.py", "b.py"), ("a.py", "c.py"), ("b.py", "c.py"),
          ("a.py", "b.py", "c.py")}


def test_paths_map_to_modules_only_by_exact_static_analysis_path():
    changesets = (
        _changeset("c1", ("src/shop/api.py", "tests/test_api.py")),
        _changeset("c2", ("src/shop/api.py", "tests/test_api.py")),
    )

    result = _mine(
        changesets,
        min_support=2,
        path_modules=(
            ("src/shop/api.py", "shop.api"),
            ("tests/test_api.py", "tests.test_api"),
        ),
    )

    pattern = _pattern(result, "module_co_change", ("shop.api", "tests.test_api"))
    assert pattern.support.supporting == 2


def test_deleted_or_missing_current_path_keeps_file_pattern_without_fake_module():
    changeset = _changeset(
        "c1", ("src/current.py", "src/removed.py"), deleted=("src/removed.py",)
    )

    result = _mine(
        (changeset,),
        path_modules=(("src/current.py", "src.current"),),
    )

    _pattern(result, "file_co_change", ("src/current.py", "src/removed.py"))
    assert all(pattern.kind.value != "module_co_change" for pattern in result.patterns)


def test_merge_and_empty_changesets_do_not_enter_support_denominator():
    merge = _changeset("merge", ("a.py", "b.py"), is_merge=True)
    regular = _changeset("regular", ("a.py", "b.py"))
    empty = _changeset("empty", ())

    result = _mine((merge, empty, regular))
    pattern = _pattern(result, "file_co_change", ("a.py", "b.py"))

    assert pattern.support.supporting == 1
    assert pattern.support.examined == 1
    assert pattern.supporting_commits == ("regular",)


def test_pattern_identity_and_canonical_order_ignore_input_commit_and_file_order():
    first = _changeset("c1", ("b.py", "a.py", "c.py"))
    second = _changeset("c2", ("c.py", "a.py", "b.py"))

    forward = _mine((first, second))
    reverse = _mine((second, first))

    assert [pattern.pattern_id for pattern in forward.patterns] == [
        pattern.pattern_id for pattern in reverse.patterns
    ]
    assert [pattern.items for pattern in forward.patterns] == [
        pattern.items for pattern in reverse.patterns
    ]
    assert forward.evidence == reverse.evidence
    sort_keys = [
        (
            -pattern.support.supporting,
            pattern.kind.value,
            tuple(item.value for item in pattern.items),
        )
        for pattern in forward.patterns
    ]
    assert sort_keys == sorted(sort_keys)
    assert all(
        tuple(item.value for item in pattern.items)
        == tuple(sorted(item.value for item in pattern.items))
        for pattern in forward.patterns
    )


def test_empty_history_serializes_to_json():
    result = _mine((), min_support=2)

    assert result.patterns == ()
    assert result.evidence == ()
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload == {
        "repository_id": "repo", "patterns": [], "evidence": [], "noise_signals": []
    }


def test_dangling_changeset_evidence_reference_fails_explicitly():
    changeset = _changeset("c1", ("a.py", "b.py"))
    history = GitHistoryResult("repo", (changeset,), ())

    with pytest.raises(ValueError, match="Missing valid git_log Evidence"):
        ChangePatternMiner(min_support=1).mine(history, _static_result())


def test_pair_evidence_only_contains_member_files_and_commit_log():
    changeset = _changeset("c1", ("a.py", "b.py", "c.py", "d.py"))
    result = _mine((changeset,))
    pattern = _pattern(result, "file_co_change", ("a.py", "b.py"))
    assert pattern.evidence_ids == (
        "c1:a.py:diff", "c1:a.py:numstat", "c1:b.py:diff", "c1:b.py:numstat", "c1:log"
    )
    assert all(
        change.evidence_ids == (
            f"c1:{change.path}:diff", f"c1:{change.path}:numstat"
        ) for change in changeset.files
    )
    assert set(changeset.evidence_ids) == {
        "c1:log", *(evidence_id for change in changeset.files
                    for evidence_id in change.evidence_ids)
    }


def test_module_pattern_includes_only_member_static_mapping_evidence():
    changeset = _changeset("c1", ("a.py", "b.py", "c.py"))
    history = _history((changeset,))
    modules = tuple(
        PythonModuleAnalysis(
            module=path[0], path=path, is_package=False, symbols=(), imports=(),
            parse_status=PythonParseStatus.PARSED, parse_issue=None,
            evidence_ids=(f"static:{path}",),
        )
        for path in ("a.py", "b.py", "c.py")
    )
    static = StaticAnalysisResult(
        "repo", modules,
        tuple(Evidence(f"static:{path}", "repo", EvidenceType.SOURCE,
                       SourceLocation(path), "module", "python_ast")
              for path in ("a.py", "b.py", "c.py")), (),
    )
    result = ChangePatternMiner(min_support=1).mine(history, static)
    pattern = _pattern(result, "module_co_change", ("a", "b"))
    assert set(pattern.evidence_ids) == {
        "c1:log", "c1:a.py:diff", "c1:a.py:numstat",
        "c1:b.py:diff", "c1:b.py:numstat", "static:a.py", "static:b.py"
    }
    assert "static:c.py" not in pattern.evidence_ids


def test_min_support_must_be_positive():
    with pytest.raises(ValueError, match="min_support"):
        ChangePatternMiner(min_support=0)


def test_pair_association_uses_canonical_direction_and_raw_universe():
    changesets = (
        _changeset("c1", ("b.py", "a.py")),
        _changeset("c2", ("a.py", "b.py")),
        _changeset("c3", ("a.py",)),
        _changeset("c4", ("c.py",)),
    )
    result = _mine(changesets)
    pattern = _pattern(result, "file_co_change", ("a.py", "b.py"))
    assert pattern.item_occurrences == (3, 2)
    assert pattern.support_ratio == 0.5
    assert pattern.pair_association.first_occurrences == 3
    assert pattern.pair_association.second_occurrences == 2
    assert pattern.pair_association.jaccard == 2 / 3
    assert pattern.pair_association.second_given_first == 2 / 3
    assert pattern.pair_association.first_given_second == 1.0
    assert result.to_dict() == _mine(tuple(reversed(changesets))).to_dict()


def test_triple_requires_real_transaction_support_and_respects_max_size():
    repeated = (
        _changeset("c1", ("a.py", "b.py", "c.py")),
        _changeset("c2", ("a.py", "b.py", "c.py")),
        _changeset("c3", ("a.py", "b.py", "c.py")),
    )
    pair_only = _mine(repeated, min_support=2, max_pattern_size=2)
    assert all(len(pattern.items) == 2 for pattern in pair_only.patterns)
    triples = _mine(repeated, min_support=2, max_pattern_size=3,
                    path_modules=(("a.py", "a"), ("b.py", "b"), ("c.py", "c")))
    for kind, values in (
        ("file_co_change", ("a.py", "b.py", "c.py")),
        ("module_co_change", ("a", "b", "c")),
    ):
        pattern = _pattern(triples, kind, values)
        assert pattern.support.supporting == 3
        assert pattern.pair_association is None
        assert pattern.item_occurrences == (3, 3, 3)
    assert _pattern(triples, "file_co_change", ("a.py", "b.py")).pattern_id == (
        _pattern(pair_only, "file_co_change", ("a.py", "b.py")).pattern_id
    )
    pair_overlap = (
        _changeset("x1", ("a.py", "b.py")),
        _changeset("x2", ("a.py", "b.py")),
        _changeset("x3", ("a.py", "c.py")),
        _changeset("x4", ("a.py", "c.py")),
        _changeset("x5", ("b.py", "c.py")),
        _changeset("x6", ("b.py", "c.py")),
    )
    result = _mine(pair_overlap, min_support=2)
    assert all(len(pattern.items) == 2 for pattern in result.patterns)
    assert _mine(repeated, min_support=4).patterns == ()


def test_max_size_invalid():
    with pytest.raises(ValueError, match="max_pattern_size"):
        ChangePatternMiner(max_pattern_size=1)
    with pytest.raises(SystemExit):
        main(["patterns", ".", "--max-size", "1"])


def test_noise_signals_are_evidence_linked_and_do_not_change_raw_metrics():
    changesets = tuple(
        _changeset(f"c{i}", ("a.py", "b.py", "poetry.lock"))
        for i in range(5)
    )
    normal = _mine(changesets, min_support=2)
    annotated = _mine(
        changesets, min_support=2, large_changeset_threshold=2,
        ubiquitous_threshold=0.8, ubiquitous_min_occurrences=5,
    )
    assert {signal.kind.value for signal in annotated.noise_signals} == {
        "lock_file", "large_changeset", "ubiquitous_item"
    }
    assert len([s for s in annotated.noise_signals if s.kind.value == "large_changeset"]) == 5
    assert [pattern.support for pattern in normal.patterns] == [
        pattern.support for pattern in annotated.patterns
    ]
    assert [pattern.pair_association for pattern in normal.patterns] == [
        pattern.pair_association for pattern in annotated.patterns
    ]
    evidence_ids = {item.evidence_id for item in annotated.evidence}
    assert all(set(signal.evidence_ids) <= evidence_ids for signal in annotated.noise_signals)


def test_small_sample_does_not_trigger_ubiquitous_hint():
    result = _mine((_changeset("c1", ("a.py", "b.py")),))
    assert not any(signal.kind.value == "ubiquitous_item" for signal in result.noise_signals)


def test_public_v2_domain_contracts_are_exported():
    from skillforge import domain

    assert domain.PairAssociation is PairAssociation
    assert domain.NoiseSignal is NoiseSignal
    assert domain.NoiseSignalKind is NoiseSignalKind
    assert {"PairAssociation", "NoiseSignal", "NoiseSignalKind"} <= set(domain.__all__)


def test_missing_commit_log_fails_before_mining():
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    history = replace(
        history, evidence=tuple(item for item in history.evidence
                                if item.provenance != "git_log")
    )
    with pytest.raises(ValueError, match="git_log"):
        ChangePatternMiner(min_support=1).mine(history, _static_result())


@pytest.mark.parametrize("fault", ("wrong_kind", "wrong_commit", "wrong_repo"))
def test_invalid_commit_log_fails(fault):
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    log = next(item for item in history.evidence if item.provenance == "git_log")
    if fault == "wrong_kind":
        replacement = replace(log, kind=EvidenceType.SOURCE)
    elif fault == "wrong_commit":
        replacement = replace(log, location=SourceLocation(".", "other"))
    else:
        replacement = replace(log, repository_id="another-repo")
    history = replace(history, evidence=tuple(
        replacement if item == log else item for item in history.evidence
    ))
    with pytest.raises(ValueError, match="git_log"):
        ChangePatternMiner(min_support=1).mine(history, _static_result())


@pytest.mark.parametrize("missing", ("git_diff", "git_numstat"))
def test_missing_file_evidence_kind_fails(missing):
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    history = replace(
        history, evidence=tuple(item for item in history.evidence
                                if not (item.location.path == "a.py"
                                        and item.provenance == missing))
    )
    with pytest.raises(ValueError, match="FileChange Evidence"):
        ChangePatternMiner(min_support=1).mine(history, _static_result())


def test_file_evidence_cannot_point_at_another_path():
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    evidence = tuple(
        replace(item, location=SourceLocation("b.py", "c1"))
        if item.evidence_id == "c1:a.py:diff" else item
        for item in history.evidence
    )
    with pytest.raises(ValueError, match="Invalid FileChange Evidence"):
        ChangePatternMiner(min_support=1).mine(
            replace(history, evidence=evidence), _static_result()
        )


@pytest.mark.parametrize("fault", ("wrong_kind", "wrong_commit", "wrong_provenance"))
def test_invalid_file_evidence_fails(fault):
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    evidence = next(item for item in history.evidence
                    if item.evidence_id == "c1:a.py:diff")
    if fault == "wrong_kind":
        replacement = replace(evidence, kind=EvidenceType.SOURCE)
    elif fault == "wrong_commit":
        replacement = replace(evidence, location=SourceLocation("a.py", "other"))
    else:
        replacement = replace(evidence, provenance="git_log")
    history = replace(history, evidence=tuple(
        replacement if item == evidence else item for item in history.evidence
    ))
    with pytest.raises(ValueError, match="Invalid FileChange Evidence"):
        ChangePatternMiner(min_support=1).mine(history, _static_result())


def test_file_evidence_must_belong_to_changeset():
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    changeset = history.changesets[0]
    altered = replace(
        changeset, evidence_ids=tuple(item for item in changeset.evidence_ids
                                      if item != "c1:a.py:diff")
    )
    with pytest.raises(ValueError, match="not in ChangeSet"):
        ChangePatternMiner(min_support=1).mine(
            replace(history, changesets=(altered,)), _static_result()
        )


@pytest.mark.parametrize(
    "fault", ("missing_ids", "missing_record", "wrong_path", "wrong_repo", "wrong_kind")
)
def test_mapped_module_requires_valid_static_evidence(fault):
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    static = _static_result((("a.py", "a"), ("b.py", "b")))
    if fault == "missing_ids":
        static = replace(static, modules=(replace(static.modules[0], evidence_ids=()),
                                          static.modules[1]))
    elif fault == "missing_record":
        static = replace(static, evidence=static.evidence[1:])
    elif fault == "wrong_path":
        static = replace(static, evidence=(
            replace(static.evidence[0], location=SourceLocation("b.py")),
            static.evidence[1],
        ))
    elif fault == "wrong_repo":
        static = replace(static, evidence=(
            replace(static.evidence[0], repository_id="another-repo"),
            static.evidence[1],
        ))
    else:
        static = replace(static, evidence=(
            replace(static.evidence[0], kind=EvidenceType.CONFIG),
            static.evidence[1],
        ))
    with pytest.raises(ValueError, match="Static module"):
        ChangePatternMiner(min_support=1).mine(history, static)


def test_git_evidence_repository_must_match_history():
    history = _history((_changeset("c1", ("a.py", "b.py")),))
    evidence = tuple(
        replace(item, repository_id="another-repo")
        if item.evidence_id == "c1:a.py:diff" else item
        for item in history.evidence
    )
    with pytest.raises(ValueError, match="Invalid FileChange Evidence"):
        ChangePatternMiner(min_support=1).mine(
            replace(history, evidence=evidence), _static_result()
        )


def test_duplicate_commit_id_fails_before_support_calculation():
    changeset = _changeset("c1", ("a.py", "b.py"))
    history = _history((changeset,))
    history = replace(history, changesets=(changeset, changeset))
    with pytest.raises(ValueError, match="Duplicate commit_id"):
        ChangePatternMiner(min_support=2).mine(history, _static_result())


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True
    )


def _write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=SkillForge Test",
        "-c",
        "user.email=skillforge@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )


def _cli_repo(repo):
    _git(repo, "init", "--quiet", "-b", "main")
    _write(repo, "src/app.py", "VALUE = 1\n")
    _write(repo, "tests/test_app.py", "assert True\n")
    _commit(repo, "Initial files")
    _write(repo, "src/app.py", "VALUE = 2\n")
    _write(repo, "tests/test_app.py", "assert 2 == 2\n")
    _commit(repo, "Update app and test")


def test_patterns_cli_human_readable_output(tmp_path, capsys):
    _cli_repo(tmp_path)

    assert main(["patterns", str(tmp_path)]) == 0
    output = capsys.readouterr().out

    assert "Patterns: 2" in output
    assert "file co-change [2/2, 100.0%]" in output
    assert "module co-change [2/2, 100.0%]" in output
    assert "src/app.py" in output
    assert "app" in output


def test_patterns_cli_json_serializes_patterns(tmp_path, capsys):
    _cli_repo(tmp_path)

    assert main(["patterns", str(tmp_path), "--min-support", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["repository_id"]
    assert {pattern["kind"] for pattern in payload["patterns"]} == {
        "file_co_change",
        "module_co_change",
    }
    assert all(
        pattern["support"] == {"supporting": 2, "examined": 2}
        for pattern in payload["patterns"]
    )
    evidence_by_id = {item["evidence_id"]: item for item in payload["evidence"]}
    assert evidence_by_id
    assert len(evidence_by_id) == len(payload["evidence"])
    assert all(
        evidence_id in evidence_by_id
        for pattern in payload["patterns"]
        for evidence_id in pattern["evidence_ids"]
    )
    assert {item["kind"] for item in evidence_by_id.values()} == {"git", "source"}
