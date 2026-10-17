import json
import subprocess
from dataclasses import replace
from itertools import combinations

import pytest

from skillforge.cli import main
from skillforge.domain.change import (
    ChangePattern,
    ChangePatternItem,
    ChangePatternKind,
    ChangePatternResult,
    ChangeSet,
    FileChange,
    FileChangeKind,
    GitHistoryResult,
    NoiseSignal,
    NoiseSignalKind,
    PairAssociation,
)
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.static_analysis import (
    ImportRelation,
    ImportResolution,
    PythonModuleAnalysis,
    PythonParseStatus,
    StaticAnalysisResult,
)
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


@pytest.mark.parametrize("random_seed", [4, 17, 29])
def test_indexed_closed_patterns_and_families_match_original_rules(random_seed):
    import random

    from skillforge.repository.workflow_candidates import _closed_seeds, _pattern_families

    randomizer = random.Random(random_seed)
    patterns = [replace(_pattern(
        randomizer.choice((FILE, MODULE)),
        randomizer.sample(list("abcdefgh"), randomizer.randrange(2, 7)),
        commits=randomizer.choice((("c1", "c2"), ("c1", "c3"), ("c2", "c3"))),
    ), pattern_id=str(index % 19)) for index in range(160)]
    patterns += [patterns[5], patterns[7]]  # preserve duplicate IDs and input ordering
    expected = [p for p in patterns if not any(
        p.kind == larger.kind and set(item.value for item in p.items)
        < set(item.value for item in larger.items)
        and p.supporting_commits == larger.supporting_commits for larger in patterns
    )]
    assert _closed_seeds(patterns) == expected
    expected_families = [(p, tuple(sorted((other for other in patterns
        if other.kind == p.kind and set(item.value for item in other.items)
        <= set(item.value for item in p.items)), key=lambda item: item.pattern_id)))
        for p in expected]
    assert _pattern_families(expected, patterns) == expected_families


def test_custom_large_itemsets_preserve_strict_subset_and_support_identity():
    from skillforge.repository.workflow_candidates import _closed_seeds, _pattern_families

    small = _pattern(FILE, tuple("abcdefghi"))
    large = _pattern(FILE, tuple("abcdefghij"))
    distinct_support = _pattern(FILE, tuple("abcdefghi"), commits=("c2", "c3"))
    patterns = [small, large, distinct_support, large]
    assert _closed_seeds(patterns) == [large, distinct_support, large]
    for seed, family in _pattern_families(_closed_seeds(patterns), patterns):
        assert family == tuple(sorted((p for p in patterns if p.kind == seed.kind
            and set(item.value for item in p.items)
            <= set(item.value for item in seed.items)), key=lambda p: p.pattern_id))


def test_dense_itemsets_do_not_rebuild_values_for_each_pattern_pair(monkeypatch):
    import skillforge.repository.workflow_candidates as mining

    patterns = [_pattern(FILE, members) for size in range(2, 6)
                for members in combinations("abcdefghijklmn", size)]
    calls = 0
    original = mining._values

    def counted(pattern):
        nonlocal calls
        calls += 1
        return original(pattern)

    monkeypatch.setattr(mining, "_values", counted)
    closed = mining._closed_seeds(patterns)
    families = mining._pattern_families(closed, patterns)
    assert len(closed) == 2002
    assert all(len(seed.items) == 5 for seed in closed)
    assert all(len(family) == 26 for _, family in families)
    assert calls < 5 * len(patterns)

FILE = ChangePatternKind.FILE_CO_CHANGE
MODULE = ChangePatternKind.MODULE_CO_CHANGE


def _evidence(evidence_id, *, repository_id="repo"):
    return Evidence(
        evidence_id, repository_id, EvidenceType.SOURCE,
        SourceLocation("a.py"), evidence_id, "fixture",
    )


def _pattern(kind, values, commits=("c1", "c2"), *, examined=4, metrics=None):
    values = tuple(sorted(values))
    metrics = metrics or PairAssociation(3, 4, 0.4, 0.6, 0.5)
    return ChangePattern(
        pattern_id=f"{kind.value}:{','.join(values)}",
        repository_id="repo", kind=kind,
        items=tuple(ChangePatternItem(value) for value in values),
        supporting_commits=commits,
        evidence_ids=(f"e:{kind.value}:{','.join(values)}",),
        support=Support(len(commits), examined),
        support_ratio=len(commits) / examined,
        item_occurrences=(len(commits),) * len(values),
        pair_association=metrics if len(values) == 2 else None,
    )


def _family(kind, values, commits=("c1", "c2"), *, examined=4):
    return (
        *(_pattern(kind, pair, commits, examined=examined)
          for pair in combinations(sorted(values), 2)),
        _pattern(kind, values, commits, examined=examined),
    )


def _static(relations=(), *, evidence=(), path_modules=(), module_evidence_ids=None):
    module_evidence_ids = module_evidence_ids or {}
    modules = tuple(PythonModuleAnalysis(
        module=module_name, path=path, is_package=False, symbols=(),
        imports=tuple(relation for relation in relations
                      if relation.source_module == module_name),
        parse_status=PythonParseStatus.PARSED,
        parse_issue=None, evidence_ids=tuple(module_evidence_ids.get(path, ())),
    ) for path, module_name in path_modules)
    mapped_sources = {module_name for _, module_name in path_modules}
    modules += tuple(PythonModuleAnalysis(
        module=None, path=f"orphan{i}.py", is_package=False, symbols=(),
        imports=(relation,), parse_status=PythonParseStatus.PARSED,
        parse_issue=None, evidence_ids=(),
    ) for i, relation in enumerate(relations)
        if relation.source_module not in mapped_sources)
    return StaticAnalysisResult("repo", modules, evidence, ())


def _import(source, target, *, resolution=ImportResolution.INTERNAL, evidence_id=None):
    evidence_id = evidence_id or f"import:{source}:{target}"
    return ImportRelation(
        "repo", source, target, target, (), False, resolution,
        "a.py", 1, (evidence_id,),
    )


def _mine(
    patterns, *, relations=(), signals=(), static_evidence=None,
    path_modules=(), missing_modules=(), history_commits=None, module_evidence_ids=None,
):
    pattern_ids = {evidence_id for pattern in patterns for evidence_id in pattern.evidence_ids}
    signal_ids = {evidence_id for signal in signals for evidence_id in signal.evidence_ids}
    pattern_result = ChangePatternResult(
        "repo", tuple(patterns),
        tuple(_evidence(key) for key in sorted(pattern_ids | signal_ids)),
        tuple(signals),
    )
    if static_evidence is None:
        static_evidence = tuple(_evidence(key) for key in sorted({
            evidence_id for relation in relations for evidence_id in relation.evidence_ids
        } | {
            evidence_id for ids in (module_evidence_ids or {}).values() for evidence_id in ids
        }))
    module_values = {
        item.value for pattern in patterns if pattern.kind == MODULE for item in pattern.items
    } - set(missing_modules)
    mappings = tuple(path_modules) + tuple(
        (f"{value.replace('.', '/')}.py", value) for value in sorted(module_values)
    )
    if history_commits is None:
        history_commits = sorted({
            commit for pattern in patterns for commit in pattern.supporting_commits
        })
    history = GitHistoryResult("repo", tuple(ChangeSet(
        "repo", commit, "2026-09-25T00:00:00+00:00",
        (FileChange("a", FileChangeKind.MODIFIED),), (), parent_ids=("parent",),
    ) for commit in history_commits), ())
    return WorkflowCandidateMiner().mine(
        history, _static(
            relations, evidence=tuple(static_evidence), path_modules=mappings,
            module_evidence_ids=module_evidence_ids,
        ),
        pattern_result,
    )


def test_repeated_pair_is_candidate_with_complete_source_metrics_and_evidence():
    pattern = _pattern(FILE, ("a", "b"))
    result = _mine((pattern,))
    candidate, = result.candidates
    assert tuple(m.value for m in candidate.members) == ("a", "b")
    assert candidate.support == pattern.support
    assert candidate.supporting_commits == pattern.supporting_commits
    assert candidate.support_ratio == pattern.support_ratio
    assert candidate.source_pattern_ids == (pattern.pattern_id,)
    assert candidate.pair_associations[0].source_pattern_id == pattern.pattern_id
    assert candidate.pair_associations[0].jaccard == pattern.pair_association.jaccard
    assert candidate.evidence_ids == pattern.evidence_ids
    assert result.source_patterns == (pattern,)
    assert tuple(e.evidence_id for e in result.evidence) == pattern.evidence_ids
    assert candidate.observation is None


def test_single_pair_occurrence_has_no_candidate_or_evidence():
    result = _mine((_pattern(FILE, ("a", "b"), ("c1",)),))
    assert result.candidates == ()
    assert result.source_patterns == ()
    assert result.evidence == ()


@pytest.mark.parametrize("values", [("a", "b", "c"), ("a", "b", "c", "d")])
def test_multi_item_seed_preserves_support_commits_and_all_pair_metrics(values):
    patterns = _family(MODULE, values)
    result = _mine(patterns)
    candidate = result.candidates[0]
    source = patterns[-1]
    assert tuple(member.value for member in candidate.members) == values
    assert tuple(member.current_module for member in candidate.members) == values
    assert candidate.current_modules == values
    assert candidate.support == source.support
    assert candidate.support_ratio == source.support_ratio
    assert candidate.supporting_commits == source.supporting_commits
    assert len(candidate.pair_associations) == len(tuple(combinations(values, 2)))
    for association in candidate.pair_associations:
        pair = next(pattern for pattern in patterns if
                    tuple(item.value for item in pattern.items)
                    == (association.first, association.second))
        assert association.source_pattern_id == pair.pattern_id
        assert association.jaccard == pair.pair_association.jaccard
        assert association.second_given_first == pair.pair_association.second_given_first
        assert association.first_given_second == pair.pair_association.first_given_second
        assert pair.evidence_ids[0] in candidate.evidence_ids


def test_equivalent_nested_seed_is_suppressed_but_remains_source_family():
    patterns = (*_family(FILE, ("a", "b", "c", "d")),
                _pattern(FILE, ("a", "b", "c")))
    result = _mine(patterns)
    assert len(result.candidates) == 1
    assert len(result.candidates[0].members) == 4
    assert patterns[-1].pattern_id in result.candidates[0].source_pattern_ids
    assert patterns[-1].evidence_ids[0] in {
        item.evidence_id for item in result.evidence
    }


def test_nested_seed_with_different_support_and_different_kind_are_retained():
    base = _family(FILE, ("a", "b", "c", "d"), ("c1", "c2"))
    triple = _pattern(FILE, ("a", "b", "c"), ("c1", "c2", "c3"))
    other_kind = _family(MODULE, ("a", "b", "c"))
    result = _mine((*base, triple, *other_kind))
    assert {(candidate.kind, len(candidate.members)) for candidate in result.candidates} == {
        (FILE, 3), (FILE, 4), (MODULE, 3),
    }
    four = next(candidate for candidate in result.candidates
                if candidate.kind == FILE and len(candidate.members) == 4)
    assert triple.pattern_id in four.source_pattern_ids


def test_source_family_contains_pairs_multi_item_subsets_and_no_unrelated_pattern():
    values = ("a", "b", "c", "d")
    pairs = tuple(_pattern(FILE, pair) for pair in combinations(values, 2))
    triples = tuple(_pattern(FILE, triple) for triple in combinations(values, 3))
    seed = _pattern(FILE, values)
    unrelated = _pattern(FILE, ("x", "y"))
    result = _mine((*pairs, *triples, seed, unrelated))
    assert len(result.candidates) == 2
    candidate = next(c for c in result.candidates if c.seed_pattern_id == seed.pattern_id)
    expected_ids = {pattern.pattern_id for pattern in (*pairs, *triples, seed)}
    assert candidate.seed_pattern_id == seed.pattern_id
    assert set(candidate.source_pattern_ids) == expected_ids
    assert {pattern.pattern_id for pattern in result.source_patterns} == expected_ids | {
        unrelated.pattern_id
    }
    assert unrelated.pattern_id not in candidate.source_pattern_ids
    assert unrelated.evidence_ids[0] not in candidate.evidence_ids
    assert {pair.source_pattern_id for pair in candidate.pair_associations} <= expected_ids
    assert {evidence_id for source in result.source_patterns
            for evidence_id in source.evidence_ids} <= {
                item.evidence_id for item in result.evidence
            }


def test_different_support_nested_candidates_keep_separate_source_families():
    values = ("a", "b", "c", "d")
    eight = tuple(f"c{i}" for i in range(8))
    five = eight[:5]
    pairs = tuple(_pattern(FILE, pair, eight, examined=10)
                  for pair in combinations(values, 2))
    triple = _pattern(FILE, ("a", "b", "c"), eight, examined=10)
    four = _pattern(FILE, values, five, examined=10)
    result = _mine((*pairs, triple, four))
    assert len(result.candidates) == 5
    by_seed = {candidate.seed_pattern_id: candidate for candidate in result.candidates}
    assert triple.pattern_id in by_seed[four.pattern_id].source_pattern_ids
    assert four.pattern_id not in by_seed[triple.pattern_id].source_pattern_ids
    assert by_seed[triple.pattern_id].support.supporting == 8
    assert by_seed[four.pattern_id].support.supporting == 5


def test_missing_pair_pattern_fails_explicitly():
    patterns = _family(FILE, ("a", "b", "c"))
    with pytest.raises(ValueError, match="Missing pair ChangePattern"):
        _mine(patterns[1:])


def test_internal_links_are_directed_deduplicated_and_coherence_counts_unordered_pairs():
    patterns = _family(MODULE, ("a", "b", "c", "d"))
    relations = (_import("a", "b"), _import("b", "a"), _import("b", "c"),
                 _import("a", "outside"),
                 _import("a", "c", resolution=ImportResolution.EXTERNAL),
                 _import("c", "d", resolution=ImportResolution.UNRESOLVED))
    candidate = _mine(patterns, relations=relations).candidates[0]
    assert [(link.source, link.target) for link in candidate.structural_links] == [
        ("a", "b"), ("b", "a"), ("b", "c"),
    ]
    assert candidate.structural_coherence.linked_pairs == 2
    assert candidate.structural_coherence.possible_pairs == 6
    assert candidate.structural_coherence.link_ratio == 2 / 6
    assert all(set(link.evidence_ids) <= set(candidate.evidence_ids)
               for link in candidate.structural_links)


def test_zero_links_does_not_filter_module_or_mapped_file_candidate():
    patterns = (*_family(MODULE, ("a", "b", "c")),
                *_family(FILE, ("a.py", "b.py", "c.py")))
    result = _mine(patterns)
    module = next(candidate for candidate in result.candidates if candidate.kind == MODULE)
    file = next(candidate for candidate in result.candidates if candidate.kind == FILE)
    assert module.structural_links == ()
    assert module.structural_coherence.link_ratio == 0
    assert file.structural_links == ()
    assert file.current_modules == ("a", "b", "c")
    assert file.structural_coherence.link_ratio == 0


def test_file_candidate_exact_current_mapping_imports_and_unmapped_member():
    patterns = _family(FILE, ("a.py", "b.py", "unmapped.py"))
    result = _mine(
        patterns, relations=(_import("pkg.a", "pkg.b"),),
        path_modules=(("a.py", "pkg.a"), ("b.py", "pkg.b"),
                      ("elsewhere/unmapped.py", "pkg.unrelated")),
        module_evidence_ids={"a.py": ("module:a",), "b.py": ("module:b",)},
    )
    candidate = result.candidates[0]
    assert tuple(member.value for member in candidate.members) == (
        "a.py", "b.py", "unmapped.py"
    )
    assert candidate.current_modules == ("pkg.a", "pkg.b")
    assert [(member.value, member.current_module) for member in candidate.members] == [
        ("a.py", "pkg.a"), ("b.py", "pkg.b"), ("unmapped.py", None),
    ]
    assert [(link.source, link.target) for link in candidate.structural_links] == [
        ("pkg.a", "pkg.b")
    ]
    assert candidate.structural_coherence.linked_pairs == 1
    assert candidate.structural_coherence.possible_pairs == 1
    assert candidate.structural_coherence.link_ratio == 1
    assert {"module:a", "module:b", "import:pkg.a:pkg.b"} <= set(candidate.evidence_ids)
    assert {"module:a", "module:b", "import:pkg.a:pkg.b"} <= {
        item.evidence_id for item in result.evidence
    }


@pytest.mark.parametrize("mapped", [(), (("a.py", "pkg.a"),)])
def test_file_candidate_with_fewer_than_two_current_modules_has_none_coherence(mapped):
    candidate = _mine(
        _family(FILE, ("a.py", "b.py", "c.py")), path_modules=mapped
    ).candidates[0]
    assert candidate.current_modules == tuple(sorted(module for _, module in mapped))
    assert candidate.structural_links == ()
    assert candidate.structural_coherence is None


def test_module_member_missing_from_current_static_analysis_fails():
    with pytest.raises(ValueError, match="Missing current module.*c"):
        _mine(_family(MODULE, ("a", "b", "c")), missing_modules=("c",))


@pytest.mark.parametrize("missing", ["c1", "c2"])
def test_missing_supporting_commit_in_seed_or_source_fails(missing):
    patterns = _family(FILE, ("a", "b", "c"))
    with pytest.raises(ValueError, match="Missing supporting commits"):
        _mine(patterns, history_commits=(missing,))


def test_missing_commit_referenced_only_by_pair_source_fails():
    patterns = list(_family(FILE, ("a", "b", "c")))
    patterns[0] = _pattern(FILE, ("a", "b"), ("c1", "c2", "c3"))
    with pytest.raises(ValueError, match="Missing supporting commits.*c3"):
        _mine(patterns, history_commits=("c1", "c2"))


def test_noise_is_relevant_only_and_does_not_change_raw_metrics():
    patterns = _family(FILE, ("a", "b", "poetry.lock"))
    signals = (
        NoiseSignal(NoiseSignalKind.LOCK_FILE, "poetry.lock", "exact", ("noise:lock",)),
        NoiseSignal(
            NoiseSignalKind.UBIQUITOUS_ITEM, "file_co_change:a", "count", ("noise:common",)
        ),
        NoiseSignal(NoiseSignalKind.LARGE_CHANGESET, "c1", "size", ("noise:large",)),
        NoiseSignal(NoiseSignalKind.LARGE_CHANGESET, "unrelated", "size", ("noise:other",)),
        NoiseSignal(
            NoiseSignalKind.UBIQUITOUS_ITEM, "module_co_change:a", "count",
            ("noise:wrong-kind",),
        ),
    )
    candidate = _mine(patterns, signals=signals).candidates[0]
    baseline = _mine(patterns).candidates[0]
    assert {signal.kind for signal in candidate.noise_signals} == set(NoiseSignalKind)
    assert candidate.support == baseline.support
    assert candidate.pair_associations == baseline.pair_associations
    assert candidate.structural_coherence == baseline.structural_coherence
    assert "noise:other" not in candidate.evidence_ids
    assert "noise:wrong-kind" not in candidate.evidence_ids


def test_evidence_closure_shared_dedup_and_determinism():
    patterns = _family(MODULE, ("a", "b", "c"))
    relations = (_import("a", "b"), _import("b", "c"))
    first = _mine(patterns, relations=relations)
    second = _mine(tuple(reversed(patterns)), relations=tuple(reversed(relations)))
    assert first == second
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    candidate = first.candidates[0]
    assert tuple(item.evidence_id for item in first.evidence) == candidate.evidence_ids
    assert len(first.evidence) == len(set(candidate.evidence_ids))
    assert candidate.candidate_id == second.candidates[0].candidate_id


def test_candidate_identity_does_not_depend_on_support_or_evidence():
    first = _mine(_family(FILE, ("a", "b", "c"))).candidates[0]
    later = _mine(_family(
        FILE, ("a", "b", "c"), ("c1", "c2", "c3"), examined=5
    )).candidates[0]
    assert first.candidate_id == later.candidate_id
    assert first.support != later.support


def test_dangling_import_evidence_fails_even_for_unselected_relation():
    with pytest.raises(ValueError, match="Missing ImportRelation Evidence"):
        _mine(_family(FILE, ("a", "b", "c")),
              relations=(_import("outside", "elsewhere"),), static_evidence=())


def test_repository_mismatch_and_conflicting_evidence_fail():
    patterns = _family(MODULE, ("a", "b", "c"))
    pattern_result = ChangePatternResult(
        "repo", patterns,
        tuple(_evidence(key) for key in sorted(
            pattern.evidence_ids[0] for pattern in patterns
        )),
    )
    with pytest.raises(ValueError, match="same repository_id"):
        WorkflowCandidateMiner().mine(
            GitHistoryResult("other", (), ()), _static(), pattern_result,
        )
    relation = _import("a", "b", evidence_id=patterns[0].evidence_ids[0])
    conflict = replace(_evidence(patterns[0].evidence_ids[0]), description="different")
    with pytest.raises(ValueError, match="Conflicting Evidence ID"):
        WorkflowCandidateMiner().mine(
            GitHistoryResult("repo", tuple(ChangeSet(
                "repo", commit, "2026-09-25T00:00:00+00:00", (), ()
            ) for commit in ("c1", "c2")), ()),
            _static((relation,), evidence=(conflict,), path_modules=(
                ("a.py", "a"), ("b.py", "b"), ("c.py", "c"),
            )),
            pattern_result,
        )


@pytest.mark.parametrize("flag", ["--min-support", "--max-size"])
def test_cli_rejects_invalid_positive_options(flag):
    with pytest.raises(SystemExit) as error:
        main(["workflow-candidates", ".", flag, "0"])
    assert error.value.code == 2


def test_cli_human_json_and_empty_candidate_result(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert main(["workflow-candidates", str(tmp_path)]) == 0
    assert "Workflow Candidates: 0" in capsys.readouterr().out
    assert main(["workflow-candidates", str(tmp_path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["candidates"] == []
    assert output["source_patterns"] == []
    assert output["evidence"] == []
