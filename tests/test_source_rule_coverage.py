"""Explicit task entrypoints survive delivery without expanding evidence or budgets."""

from skillforge.integration.source_rule_coverage import (
    coverage_report,
    requested_bindings,
    rule_bindings,
)
from skillforge.integration.source_rule_packing import prioritize_rule_baseline


def test_unique_explicit_source_names_only_and_duplicate_chunks_are_one_binding():
    sources = [dict(path='engine.py', symbol='Engine.commit'),
               dict(path='engine.py', symbol='Engine.commit'),
               dict(path='legacy.py', symbol='Legacy.commit'),
               dict(path='engine.py', symbol='module.SETTINGS')]
    assert requested_bindings(sources, 'Fix Engine.commit') == {('engine.py', 'Engine.commit')}
    assert not requested_bindings(sources, 'Fix commit()')
    assert not requested_bindings(sources, 'handle transactional consistency')
    assert not requested_bindings(sources, 'fix invented_method()')
    assert not requested_bindings(sources, 'Change module.SETTINGS')


def test_unrelated_primary_rules_cannot_crowd_out_explicit_requested_entrypoint():
    bindings = {i: frozenset({('engine.py', symbol)})
                for i, symbol in enumerate(('read', 'list', 'commit'))}
    def measure(indexes):
        return len(indexes) * 10, len(indexes) * 12
    selected = prioritize_rule_baseline(
        range(3), (0, 1), bindings, measure, max_bytes=24,
        preferred=frozenset({('engine.py', 'commit')}),
    )
    assert selected == (2, 0)
    assert max(measure(selected)) <= 24
    # Vague tasks and an already-covered explicit entrypoint keep the old packet.
    for preferred in (frozenset(), frozenset({('engine.py', 'read')})):
        assert prioritize_rule_baseline(
            range(3), (0, 1), bindings, measure, max_bytes=24, preferred=preferred,
        ) == (0, 1)


def test_oversized_requested_rule_is_not_truncated_or_reported_as_delivered():
    bindings = {0: frozenset({('engine.py', 'read')}),
                1: frozenset({('engine.py', 'commit')})}
    selected = prioritize_rule_baseline(
        range(2), (0,), bindings,
        lambda ids: (sum(10 if i == 0 else 100 for i in ids),) * 2,
        max_bytes=20, preferred=frozenset({('engine.py', 'commit')}),
    )
    assert selected == (0,)
    report = coverage_report(frozenset({('engine.py', 'commit'), ('engine.py', 'abort')}),
                             bindings, selected)
    assert report['missing_from_generation'] == [['engine.py', 'abort']]
    assert report['dropped_by_budget'] == [['engine.py', 'commit']]
    assert report['delivered'] == []


def test_coverage_is_bound_to_real_source_ids_not_similarity_or_claimed_prose():
    sources = [dict(source_id='s1', path='engine.py', symbol='commit')]
    rules = [dict(entrypoints=[dict(source_id='s1', symbol='commit')],
                  action='also handles abort', evidence=[])]
    report = coverage_report(frozenset({('engine.py', 'commit'), ('engine.py', 'abort')}),
                             rule_bindings(rules, sources), (0,))
    assert report['delivered'] == [['engine.py', 'commit']]
    assert report['missing_from_generation'] == [['engine.py', 'abort']]
