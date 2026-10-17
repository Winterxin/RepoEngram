"""Coverage trades redundant rules, not primary constraints or byte limits."""

import pytest

from skillforge.integration.source_rule_packing import choose_rule_subset


def _measure(indexes):
    return len(indexes) * 10, len(indexes) * 12


def test_new_entrypoint_displaces_repeat_of_same_binding():
    bindings = {
        0: frozenset({("a.py", "apply")}),
        1: frozenset({("a.py", "apply")}),
        2: frozenset({("b.py", "validate")}),
        3: frozenset({("c.py", "notify")}),
    }
    assert choose_rule_subset(
        range(4), (0, 1, 2), bindings, _measure, max_bytes=36,
    ) == (0, 2, 3)


def test_primary_constraints_cannot_be_evicted_for_large_novel_binding_set():
    bindings = {
        0: frozenset({("a.py", "apply")}),
        1: frozenset({("b.py", "validate")}),
        2: frozenset({("c.py", "one"), ("d.py", "two"), ("e.py", "three")}),
    }
    assert choose_rule_subset(
        range(3), (0, 1), bindings, _measure, max_bytes=24,
    ) == (0, 1)


def test_json_limit_is_enforced_even_when_raw_bytes_fit():
    bindings = {0: frozenset({("a.py", "one")}), 1: frozenset({("b.py", "two")})}
    assert choose_rule_subset(
        range(2), (0,), bindings,
        lambda indexes: (len(indexes) * 10, len(indexes) * 20), max_bytes=30,
    ) == (0,)


def test_candidate_count_is_bounded():
    with pytest.raises(ValueError, match="eight"):
        choose_rule_subset(range(9), (), {}, _measure, max_bytes=100)


def test_same_coverage_keeps_original_packet_despite_more_compact_alternatives():
    bindings = {i: frozenset({("a.py", "apply")}) for i in range(4)}
    costs = [20, 20, 8, 8]
    assert choose_rule_subset(
        range(4), (0, 1), bindings,
        lambda indexes: (sum(costs[i] for i in indexes),) * 2, max_bytes=40,
    ) == (0, 1)
