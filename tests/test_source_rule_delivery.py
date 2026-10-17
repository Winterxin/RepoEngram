"""Lossless candidate rendering and atomic delivery byte budgets."""

import json

import pytest

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_rule_delivery import (
    budget_source_rules,
    render_source_rules,
)


def _scope():
    return ProjectScope(ProjectScopeKind.DIRECTORY, ("src",))


def _sources():
    return [
        {"source_id": "source-a", "path": "src/policy.py", "symbol": "apply", "text": ""},
        {"source_id": "source-b", "path": "src/policy.py", "symbol": "fallback", "text": ""},
    ]


def _rule(condition, action, quote, *, source_id="source-a", exceptions=None, ordering=None):
    return {
        "condition": condition,
        "action": action,
        "exceptions": exceptions or [],
        "ordering": ordering or [],
        "entrypoints": [{"source_id": source_id, "symbol": "apply"}],
        "evidence": [{"source_id": source_id, "quote": quote}],
    }


def _sizes(body, evidence):
    raw = len(body.encode("utf-8")) + len(evidence.encode("utf-8"))
    escaped = len(json.dumps([body, evidence], ensure_ascii=True,
                             separators=(",", ":")).encode("utf-8"))
    return raw, escaped


def _candidate_fields(body, index):
    section = body.split(f"## Candidate {index}\n", 1)[1].split("## Candidate ", 1)[0]
    return {
        name: json.loads(value)
        for line in section.splitlines()
        if line.startswith("- ")
        for name, value in [line[2:].split(": ", 1)]
    }


def test_render_preserves_fields_order_scope_and_maps_deduplicated_quotes():
    rules = [
        _rule(
            "first condition with `markup`\nand another line",
            "preserve zero → result",
            "if value == 0:\r\n    return 0\r\n",
            exceptions=["missing remains distinct", "explicitly ordered"],
            ordering=["check missing first"],
        ),
        _rule(
            "second condition",
            "raise the original error",
            "if value == 0:\n    return 0\n",
            exceptions=["do not replace the error"],
        ),
        _rule("third condition", "return fallback", "return fallback", source_id="source-b"),
    ]
    body, evidence = render_source_rules(rules, _sources(), _scope())

    assert "CURRENT_SOURCE_RULE_CANDIDATE" in body
    assert "advisory" in body and "Explicit scope" in body
    assert (
        body.index("## Candidate 1") < body.index("## Candidate 2")
        < body.index("## Candidate 3")
    )
    assert "- condition: \"first condition with `markup`\\nand another line\"" in body
    assert "- exceptions: [\"missing remains distinct\",\"explicitly ordered\"]" in body
    assert "- ordering: [\"check missing first\"]" in body
    assert "- entrypoints: [[\"source-a\",\"apply\"]]" in body
    assert "- evidence_refs: [\"E1\"]" in body
    assert body.count("- evidence_refs: [\"E1\"]") == 2
    assert "- evidence_refs: [\"E2\"]" in body
    fields = _candidate_fields(body, 1)
    assert fields == {
        "condition": "first condition with `markup`\nand another line",
        "action": "preserve zero → result",
        "exceptions": ["missing remains distinct", "explicitly ordered"],
        "ordering": ["check missing first"],
        "entrypoints": [["source-a", "apply"]],
        "evidence_refs": ["E1"],
    }
    assert "exceptions" not in _candidate_fields(body, 3)
    assert "ordering" not in _candidate_fields(body, 3)
    assert evidence.count(
        '- E1 ("source-a"): ' + json.dumps("if value == 0:\n    return 0\n")
    ) == 1
    assert evidence.count('- "source-a" -> P1') == 1
    assert evidence.count('- P1 -> "src/policy.py"') == 1
    assert "return fallback" in evidence


def test_budget_keeps_whole_rules_in_model_order_and_skips_oversized_rule():
    rules = [
        _rule("first", "keep first", "return first"),
        _rule("large", "drop whole", "x" * 600),
        _rule("last", "keep last", "return last", source_id="source-b"),
    ]
    only_kept = [rules[0], rules[2]]
    body, evidence = render_source_rules(only_kept, _sources(), _scope())
    max_bytes = max(_sizes(body, evidence))

    delivered = budget_source_rules(rules, _sources(), _scope(), max_bytes=max_bytes)
    assert delivered.retained_rule_indexes == (0, 2)
    assert delivered.dropped_rules == ((1, "rule_exceeds_byte_budget"),)
    assert delivered.body.index("keep first") < delivered.body.index("keep last")
    assert "drop whole" not in delivered.body
    assert "return first" in delivered.evidence and "return last" in delivered.evidence
    assert "x" * 600 not in delivered.evidence
    assert max(_sizes(delivered.body, delivered.evidence)) <= max_bytes


def test_oversized_single_rule_is_skipped_and_prefix_budget_fails_explicitly():
    rule = _rule("large", "large", "payload" * 200)
    prefix, evidence = render_source_rules((), _sources(), _scope())
    prefix_bytes = max(_sizes(prefix, evidence))
    delivered = budget_source_rules([rule], _sources(), _scope(), max_bytes=prefix_bytes)
    assert delivered.retained_rule_indexes == ()
    assert delivered.dropped_rules == ((0, "rule_exceeds_byte_budget"),)
    with pytest.raises(ValueError, match="prefix exceeds byte budget"):
        budget_source_rules([rule], _sources(), _scope(), max_bytes=prefix_bytes - 1)


def test_cap_length_unclosed_backtick_summary_is_skipped_with_reason():
    rule = _rule("condition", "`" + "x" * 399, "return value")
    delivered = budget_source_rules([rule], _sources(), _scope())
    assert delivered.retained_rule_indexes == ()
    assert delivered.dropped_rules == ((0, "suspected_truncated_summary"),)


def test_cap_length_balanced_backticks_remain_deliverable():
    rule = _rule("condition", "`" + "x" * 398 + "`", "return value")
    delivered = budget_source_rules([rule], _sources(), _scope())
    assert delivered.retained_rule_indexes == (0,)
    assert delivered.dropped_rules == ()
    assert "`" + "x" * 398 + "`" in delivered.body


@pytest.mark.parametrize("max_bytes", [-1, True, 1.5])
def test_budget_rejects_invalid_byte_limit(max_bytes):
    with pytest.raises(ValueError, match="max_bytes"):
        budget_source_rules([], _sources(), _scope(), max_bytes=max_bytes)


def test_budget_preserves_primary_rule_and_adds_missing_entrypoint_with_complete_evidence():
    sources = [
        {"source_id": "a1", "path": "src/a.py", "symbol": "apply"},
        {"source_id": "a2", "path": "src/a.py", "symbol": "apply"},
        {"source_id": "b", "path": "src/b.py", "symbol": "notify"},
    ]
    rules = [
        _rule("primary", "keep primary", "p" * 80, source_id="a1"),
        _rule("repeat", "repeat same function", "r" * 250, source_id="a2"),
        _rule("failure", "preserve failure contract", "f" * 100, source_id="b",
              exceptions=["propagate failure"], ordering=["write before notify"]),
    ]
    rules[2]["entrypoints"][0]["symbol"] = "notify"
    maximum = max(_sizes(*render_source_rules(rules[:2], sources, _scope())))
    delivery = budget_source_rules(rules, sources, _scope(), max_bytes=maximum)
    assert delivery.retained_rule_indexes == (0, 2)
    assert "keep primary" in delivery.body and "preserve failure contract" in delivery.body
    assert _candidate_fields(delivery.body, 2)["exceptions"] == ["propagate failure"]
    assert _candidate_fields(delivery.body, 2)["ordering"] == ["write before notify"]
    assert "f" * 100 in delivery.evidence and "r" * 250 not in delivery.evidence
    assert max(_sizes(delivery.body, delivery.evidence)) <= maximum


def test_explicit_entrypoint_can_replace_irrelevant_rule_without_cutting_evidence():
    sources = _sources()
    rules = [_rule('first', 'unrelated behavior', 'a' * 150),
             _rule('on failure', 'preserve caller fallback', 'b' * 150,
                   source_id='source-b', exceptions=['keep zero'], ordering=['validate first'])]
    rules[1]['entrypoints'][0]['symbol'] = 'fallback'
    limit = max(max(_sizes(*render_source_rules([rule], sources, _scope()))) for rule in rules)
    old = budget_source_rules(rules, sources, _scope(), max_bytes=limit)
    new = budget_source_rules(rules, sources, _scope(), max_bytes=limit,
                              preferred_bindings=frozenset({('src/policy.py', 'fallback')}))
    assert old.retained_rule_indexes == (0,)
    assert new.retained_rule_indexes == (1,)
    assert 'b' * 150 in new.evidence
    assert _candidate_fields(new.body, 1)['exceptions'] == ['keep zero']
    assert _candidate_fields(new.body, 1)['ordering'] == ['validate first']
    assert max(_sizes(new.body, new.evidence)) <= limit
