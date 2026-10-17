"""Exact sharing cannot merge candidates, grant Evidence or lose source bytes."""

import json

import pytest

from skillforge.repository.semantic_context_catalog import ExactContextCatalog
from skillforge.repository.semantic_wire import CandidateAliases


def test_repeated_source_is_shared_without_sharing_candidate_identity():
    source = "def route():\n    return '你好'\n" * 100
    contexts = [
        {
            "candidate_id": key * 24,
            "evidence": [{"evidence_id": key * 32, "body": source, "path": f"{key}.py"}],
        }
        for key in ("a", "b")
    ]
    aliases = [CandidateAliases(f"r{i}", value) for i, value in enumerate(contexts)]
    encoded = [alias.encode(value) for alias, value in zip(aliases, contexts, strict=True)]
    catalog = ExactContextCatalog(encoded)
    assert list(catalog.blocks.values()) == [source]
    assert [
        alias.decode(catalog.expand(value, catalog.blocks))
        for alias, value in zip(aliases, catalog.contexts, strict=True)
    ] == contexts
    assert len(json.dumps([catalog.contexts, catalog.blocks]).encode()) < len(
        json.dumps(encoded).encode()
    )
    with pytest.raises(ValueError, match="foreign"):
        aliases[0].decode({"evidence_ids": [aliases[1].encode_ids["b" * 32]]})


def test_exact_subtree_sharing_keeps_order_duplicates_and_conflicting_records():
    common = {"source": "x" * 800, "path": "a.py", "lines": [3, 1, 3]}
    contexts = [
        {"records": [common, common], "same_id": "a"},
        {"records": [common, {**common, "path": "b.py"}], "same_id": "a"},
    ]
    catalog = ExactContextCatalog(contexts)
    assert [catalog.expand(value, catalog.blocks) for value in catalog.contexts] == contexts
    assert any("$skillforge_context" in json.dumps(block) for block in catalog.blocks.values())


@pytest.mark.parametrize(
    "value,blocks,reason",
    [
        ({"$skillforge_context": "absent"}, {}, "unknown"),
        ({"$skillforge_context": "b0", "extra": 1}, {"b0": "x"}, "ambiguous"),
        ({"$skillforge_context": "b0"}, {"b0": {"$skillforge_context": "b0"}}, "cyclic"),
        ({"$skillforge_context": []}, {}, "unknown"),
    ],
)
def test_invalid_reference_fails_closed(value, blocks, reason):
    with pytest.raises(ValueError, match=reason):
        ExactContextCatalog.expand(value, blocks)


def test_reserved_original_input_and_unprofitable_sharing():
    with pytest.raises(ValueError, match="reserved"):
        ExactContextCatalog([{"$skillforge_context": "original"}])
    values = [{"body": "a" * 500}, {"body": "b" * 500}]
    catalog = ExactContextCatalog(values)
    assert not catalog.blocks
    assert catalog.contexts == values


def test_shared_spelling_does_not_authorize_foreign_evidence():
    shared = {"a" * 24: "s_e0", "b" * 24: "s_e1"}
    left = CandidateAliases("r0", {"evidence": [{"evidence_id": "a" * 24}]}, shared_ids=shared)
    right = CandidateAliases("r1", {"evidence": [{"evidence_id": "b" * 24}]}, shared_ids=shared)
    assert left.decode({"evidence_ids": ["s_e0"]}) == {"evidence_ids": ["a" * 24]}
    assert right.decode({"evidence_ids": ["s_e1"]}) == {"evidence_ids": ["b" * 24]}
    with pytest.raises(ValueError, match="foreign"):
        left.decode({"evidence_ids": ["s_e1"]})
