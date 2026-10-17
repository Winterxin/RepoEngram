"""Bounded Evolution persistence for legitimate repository-scale artifacts."""

import json
from dataclasses import replace

import pytest
from test_evolution_productization import _prepare
from test_evolution_productization import repo as _repo_fixture

from skillforge.productization.evolution_artifacts import (
    _baseline_value,
    load_baseline,
    make_baseline,
    save_baseline,
)
from skillforge.verify.artifacts import _canonical_json, _write_json

MIB = 1_048_576


@pytest.fixture
def repo(tmp_path):
    return _repo_fixture.__wrapped__(tmp_path)


def _baseline(root, message_bytes=0):
    prepared = _prepare(root)
    source = load_baseline(root, prepared.evolution_baseline_path)
    # A legacy v1 history message is typed data, with no imposed string limit.
    changes = source.source_state.history.changesets
    history = replace(source.source_state.history, changesets=(
        replace(changes[0], message="x" * message_bytes), *changes[1:],
    ))
    state = replace(source.source_state, history=history)
    return make_baseline(
        root, source.head_commit, source.history_limit, source.source_dna,
        source.source_compilation, state, source.semantic_structural,
    )


def test_large_legal_baseline_round_trip_preserves_canonical_identity(repo):
    baseline = _baseline(repo, 17 * MIB)
    canonical = _canonical_json(_baseline_value(baseline))
    assert 16 * MIB < len(canonical) < 64 * MIB
    path = save_baseline(repo, baseline)
    loaded = load_baseline(repo, path)
    assert loaded == baseline
    assert loaded.baseline_id == baseline.baseline_id
    assert (repo / path).read_bytes() == canonical
    assert save_baseline(repo, loaded) == path
    assert (repo / path).read_bytes() == canonical


def test_existing_conflicting_baseline_is_not_overwritten(repo):
    baseline = _baseline(repo)
    path = save_baseline(repo, baseline)
    value = _baseline_value(baseline)
    value["head_commit"] = "tampered"
    conflicting = _canonical_json(value)
    (repo / path).write_bytes(conflicting)
    with pytest.raises(ValueError, match="conflicts"):
        save_baseline(repo, baseline)
    assert (repo / path).read_bytes() == conflicting


def test_over_64_mib_baseline_fails_closed_on_save_and_load(repo):
    baseline = _baseline(repo, 64 * MIB)
    with pytest.raises(ValueError, match="size limit"):
        save_baseline(repo, baseline)
    path = repo / ".skillforge/productization/evolution/baselines/oversized.json"
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.seek(64 * MIB)
        stream.write(b"\n")
    with pytest.raises(ValueError, match="size limit"):
        load_baseline(repo, path)


@pytest.mark.parametrize("corruption", ["malformed", "tampered", "unknown_field"])
def test_small_v1_baseline_still_rejects_corruption(repo, corruption):
    baseline = _baseline(repo)
    path = save_baseline(repo, baseline)
    value = _baseline_value(baseline)
    assert "production_knowledge" not in value
    assert load_baseline(repo, path) == baseline
    assert (repo / path).read_bytes() == _canonical_json(value)
    if corruption == "malformed":
        (repo / path).write_text("{", encoding="utf-8")
    else:
        value["baseline_id" if corruption == "tampered" else "extra"] = "tampered"
        (repo / path).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(repo, path)


def test_generic_artifact_limit_does_not_expand(repo):
    from pathlib import Path

    with pytest.raises(ValueError, match="size limit"):
        _write_json(repo, Path(".skillforge/oversized-generic.json"), "x" * MIB)
