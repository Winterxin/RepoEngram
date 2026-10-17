"""Bound cost while retaining cross-module evidence and scope identity."""
from dataclasses import replace

import pytest
import test_semantic_workflows as workflow_fixtures
from test_incremental_evolution_v2 import rich_repo as rich_repo

from skillforge.repository.bounded_candidate_selection import (
    bounded_structural,
    scoped_candidates,
    select_structural_candidates,
    selection_rows,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


def test_structural_budget_preserves_method_diversity(rich_repo):
    from types import SimpleNamespace
    facts = PythonStaticAnalyzer().analyze(rich_repo)
    mined = StructuralRuleMiner().mine(facts)
    methods = {item.method for item in mined.candidates}
    assert len(methods) == 3
    selected = select_structural_candidates(mined, limit=3)
    assert {item.method for item in selected} == methods
    reversed_result = SimpleNamespace(candidates=tuple(reversed(mined.candidates)))
    assert select_structural_candidates(reversed_result, limit=3) == selected


@pytest.mark.parametrize("package", ["store", "reports", "framework"])
def test_limits_and_scoped_selection_across_unrelated_packages(tmp_path, package):
    for n in range(16):
        folder = tmp_path / "src" / package / f"layer{n}"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("")
        for index, name in enumerate(["one", "two", "three"]):
            (folder / f"{name}.py").write_text(
                f"from {package}.shared.model{index % 2} import Shared\n"
            )
    (tmp_path / "src" / package / "__init__.py").write_text("")
    shared = tmp_path / "src" / package / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    for index in range(2):
        (shared / f"model{index}.py").write_text("class Shared: pass\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    mined = StructuralRuleMiner().mine(facts)
    assert len(mined.candidates) == 16
    general = bounded_structural(mined, facts, None, (), ())
    assert len(general.candidates) == 11
    assert bounded_structural(mined, facts, None, (), ()) == general
    full = bounded_structural(mined, facts, None, (None,) * 5, (None,) * 5)
    assert len(full.candidates) == 1  # All three channels together remain <=11.
    scoped = bounded_structural(mined, facts, f"{package}.layer2.one", (), ())
    assert len(scoped.candidates) == 1
    assert scoped.candidates[0].source_modules == (
        f"{package}.layer2.one", f"{package}.layer2.three", f"{package}.layer2.two")
    assert scoped.candidates[0].rule.evidence_ids
    unrelated = bounded_structural(mined, facts, f"{package}.layer2_extra", (), ())
    assert unrelated.candidates == ()
    rows = selection_rows(mined, scoped.candidates, facts, f"{package}.layer2.one", "structural")
    assert len(rows) == 16 and sum(row["selected"] for row in rows) == 1
    assert sum(row["reason"] == "outside declared module scope" for row in rows) == 15


def test_workflow_scope_keeps_complete_companion_candidate():
    _, facts, candidates = workflow_fixtures._fixture()
    item = next(c for c in candidates.candidates if c.current_modules)
    selected = scoped_candidates(candidates, facts, item.current_modules[0])
    assert item in selected.candidates
    preserved = next(c for c in selected.candidates if c.candidate_id == item.candidate_id)
    assert preserved.members == item.members
    assert scoped_candidates(candidates, facts) is candidates
    assert scoped_candidates(candidates, facts, "absent_package").candidates == ()


def test_scoped_history_identity_round_trips_without_relabeling_baselines(tmp_path):
    from test_incremental_evolution import setup

    from skillforge.domain.evolution import EvolutionHistoryState
    from skillforge.productization.evolution_artifacts import _restore
    from skillforge.repository.git_history import GitHistoryMiner
    from skillforge.repository.incremental_evolution import IncrementalEvolution, production_inputs
    head = setup(tmp_path)
    history = GitHistoryMiner().mine(tmp_path, 4)
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    engine = IncrementalEvolution()
    plain = engine.make_state(history, head, 4, facts)
    scoped = replace(plain, current_module="absent_package")
    assert plain.state_id != scoped.state_id
    assert production_inputs(scoped, facts) == ()
    assert _restore(EvolutionHistoryState, scoped.to_dict()) == scoped
    assert "current_module" not in plain.to_dict()
    assert _restore(EvolutionHistoryState, plain.to_dict()) == plain


def test_antipattern_scope_preserves_signal_and_evidence_closure():
    from types import SimpleNamespace

    from test_semantic_antipatterns import _candidate_result, _history
    history = _history(
        {"message": "Fix parser bug", "paths": ("pkg/a.py",)},
        {"message": "Fix reader bug", "paths": ("pkg/b.py",)},
    )
    candidates = _candidate_result(history)
    assert len(candidates.candidates) == 2
    facts = SimpleNamespace(modules=tuple(
        SimpleNamespace(path=f"pkg/{name}.py", module=f"pkg.{name}") for name in ("a", "b")))
    selected = scoped_candidates(candidates, facts, "pkg.a")
    assert len(selected.candidates) == 1
    assert selected.candidates[0] in candidates.candidates
    assert {s.signal_id for s in selected.signals} == set(selected.candidates[0].signal_ids)
    assert {e.evidence_id for e in selected.evidence} == set(selected.candidates[0].evidence_ids)
    assert scoped_candidates(candidates, facts, "missing").signals == ()
