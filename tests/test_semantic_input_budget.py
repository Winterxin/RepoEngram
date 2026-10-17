"""Cumulative prompt/schema accounting fails before inference, including repairs."""
import pytest

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget
from skillforge.repository import semantic_execution


def test_cumulative_input_stops_at_boundary_without_spending_next_call():
    budget = SemanticBudget(max_input_bytes=100)
    budget.reserve(input_bytes=60)
    budget.reserve(input_bytes=39)
    with pytest.raises(BudgetExceeded, match="BEFORE next spawn"):
        budget.reserve(input_bytes=1)
    assert (budget.calls, budget.input_bytes) == (2, 99)
    with pytest.raises(BudgetExceeded):
        budget.reserve(input_bytes=0)


def test_budgeted_transport_cannot_omit_accounting():
    budget = SemanticBudget(max_input_bytes=410000)
    with pytest.raises(BudgetExceeded, match="missing"):
        budget.reserve()
    assert (budget.calls, budget.input_bytes) == (0, 0)


@pytest.mark.parametrize("limit", [9, 10])
def test_native_schema_and_utf8_count_before_process_spawn(tmp_path, monkeypatch, limit):
    schema = tmp_path / "schema.json"
    schema.write_bytes(b"1234567")
    class Model:
        ordinary_budget = SemanticBudget(max_input_bytes=limit)
        cost_ledger = CostLedger()
    def forbidden(*args):
        raise AssertionError("must not start any native process")
    monkeypatch.setattr(semantic_execution, "_execute_native", forbidden)
    with pytest.raises(BudgetExceeded, match="input byte budget"):
        semantic_execution.run_semantic(Model(), ["codex", "exec", "--output-schema", str(schema)],
                                        input="中", timeout=100)
    assert Model.ordinary_budget.calls == 0


def test_invalid_limit_rejected():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            SemanticBudget(max_input_bytes=value)
