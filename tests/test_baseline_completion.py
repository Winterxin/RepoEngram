import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"evals"))
from u07_finish_baseline import completion_plan
from skillforge.cost_control import BudgetExceeded


def test_plan_keeps_cumulative_cost_and_forbids_new_semantics():
    plan=completion_plan({"reason":"warm fast path slower than observed cold analysis stage","wall_seconds":199.8277692},
        {"semantic_calls":1,"semantic_seconds":63.9745909,"total_prepare_seconds":181.9998692})
    assert plan["max_new_semantic_calls"] == 0
    assert plan["execution_remaining_seconds"]+199.8277692 == 600
    assert plan["prepare_remaining_seconds"]+181.9998692 == 349.862687


@pytest.mark.parametrize("seconds,calls",[(165,1),(64,0),(64,2)])
def test_no_budget_reset_or_unverified_semantic_replay(seconds,calls):
    with pytest.raises(BudgetExceeded):
        completion_plan({"reason":"warm fast path slower than observed cold analysis stage","wall_seconds":200},
            {"semantic_calls":calls,"semantic_seconds":seconds,"total_prepare_seconds":182})
