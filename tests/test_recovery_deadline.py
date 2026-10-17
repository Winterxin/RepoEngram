import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import u07_cost_control as campaign


def test_deadline_reads_private_plan_without_changing_environment(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(campaign, "RECOVER_PAID", True)
    monkeypatch.setattr(campaign.recovery, "CAMPAIGN", tmp_path)
    (tmp_path / "prepare-deadline-plan.json").write_text(json.dumps({"prepare_seconds": 110.0}))
    before = dict(os.environ)
    budget = campaign.campaign_budget()
    assert budget.max_calls == 2 and budget.prepare_limit == 110.0
    assert budget.semantic_limit == 164.720042
    assert dict(os.environ) == before


def test_private_plan_cannot_raise_production_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "RECOVER_PAID", True)
    monkeypatch.setattr(campaign.recovery, "CAMPAIGN", tmp_path)
    (tmp_path / "prepare-deadline-plan.json").write_text(json.dumps({"prepare_seconds": 500.0}))
    assert campaign.campaign_budget().prepare_limit == 349.862687
