"""Focused retrieval remains local, bounded and compatible with unknown tasks."""

import json

from test_source_rule_distillation import FakeModel, _export, _route, _scope, _write

from skillforge.integration.knowledge_trace import KnowledgeTrace
from skillforge.integration.source_rule_distillation import _chunks


def test_explicit_entrypoint_exports_only_its_project_context_without_task_leak(tmp_path):
    _write(tmp_path, "src/policy.py",
           "def format_total(value):\n    return round(value)\n\n"
           "def ensure_open(state):\n    if state != 'open': raise ValueError(state)\n\n"
           "def reserve_seat(state):\n    ensure_open(state)\n    return True\n")
    task = "Fix reserve_seat and preserve its current constraints PRIVATE_TASK_CANARY"
    model = FakeModel()
    plan = _export(tmp_path, model, route=_route(task=task))
    assert plan is not None
    request = model.calls[0]
    assert {row["symbol"] for row in request["sources"]} == {"reserve_seat", "ensure_open"}
    assert request["sources"][0]["symbol"] == "reserve_seat"
    assert "PRIVATE_TASK_CANARY" not in json.dumps(request)


def test_unmatched_task_preserves_exact_previous_selection(tmp_path):
    _write(tmp_path, "src/a.py", "def calculate_total(amount):\n    return amount * 2\n")
    scope = _scope()
    baseline = _chunks(tmp_path, scope)
    assert _chunks(tmp_path, scope, focus_task="处理订单边界，不改变行为") == baseline
    assert _chunks(tmp_path, scope, focus_task="Fix nonexistent_entrypoint") == baseline


def test_unfit_anchor_falls_back_instead_of_exporting_only_its_helper(tmp_path):
    # The legacy per-file span allowance cannot hold this whole entrypoint.
    body = "    data.append(value)\n" * 3000
    _write(tmp_path, "src/a.py",
           "def helper(value):\n    return value\n\n"
           "def requested_entrypoint(value):\n    helper(value)\n    data = []\n"
           + body + "    return data\n")
    trace = KnowledgeTrace("repo")
    baseline = _chunks(tmp_path, _scope())
    focused = _chunks(tmp_path, _scope(), focus_task="Fix requested_entrypoint", trace=trace)
    assert focused == baseline
    assert any(event["reason"] == "ANCHOR_EXCEEDS_BUDGET" for event in trace.events)


def test_focused_source_stays_below_previous_raw_and_serialized_allowances(tmp_path):
    for index in range(5):
        _write(tmp_path, f"src/mod{index}.py",
               f"def unrelated_{index}(value):\n    if value: return value\n    return None\n")
    _write(tmp_path, "src/target.py",
           "def apply_state(value):\n    if value != 'ready': raise ValueError()\n"
           "    return value\n")
    baseline, _ = _chunks(tmp_path, _scope())
    focused, _ = _chunks(tmp_path, _scope(), focus_task="Change apply_state")
    assert {row["symbol"] for row in focused} == {"apply_state"}
    assert sum(len(row["text"].encode()) for row in focused) <= sum(
        len(row["text"].encode()) for row in baseline
    )
    assert len(json.dumps(focused, ensure_ascii=True).encode()) <= len(
        json.dumps(baseline, ensure_ascii=True).encode()
    )
