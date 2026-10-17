"""Interrupted spans remain honest, immutable and separate from completions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from cost_reconcile import reconcile_interrupted


def test_unclosed_semantic_span_unknown_usage_and_original_bytes_preserved(tmp_path):
    path = tmp_path / "events.jsonl"
    starts = path.with_suffix(".starts.jsonl")
    entries = [{"kind": "semantic", "start_ns": 100, "counter": 1},
               {"kind": "semantic", "start_ns": 200, "counter": 2},
               {"kind": "prepare", "start_ns": 250, "counter": 1}]
    starts.write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in entries) + b'{"partial":')
    path.write_text(json.dumps({**entries[0], "end_ns": 150}) + "\n", "utf-8")
    originals = starts.read_bytes(), path.read_bytes()
    events = reconcile_interrupted(path, {"end_ns": 400, "exit_code": 1})
    assert len(events) == 1
    event = events[0]
    assert event["start_ns"] == 200 and event["end_ns"] == 400
    assert event["usage"] is None and event["model_completion"] is False
    assert event["process_family_termination"] == "NOT_RECORDED"
    assert event["seconds"] == 200 / 1e9
    assert originals == (starts.read_bytes(), path.read_bytes())
    assert reconcile_interrupted(path, {"end_ns": 400, "exit_code": 1}) == events


def test_no_start_is_not_a_model_call(tmp_path):
    assert reconcile_interrupted(tmp_path / "events.jsonl", {"end_ns": 1, "exit_code": 0}) == []


def test_completed_call_does_not_gain_duplicate_interruption(tmp_path):
    path = tmp_path / "events.jsonl"
    start = {"kind": "semantic", "start_ns": 100}
    path.with_suffix(".starts.jsonl").write_text(json.dumps(start) + "\n", "utf-8")
    path.write_text(json.dumps({**start, "end_ns": 150}) + "\n", "utf-8")
    assert reconcile_interrupted(path, {"end_ns": 200, "exit_code": 0}) == []
