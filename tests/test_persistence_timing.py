import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from persistence_timing import PersistenceTimer


def test_nested_persistence_counted_once_and_exception_restores_depth():
    timer = PersistenceTimer()
    def _write_json():
        time.sleep(.01)
        raise ValueError("expected fixture failure")
    def save_baseline():
        _write_json()
    previous = sys.getprofile()
    try:
        sys.setprofile(timer.profile)
        try: save_baseline()
        except ValueError: pass
    finally:
        sys.setprofile(previous)
    assert timer.stats["depth"] == 0 and not timer.matched
    assert .01 <= timer.stats["seconds"] < .2


def test_unrelated_events_do_not_read_frame_locals():
    class Code: co_name = "unrelated_function"
    class Frame:
        f_code = Code()
        @property
        def f_locals(self): raise AssertionError("unrelated locals materialized")
    timer = PersistenceTimer()
    for event in ("call", "return", "c_call", "c_return", "c_exception"):
        timer.profile(Frame(), event, None)
    assert timer.stats["seconds"] == 0
