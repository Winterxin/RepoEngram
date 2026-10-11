"""Explicit economic gates. Budget exceptions must not become candidate failures."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(Exception):
    """Fatal orchestration failure, outside semantic adapter retry exceptions."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class CostLedger:
    def __init__(self, path: Path | None = None):
        self.path, self.events = path, []
        if path and path.exists():
            self.events = [json.loads(line) for line in path.read_text("utf-8").splitlines()]

    def append(self, event):
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    @contextmanager
    def measure(self, kind, payload, **metadata):
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        event = {"kind": kind, "counter": 1 + sum(e["kind"] == kind for e in self.events),
                 "start_ns": time.monotonic_ns(), "payload_sha256": hashlib.sha256(data).hexdigest(),
                 "payload_bytes": len(data), "usage": None, **metadata}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".starts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**event, "phase": "START"}, ensure_ascii=False) + "\n")
        try:
            yield event
        except BaseException as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            event["end_ns"] = time.monotonic_ns()
            event["seconds"] = (event["end_ns"] - event["start_ns"]) / 1e9
            self.append(event)


class SemanticBudget:
    def __init__(self, *, max_calls=5, semantic_seconds=164.720042,
                 prepare_seconds=349.862687, clock=time.monotonic, cold_shape=None,
                 max_input_bytes=None):
        if type(max_calls) is not int or max_calls < 0 or min(semantic_seconds, prepare_seconds) <= 0:
            raise ValueError("invalid budget")
        self.max_calls, self.semantic_limit, self.prepare_limit = max_calls, semantic_seconds, prepare_seconds
        self.clock, self.started = clock, clock()
        self.calls, self.semantic_seconds, self.failed = 0, 0.0, None
        if max_input_bytes is not None and (type(max_input_bytes) is not int or max_input_bytes <= 0):
            raise ValueError("invalid cumulative semantic input byte budget")
        self.max_input_bytes, self.input_bytes = max_input_bytes, 0
        if cold_shape is not None and (set(cold_shape) != {"workflow", "tendency", "anti-pattern", "structural"}
                or any(type(n) is not int or n < 0 for n in cold_shape.values())):
            raise ValueError("invalid declared cold shape")
        self.cold_shape = dict(cold_shape) if cold_shape is not None else None

    def abort(self, reason):
        self.failed = reason
        raise BudgetExceeded(reason)

    def check(self):
        if self.failed:
            self.abort(self.failed)
        if self.clock() - self.started >= self.prepare_limit:
            self.abort("prepare wall budget exhausted")
        if self.semantic_seconds >= self.semantic_limit:
            self.abort("semantic cumulative wall budget exhausted")

    def reserve(self, *, input_bytes=None):
        self.check()
        if self.calls >= self.max_calls:
            self.abort("real semantic call budget exhausted BEFORE next spawn")
        if input_bytes is not None and (type(input_bytes) is not int or input_bytes < 0):
            self.abort("invalid semantic input byte accounting BEFORE next spawn")
        if self.max_input_bytes is not None:
            if input_bytes is None:
                self.abort("missing semantic input byte accounting BEFORE next spawn")
            if self.input_bytes + input_bytes >= self.max_input_bytes:
                self.abort("cumulative semantic input byte budget exhausted BEFORE next spawn")
        self.input_bytes += input_bytes or 0
        self.calls += 1
        return min(self.semantic_limit - self.semantic_seconds,
                   self.prepare_limit - (self.clock() - self.started))

    def validate_shape(self, counts):
        self.check()
        if self.cold_shape is not None:
            if counts != self.cold_shape:
                self.abort("cold candidate shape differs from validated budget plan")
            return
        if sum(counts.values()) > 11 or sum(bool(n) for n in counts.values()) > 3:
            self.abort("cold candidate shape outside validated 3-channel/11-candidate budget")

    def finish(self, seconds):
        self.semantic_seconds += seconds
        self.check()

    def plan(self):
        return {"max_real_semantic_calls": self.max_calls, "semantic_seconds": self.semantic_limit,
                "prepare_seconds": self.prepare_limit, "unexpected_bypass": "ABORT_BEFORE_MODEL",
                "baseline_reuse_calls": 0, "retry_budget": "shared with batches",
                "declared_cold_shape": self.cold_shape,
                "max_cumulative_input_bytes": self.max_input_bytes,
                "input_byte_scope": "UTF-8 prompt plus exact response schema, including repairs"}
