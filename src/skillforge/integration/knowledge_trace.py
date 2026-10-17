"""Bounded local decisions, separate from repository knowledge and model inputs."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from skillforge.filesystem_safety import is_link, no_symlink

LOGGER = logging.getLogger(__name__)
MAX_EVENTS = 192
MAX_TRACE_BYTES = 64 * 1024
TRACE_SLOTS = 8
# Reuse the existing tool-artifact boundary already excluded by patch collection.
TRACE_DIRECTORY = ".skillforge/productization/prepare/knowledge-trace"


class KnowledgeTrace:
    """Record identifiers and decisions, never source, prompts or model responses."""

    def __init__(self, repository_id: str):
        self.run_id = uuid4().hex
        self.repository_id = repository_id[:256]
        self.started_at = datetime.now(UTC).isoformat()
        self.started = time.perf_counter()
        self.events: list[dict] = []
        self.dropped_events = 0
        self.summary: dict = {}
        self.progress: dict = {}

    def record(self, stage, reason, *, object_id=None, path=None, line=None,
               evidence_ids=(), input_bytes=None, output_bytes=None, count=None):
        self.progress = {"stage": stage, "reason": reason,
                         "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3)}
        if len(self.events) >= MAX_EVENTS:
            self.dropped_events += 1
            return
        row = {"stage": stage, "reason": reason,
               "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3)}
        for key, value in (("object_id", object_id), ("path", path), ("line", line),
                           ("input_bytes", input_bytes), ("output_bytes", output_bytes),
                           ("count", count)):
            if value is not None:
                row[key] = value[:256] if isinstance(value, str) else value
        if evidence_ids:
            row["evidence_ids"] = list(evidence_ids[:32])
        self.events.append(row)

    def finish(self, operation, outcome, *, object_id=None, output_bytes=0):
        # The final outcome remains visible even when detailed events hit the cap.
        self.summary = {"operation": operation, "outcome": outcome,
                        "object_id": object_id, "output_bytes": output_bytes,
                        "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3)}

    def persist(self, repository_root: Path | None):
        if repository_root is None:
            return
        temporary = None
        try:
            root = Path(repository_root).absolute()
            if any(is_link(path) for path in (root, *root.parents)):
                raise ValueError("linked repository ancestor")
            if not root.is_dir():
                return
            directory = root / TRACE_DIRECTORY
            no_symlink(directory, root)
            directory.mkdir(parents=True, exist_ok=True)
            no_symlink(directory, root)
            target = directory / f"trace-{int(self.run_id, 16) % TRACE_SLOTS}.json"
            no_symlink(target, root)
            payload = {"schema": "skillforge-knowledge-trace-v1", "run_id": self.run_id,
                       "repository_id": self.repository_id, "started_at": self.started_at,
                       "events": list(self.events), "dropped_events": self.dropped_events,
                       "progress": self.progress,
                       "summary": self.summary}
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            while len(raw) > MAX_TRACE_BYTES and payload["events"]:
                payload["events"].pop()
                payload["dropped_events"] += 1
                raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            # Fixed slots bound persistent history; the unique temp prevents partial JSON.
            temporary = directory / f".{self.run_id}.tmp"
            with temporary.open("xb") as stream:
                stream.write(raw)
            temporary.replace(target)
        except (OSError, ValueError) as exc:
            # Diagnostic failure must not turn a successful knowledge request into failure.
            LOGGER.warning("knowledge trace unavailable (%s)", type(exc).__name__)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def record(trace: KnowledgeTrace | None, stage, reason, **fields):
    if trace is not None:
        trace.record(stage, reason, **fields)


@contextmanager
def trace_operation(root, trace, repository_id, operation):
    trace = trace or KnowledgeTrace(repository_id)
    trace.record(operation, "STARTED")
    try:
        yield trace
    except Exception as exc:
        trace.finish(operation, "FAILED", object_id=type(exc).__name__)
        raise
    finally:
        trace.persist(root)
