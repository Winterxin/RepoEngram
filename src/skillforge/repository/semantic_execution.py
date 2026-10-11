"""Budget native transports without changing their cache/permission identity."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from skillforge.cost_control import BudgetExceeded, CostLedger, SemanticBudget, digest


def _execute_native(command, kwargs, timeout):
    # A timeout of subprocess.run kills only the immediate process. Native Codex
    # may spawn helpers; own the complete family before dispatching inference.
    process = subprocess.Popen(
        command,
        cwd=kwargs["cwd"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(kwargs["input"].encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                killed = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=2,
                )
                if killed.returncode and process.poll() is None:
                    raise BudgetExceeded("semantic process family cleanup unconfirmed") from None
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=1)
        except (OSError, subprocess.SubprocessError) as error:
            raise BudgetExceeded("semantic process family cleanup unconfirmed") from error
        raise BudgetExceeded("semantic deadline reached: INVALID_BUDGET") from None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
    )


def run_semantic(model, command, **kwargs):
    """Custom transports retain their independently declared accounting contract."""
    budget = getattr(model, "ordinary_budget", None)
    if budget is None:
        return model._runner(command, **kwargs)
    budget.check()
    remaining = min(
        budget.semantic_limit - budget.semantic_seconds,
        budget.prepare_limit - (budget.clock() - budget.started),
    )
    if remaining <= 3:
        budget.abort("insufficient cleanup allowance BEFORE semantic spawn")
    ledger = model.cost_ledger
    command = list(command)
    command.insert(2, "--json")
    schema = Path(command[command.index("--output-schema") + 1]).read_bytes()
    wire_bytes = len(kwargs["input"].encode("utf-8")) + len(schema)
    remaining = budget.reserve(input_bytes=wire_bytes)
    logs = None
    if ledger.path:
        logs = ledger.path.parent / "transports" / str(time.monotonic_ns())
        logs.mkdir(parents=True)
        (logs / "prompt.txt").write_text(kwargs["input"], "utf-8")
        (logs / "response-schema.json").write_bytes(schema)
    started = time.monotonic()
    with ledger.measure(
        "semantic",
        kwargs["input"],
        cache="miss",
        bypass_reason=None,
        real_call=budget.calls,
        argv_sha256=digest(command),
        schema_sha256=hashlib.sha256(schema).hexdigest(),
        input_bytes=len(kwargs["input"].encode("utf-8")),
        wire_bytes=wire_bytes, cumulative_wire_bytes=budget.input_bytes,
        input_sha256=hashlib.sha256(kwargs["input"].encode("utf-8")).hexdigest(),
    ) as event:
        try:
            result = _execute_native(command, kwargs, min(kwargs["timeout"], remaining - 3))
            if logs:
                (logs / "stdout.jsonl").write_text(result.stdout or "", "utf-8")
                (logs / "stderr.txt").write_text(result.stderr or "", "utf-8")
                response = Path(command[command.index("-o") + 1])
                if response.is_file():
                    raw = response.read_bytes()
                    (logs / "response.json").write_bytes(raw)
                    event.update(
                        response_bytes=len(raw), response_sha256=hashlib.sha256(raw).hexdigest()
                    )
                event["logs"] = str(logs)
            usage = []
            for line in (result.stdout or "").splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and item.get("type") == "turn.completed":
                    usage.append(item.get("usage"))
            event.update(usage=usage or None, exit_code=result.returncode)
            return result
        except BaseException as error:
            event.update(completion="NOT_COMPLETED", usage=None)
            if isinstance(error, BudgetExceeded):
                budget.failed = str(error)
            raise
        finally:
            budget.finish(time.monotonic() - started)


@contextmanager
def ordinary_governance(models, root, ledger=None, *, cold_shape=None, max_input_bytes=None):
    """One invocation budget for native channels, batches, singles and repairs.

    Explicit custom transports remain responsible for their own budgets. Native
    runtime attributes are restored so API clients can reuse a preparer instance.
    """
    native = [m for m in models if getattr(m, "_runner", None) is subprocess.run]
    from skillforge.filesystem_safety import no_symlink

    no_symlink(root / ".skillforge/productization/prepare", root)
    ledger = ledger or CostLedger(root / ".skillforge/productization/prepare/cost-events.jsonl")
    budget = None
    if native:
        budget = (SemanticBudget() if cold_shape is None and max_input_bytes is None else
                  SemanticBudget(cold_shape=cold_shape, max_input_bytes=max_input_bytes))
    saved = []
    names = ("ordinary_budget", "effect_budget", "cost_ledger", "semantic_cache_required")
    try:
        for model in native:
            if getattr(model, "effect_budget", None) is not None:
                raise BudgetExceeded("native adapter already governed; ambiguous budget ownership")
            saved.append(
                (
                    model,
                    {name: (hasattr(model, name), getattr(model, name, None)) for name in names},
                )
            )
            model.ordinary_budget = model.effect_budget = budget
            model.cost_ledger = ledger
            model.semantic_cache_required = True
        if budget:
            ledger.append({"kind": "budget-plan", "plan": budget.plan()})
        yield budget, ledger
        if budget:
            budget.check()
    except BudgetExceeded as error:
        if budget:
            ledger.append(
                {
                    "kind": "budget-stop",
                    "classification": "INVALID_BUDGET",
                    "reason": str(error),
                    "real_semantic_calls": budget.calls,
                    "semantic_seconds": budget.semantic_seconds,
                    "prepare_seconds": budget.clock() - budget.started,
                }
            )
        raise
    finally:
        for model, attributes in saved:
            for name, (exists, value) in attributes.items():
                if exists:
                    setattr(model, name, value)
                else:
                    delattr(model, name)
