"""Task-independent baseline reuse, bound to the complete verified source snapshot."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from skillforge.cost_control import BudgetExceeded, digest
from skillforge.filesystem_safety import no_symlink
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.verify.artifacts import _read_repo_json, _write_json

_INDEX = Path(".skillforge/productization/prepare/verified-baseline.json")


@contextmanager
def _baseline_reservation(root):
    """Kernel ownership survives neither crashes nor watchdog process kills.

    Keep the sentinel inode: unlinking after unlock races with the next owner.
    A legacy sentinel is recoverable only after legacy workers have stopped.
    """
    lock = root / ".skillforge/productization/prepare/verified-baseline.lock"
    no_symlink(lock, root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as reservation:
        reservation.seek(0, os.SEEK_END)
        if reservation.tell() == 0:
            reservation.write(b"\0")
            reservation.flush()
        reservation.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(reservation.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(reservation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise BudgetExceeded(
                "baseline preparation already reserved; no duplicate model work"
            ) from error
        try:
            yield
        finally:
            reservation.seek(0)
            if os.name == "nt":
                msvcrt.locking(reservation.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(reservation.fileno(), fcntl.LOCK_UN)


class VerifiedPreparer:
    """Exact frozen reuse; ordinary entry can opt into audited committed evolution."""

    def __init__(
        self,
        preparer,
        models,
        *,
        config,
        budget=None,
        ledger,
        rebuild=False,
        allow_incremental=False,
    ):
        self.preparer, self.models, self.config = preparer, tuple(models), config
        self.budget, self.ledger = budget, ledger
        self.mode = None
        self.rebuild = rebuild
        self.allow_incremental = allow_incremental

    def _check_budget(self):
        if self.budget is not None:
            self.budget.check()

    def prepare(self, request, *, before_prepare=None):
        from skillforge.errors import InvalidRepositoryPath
        from skillforge.filesystem_safety import is_link

        requested = Path(request.path).expanduser().absolute()
        if not requested.is_dir() or any(is_link(p) for p in (requested, *requested.parents)):
            raise InvalidRepositoryPath(
                f"Repository path is not a directory or is a link: {requested}"
            )
        root = requested.resolve()
        with _baseline_reservation(root):
            if before_prepare is not None:
                before_prepare()
            return self._prepare(request)

    def _prepare(self, request):
        root = Path(request.path).resolve()
        cache = RepoSnapshotSemanticCache(root)
        if cache.context is None:
            raise BudgetExceeded("unverifiable repository snapshot: cache bypass")
        # The SAME adapters passed to the production preparer must be eligible.
        transports = []
        for model in self.models:
            wrapped = cache.wrap(model)
            if wrapped is model:
                raise BudgetExceeded("unexpected semantic cache bypass: " + type(model).__name__)
            transports.append(wrapped.config)
        identity = {"context": cache.context, "config": self.config, "transports": transports}
        with self.ledger.measure("prepare", {"identity": identity, "task": request.task}) as event:
            self._check_budget()
            if (root / _INDEX).exists() and not self.rebuild:
                _, _, entry = _read_repo_json(root, _INDEX, "verified baseline")
                if entry.get("checksum") != digest(entry.get("payload")):
                    raise BudgetExceeded("verified baseline manifest checksum mismatch")
                saved = entry["payload"]
                if saved["identity"] != identity:
                    if self.allow_incremental and (
                        self.budget is None
                        or all(
                            getattr(m, "ordinary_budget", None) is self.budget for m in self.models
                        )
                    ):
                        return self._prepare_incremental(
                            root, request, saved, identity, cache, event
                        )
                    raise BudgetExceeded(
                        "verified baseline snapshot/config drift; explicit cold rebuild required"
                    )
                baseline_path = root / saved["baseline"]
                baseline = load_baseline(root, saved["baseline"])
                if digest(baseline_path.read_text("utf-8")) != saved["baseline_sha256"]:
                    raise BudgetExceeded("verified baseline content drift")
                if baseline.head_commit != cache.context["snapshot"]["head"]:
                    raise BudgetExceeded("verified baseline HEAD mismatch")
                before = self.budget.calls if self.budget else 0

                # A warm path must not silently dispatch any model, including
                # ordinary CLI adapters that do not have a campaign counter.
                def forbidden(*args, **kwargs):
                    raise BudgetExceeded("verified baseline attempted model transport")

                guards = []
                for model in self.models:
                    names = (
                        ("_runner",)
                        if hasattr(model, "_runner")
                        else ("complete", "complete_batch")
                    )
                    for name in names:
                        original = getattr(model, name, None)
                        if callable(original):
                            own = not hasattr(model, "__dict__") or name in model.__dict__
                            guards.append((model, name, original, own))
                try:
                    for model, name, _, _ in guards:
                        setattr(model, name, forbidden)
                    result = self.preparer.prepare(
                        replace(request, baseline=Path(saved["baseline"])),
                        _loaded_baseline=baseline,
                    )
                finally:
                    for model, name, original, own in guards:
                        if own:
                            setattr(model, name, original)
                        elif name in getattr(model, "__dict__", {}):
                            delattr(model, name)
                if self.budget and (self.budget.calls != before or self.budget.calls != 0):
                    raise BudgetExceeded(
                        "verified baseline reused with nonzero real semantic calls"
                    )
                if digest(baseline_path.read_text("utf-8")) != saved["baseline_sha256"]:
                    raise BudgetExceeded("verified baseline changed during task preparation")
                # Missing/degraded decisions stay missing/degraded; no fabricated success.
                provenance = saved["provenance"]
                result = replace(
                    result,
                    **{
                        key: (
                            tuple(value)
                            if key in {"semantic_failures", "history_semantic_selection"}
                            else value
                        )
                        for key, value in provenance.items()
                    },
                )
                self.mode = "VERIFIED_BASELINE"
                event.update(cache="hit", bypass_reason=None, real_semantic_calls=0)
            else:
                result = self.preparer.prepare(replace(request, baseline=None))
                self.mode = "COLD_BATCH"
                if not result.evolution_baseline_path:
                    raise BudgetExceeded("no production baseline available")
                if RepoSnapshotSemanticCache(root).context != cache.context:
                    raise BudgetExceeded("source changed during preparation")
                self._check_budget()
                self._save_index(root, identity, result)
                event.update(
                    cache="miss",
                    bypass_reason=None,
                    real_semantic_calls=self.budget.calls if self.budget else None,
                )
            self._check_budget()
            if RepoSnapshotSemanticCache(root).context != cache.context:
                raise BudgetExceeded("source changed while preparing baseline/route")
            return result

    def _save_index(self, root, identity, result):
        keys = [
            "semantic_failure_count",
            "semantic_failures",
            "history_semantic_selection",
            "workflow_accepted_count",
            "workflow_rejection_count",
            "tendency_accepted_count",
            "tendency_rejection_count",
            "anti_pattern_accepted_count",
            "anti_pattern_rejection_count",
            "semantic_accepted_count",
            "semantic_rejection_count",
            "workflow_semantic_selected_count",
            "anti_pattern_semantic_selected_count",
            "structural_candidate_count",
            "workflow_semantic_retry_count",
            "anti_pattern_semantic_retry_count",
        ]
        serialized = result.to_dict()
        saved = {
            "identity": identity,
            "baseline": result.evolution_baseline_path,
            "baseline_sha256": digest((root / result.evolution_baseline_path).read_text("utf-8")),
            "provenance": {key: serialized[key] for key in keys},
        }
        _write_json(root, _INDEX, {"payload": saved, "checksum": digest(saved)})

    def _prepare_incremental(self, root, request, saved, identity, cache, event):
        old = saved["identity"]
        same_configuration = {k: v for k, v in old.items() if k != "context"} == {
            k: v for k, v in identity.items() if k != "context"
        }
        old_context, context = old["context"], identity["context"]
        same_context = {k: v for k, v in old_context.items() if k != "snapshot"} == {
            k: v for k, v in context.items() if k != "snapshot"
        }
        if (
            not same_configuration
            or not same_context
            or old_context["snapshot"]["head"] == context["snapshot"]["head"]
        ):
            raise BudgetExceeded(
                "verified baseline snapshot/config drift; explicit cold rebuild required"
            )
        # Evolution is HEAD-based. Uncommitted or untracked source cannot be
        # silently folded into a committed knowledge baseline.
        for args in (
            ("diff", "--name-only", "-z", "HEAD", "--"),
            ("ls-files", "-z", "--others", "--exclude-standard"),
        ):
            checked = subprocess.run(
                ("git", "-c", f"safe.directory={root}", *args),
                cwd=root,
                capture_output=True,
                timeout=10,
                check=True,
            )
            names = checked.stdout.split(b"\0")
            if any(
                name
                and Path(os.fsdecode(name)).parts[0]
                not in {".skillforge", ".codex", ".claude", ".agents"}
                for name in names
            ):
                raise BudgetExceeded(
                    "incremental prepare requires committed source; no model fallback"
                )
        baseline_path = root / saved["baseline"]
        baseline = load_baseline(root, saved["baseline"])
        if (
            digest(baseline_path.read_text("utf-8")) != saved["baseline_sha256"]
            or baseline.head_commit != old_context["snapshot"]["head"]
            or baseline.production_knowledge is None
        ):
            raise BudgetExceeded("incremental source baseline integrity/provenance mismatch")
        result = self.preparer.prepare(
            replace(request, baseline=Path(saved["baseline"])), _loaded_baseline=baseline
        )
        if (
            result.prepare_mode != "incremental"
            or result.evolution_head_commit != context["snapshot"]["head"]
        ):
            raise BudgetExceeded("incremental prepare did not produce the requested HEAD")
        if (
            digest(baseline_path.read_text("utf-8")) != saved["baseline_sha256"]
            or RepoSnapshotSemanticCache(root).context != cache.context
        ):
            raise BudgetExceeded("source changed during incremental preparation")
        self._check_budget()
        self._save_index(root, identity, result)
        self.mode = "INCREMENTAL"
        evaluations = result.evolution_statistics["semantic_calls_incremental"]
        event.update(
            cache="incremental",
            bypass_reason=None,
            real_semantic_calls=(
                self.budget.calls if self.budget else 0 if evaluations == 0 else None
            ),
            semantic_evaluation_count=evaluations,
        )
        return result


def automatic_prepare(
    request, *, preparer=None, ledger=None, rebuild=False, cold_shape=None, max_input_bytes=None
):
    """Reuse exact baselines or audited committed deltas; config drift is explicit."""
    from skillforge.productization.prepare import (
        ANTIPATTERN_SEMANTIC_LIMIT,
        HISTORY_LIMIT,
        WORKFLOW_SEMANTIC_LIMIT,
        SkillForgePreparer,
    )
    from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
    from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
    from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
    from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel

    preparer = preparer or SkillForgePreparer()
    from skillforge.errors import InvalidRepositoryPath, SkillForgeError
    from skillforge.filesystem_safety import is_link

    requested = Path(request.path).expanduser().absolute()
    if not requested.is_dir() or any(is_link(p) for p in (requested, *requested.parents)):
        raise InvalidRepositoryPath(f"Repository path is not a directory or is a link: {requested}")
    if request.baseline is not None and rebuild:
        raise SkillForgeError("--baseline and --rebuild are mutually exclusive")
    from skillforge.repository.git_metadata import read_git_metadata

    metadata = read_git_metadata(requested.resolve())
    attributes = (
        ("_semantic_model", CodexCliStructuralSemanticModel),
        ("_workflow_model", CodexCliSemanticModel),
        ("_anti_pattern_model", CodexCliAntiPatternSemanticModel),
        ("_tendency_model", CodexCliTendencySemanticModel),
    )
    models = []
    for name, kind in attributes:
        model = getattr(preparer, name) or kind()
        setattr(preparer, name, model)
        models.append(model)
    root = Path(request.path).resolve()
    from skillforge.repository.semantic_execution import ordinary_governance

    with ordinary_governance(
        models, root, ledger, cold_shape=cold_shape, max_input_bytes=max_input_bytes
    ) as (budget, ledger):
        if request.baseline is not None:
            # Explicit evolution still shares the invocation's native transport
            # budget. It is not an attested automatic warm-baseline hit.
            return preparer.prepare(request)
        if not metadata.is_git_repository or metadata.head_commit is None:
            # Static-only mode cannot attest an exact Git snapshot cache.
            for model in models:
                if getattr(model, "ordinary_budget", None) is budget and budget is not None:
                    model.semantic_cache_required = False
            return preparer.prepare(request)
        return VerifiedPreparer(
            preparer,
            models,
            ledger=ledger,
            budget=budget,
            rebuild=rebuild,
            allow_incremental=True,
            config={
                "history_limit": HISTORY_LIMIT,
                "workflow_limit": WORKFLOW_SEMANTIC_LIMIT,
                "anti_pattern_limit": ANTIPATTERN_SEMANTIC_LIMIT,
            },
        ).prepare(request)
