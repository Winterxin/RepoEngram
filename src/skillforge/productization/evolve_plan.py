"""Connect frozen prepare inputs to deterministic incremental planning."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from skillforge.domain.feedback import HumanFeedbackLedger
from skillforge.feedback.artifacts import LEDGER_PATH, load_feedback_ledger
from skillforge.filesystem_safety import is_link
from skillforge.productization.evolution_artifacts import load_baseline, save_plan
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.identity import repository_identity
from skillforge.repository.incremental_evolution import IncrementalEvolution
from skillforge.repository.repo_snapshot_semantic_cache import _production_version
from skillforge.repository.static_analysis import PythonStaticAnalyzer


@dataclass(frozen=True, slots=True)
class EvolvePlanResult:
    summary: dict[str, object]


def _applicable(ledger: HumanFeedbackLedger, snapshot: str
                ) -> tuple[HumanFeedbackLedger | None, int, int, int]:
    active = ledger.active_records
    selected = {item.feedback_id for item in active
                if item.source_snapshot_id == snapshot}
    if not selected:
        return None, len(active), 0, len(active)
    by_id = {item.feedback_id: item for item in ledger.records}
    pending = list(selected)
    while pending:
        parent = by_id[pending.pop()].supersedes_feedback_id
        if parent is not None and parent not in selected:
            selected.add(parent)
            pending.append(parent)
    view = HumanFeedbackLedger(ledger.repository_id,
                               tuple(by_id[key] for key in sorted(selected)))
    if any(item.source_snapshot_id != snapshot for item in view.active_records):
        raise ValueError("applicable feedback view contains stale active feedback")
    return view, len(active), len(view.active_records), len(active) - len(view.active_records)


def evolve_plan(repo_path: Path, baseline_path: Path) -> EvolvePlanResult:
    requested = repo_path.expanduser().absolute()
    if any(is_link(path) for path in (requested, *requested.parents)):
        raise ValueError("Repository path must be a real directory")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")
    baseline = load_baseline(root, baseline_path)
    if (
        baseline.production_knowledge is not None
        and baseline.production_knowledge.production_version != _production_version()
    ):
        raise ValueError("Production version differs; explicitly run full prepare")
    if baseline.repository_id != repository_identity(root):
        raise ValueError("Evolution baseline repository identity mismatch")
    head_result = subprocess.run(
        ("git", "-c", f"safe.directory={root}", "rev-parse", "--verify", "HEAD^{commit}"),
        cwd=root, capture_output=True, text=True, timeout=10, check=False,
    )
    if head_result.returncode:
        raise ValueError("Repository has no committed Git HEAD")
    head = head_result.stdout.strip()
    delta = GitHistoryMiner().mine_delta(root, baseline.head_commit, head)
    if delta.changesets and baseline.production_knowledge is None:
        raise ValueError(
            "Legacy baseline lacks production request audits; explicitly run full prepare"
        )
    static = PythonStaticAnalyzer().analyze(root)
    evolution = IncrementalEvolution()
    target = evolution.advance(baseline.source_state, delta, static, repo_path=root)
    sources: dict[str, str] = {}
    if is_link(root / LEDGER_PATH):
        raise ValueError("Feedback ledger path must not be a symlink")
    if (root / LEDGER_PATH).exists():
        ledger, sources = load_feedback_ledger(root)
        if ledger.repository_id != baseline.repository_id:
            raise ValueError("Feedback ledger repository identity mismatch")
        feedback, total, applicable, stale = _applicable(
            ledger, baseline.source_dna.snapshot_id,
        )
    else:
        feedback, total, applicable, stale = None, 0, 0, 0
    plan = evolution.plan(
        baseline.source_dna,
        baseline.source_compilation,
        baseline.source_state,
        target,
        delta,
        feedback,
        baseline.semantic_structural,
        source_knowledge=baseline.production_knowledge,
        static_analysis=static,
    )
    path = save_plan(root, baseline, target, delta, plan, feedback, sources)
    return EvolvePlanResult(
        {
            "repository": str(root),
            "baseline": str(baseline_path),
            "baseline_id": baseline.baseline_id,
            "source_snapshot_id": baseline.source_dna.snapshot_id,
            "source_head_commit": baseline.head_commit,
            "target_head_commit": head,
            "new_commits": len(delta.changesets),
            "refresh_decisions": (
                [
                    {
                        "channel": item.channel,
                        "candidate_id": item.candidate_id,
                        "reason": item.reason,
                    }
                    for item in plan.production.decisions
                ]
                if plan.production
                else []
            ),
            "workflow_candidates": {
                "changed": len(plan.changed_workflow_candidate_ids),
                "new": len(plan.new_workflow_candidate_ids),
                "missing": len(plan.missing_workflow_candidate_ids),
            },
            "anti_pattern_candidates": {
                "changed": len(plan.changed_antipattern_candidate_ids),
                "new": len(plan.new_antipattern_candidate_ids),
                "missing": len(plan.missing_antipattern_candidate_ids),
            },
            "reviews": {
                "workflow": len(plan.diagnostic_workflow_candidate_ids),
                "anti_pattern": len(plan.diagnostic_antipattern_candidate_ids),
                "rules": len(plan.rule_review_ids),
                "routing_feedback": len(plan.routing_review_feedback_ids),
                "verify_feedback": len(plan.verify_review_feedback_ids),
                "unresolved_feedback": len(plan.unresolved_feedback_ids),
            },
            "feedback_total_active_count": total,
            "feedback_applicable_count": applicable,
            "feedback_ignored_stale_count": stale,
            "plan_id": plan.plan_id,
            "plan_artifact": path,
        }
    )
