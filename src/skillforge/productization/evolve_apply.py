"""Apply a frozen evolution plan with bounded semantic refresh."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from skillforge.domain.evolution import fingerprint
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.filesystem_safety import is_link
from skillforge.productization.evolution_artifacts import (
    load_baseline,
    load_plan_context,
    make_baseline,
    save_baseline,
    save_result,
)
from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.repository.identity import repository_identity
from skillforge.repository.incremental_evolution import (
    IncrementalEvolution,
    merge_production_knowledge,
    production_inputs,
)
from skillforge.repository.repo_snapshot_semantic_cache import (
    RepoSnapshotSemanticCache,
    _production_version,
)
from skillforge.repository.semantic_antipatterns import SemanticAntiPatternAbstracter
from skillforge.repository.semantic_workflows import SemanticWorkflowAbstracter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


@dataclass(frozen=True, slots=True)
class EvolveApplyResult:
    summary: dict[str, object]


def evolve_apply(
    repo_path: Path,
    baseline_path: Path,
    plan_path: Path,
    *,
    workflow_model: object | None = None,
    antipattern_model: object | None = None,
    tendency_model: object | None = None,
    structural_model: object | None = None,
    semantic_cache_root: Path | None = None,
    allow_legacy_v1: bool = False,
) -> EvolveApplyResult:
    requested = repo_path.expanduser().absolute()
    if any(is_link(path) for path in (requested, *requested.parents)):
        raise ValueError("Repository path must be a real directory")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")
    baseline = load_baseline(root, baseline_path)
    loaded = load_plan_context(root, plan_path, baseline)
    plan = loaded.plan
    if baseline.repository_id != repository_identity(root):
        raise ValueError("Evolution baseline repository identity mismatch")
    head_result = subprocess.run(
        ("git", "-c", f"safe.directory={root}", "rev-parse", "--verify", "HEAD^{commit}"),
        cwd=root, capture_output=True, text=True, timeout=10, check=False,
    )
    if head_result.returncode or head_result.stdout.strip() != plan.target_head_commit:
        raise ValueError("Repository HEAD changed since evolve-plan; rerun evolve-plan.")

    evolution = IncrementalEvolution()
    static = PythonStaticAnalyzer().analyze(root)
    reconstructed = evolution.make_state(
        loaded.target_state.history, plan.target_head_commit,
        baseline.history_limit, static,
        current_module=loaded.target_state.current_module,
    )
    if reconstructed != loaded.target_state:
        raise ValueError("Repository static facts changed since evolve-plan; rerun evolve-plan.")

    if (
        plan.source_head_commit != plan.target_head_commit
        and plan.production is None
        and not allow_legacy_v1
    ):
        raise ValueError(
            "Legacy baseline lacks production request audits; explicitly run full prepare"
        )

    if plan.production is not None:
        return _apply_production(
            root,
            baseline_path,
            plan_path,
            baseline,
            loaded,
            static,
            workflow_model,
            antipattern_model,
            tendency_model,
            structural_model,
            semantic_cache_root,
        )

    selected_w = tuple(item for item in loaded.target_state.workflow_candidates.candidates
                       if item.candidate_id in plan.workflow_refresh_ids)
    selected_a = tuple(item for item in loaded.target_state.antipattern_candidates.candidates
                       if item.candidate_id in plan.antipattern_refresh_ids)
    try:
        if selected_w:
            semantic_w = SemanticWorkflowAbstracter().abstract(
                loaded.target_state.history, static, loaded.target_state.workflow_candidates,
                workflow_model if workflow_model is not None else CodexCliSemanticModel(),
                selected_candidates=selected_w,
            )
        else:
            semantic_w = SemanticWorkflowResult(baseline.repository_id, (), (), ())
        if selected_a:
            semantic_a = SemanticAntiPatternAbstracter().abstract(
                loaded.target_state.history, loaded.target_state.antipattern_candidates,
                antipattern_model if antipattern_model is not None
                else CodexCliAntiPatternSemanticModel(), selected_candidates=selected_a,
            )
        else:
            semantic_a = SemanticAntiPatternResult(baseline.repository_id, (), (), ())
    except Exception as error:
        raise ValueError(f"Evolution semantic refresh failed: {error}") from error

    result = evolution.apply_semantic(
        baseline.source_dna, baseline.source_compilation, loaded.target_state,
        plan, semantic_w, semantic_a, baseline.semantic_structural,
    )
    next_baseline = make_baseline(
        root, plan.target_head_commit, baseline.history_limit,
        result.resulting_project_dna, result.resulting_compilation,
        result.resulting_history_state, baseline.semantic_structural,
    )
    result_path = save_result(root, baseline, plan, semantic_w, semantic_a, result)
    next_path = save_baseline(root, next_baseline)
    return EvolveApplyResult({
        "repository": str(root), "baseline": str(baseline_path), "plan": str(plan_path),
        "baseline_id": baseline.baseline_id, "plan_id": plan.plan_id,
        "source_snapshot_id": result.source_snapshot_id,
        "result_snapshot_id": result.result_snapshot_id,
        "target_head_commit": result.target_head_commit,
        "semantic_refresh": {
            "workflow_candidates": len(plan.workflow_refresh_ids),
            "anti_pattern_candidates": len(plan.antipattern_refresh_ids),
        },
        "knowledge": {
            "workflows_refreshed": len(result.refreshed_workflow_ids),
            "workflows_added": len(result.added_workflow_ids),
            "workflows_unresolved": len(result.unresolved_workflow_ids),
            "anti_patterns_refreshed": len(result.refreshed_antipattern_ids),
            "anti_patterns_added": len(result.added_antipattern_ids),
            "anti_patterns_unresolved": len(result.unresolved_antipattern_ids),
        },
        "skills": {
            "added": len(result.skill_added_ids), "removed": len(result.skill_removed_ids),
            "version_changed": len(result.skill_version_changed_ids),
            "unchanged": len(result.skill_unchanged_ids),
        },
        "reviews_remaining": {
            "rules": len(plan.rule_review_ids),
            "routing_feedback": len(plan.routing_review_feedback_ids),
            "verify_feedback": len(plan.verify_review_feedback_ids),
            "unresolved_feedback": len(plan.unresolved_feedback_ids),
        },
        "result_id": result.result_id, "result_artifact": result_path,
        "next_baseline_id": next_baseline.baseline_id,
        "next_baseline": next_path,
    })


def _apply_production(
    root,
    baseline_path,
    plan_path,
    baseline,
    loaded,
    static,
    workflow_model,
    antipattern_model,
    tendency_model,
    structural_model,
    cache_root,
):
    production = loaded.plan.production
    if (
        fingerprint(static) != production.static_hash
        or production_inputs(loaded.target_state, static) != production.inputs
    ):
        raise ValueError("Repository static facts changed since evolve-plan; rerun evolve-plan.")
    if _production_version() != production.production_version:
        raise ValueError("Production version differs; explicitly run full prepare")
    refresh_w = set(production.refresh_ids("workflow")) | set(production.refresh_ids("tendency"))
    refresh_a = set(production.refresh_ids("anti-pattern"))
    refresh_s = set(production.refresh_ids("structural"))
    state = loaded.target_state
    mined = StructuralRuleMiner().mine(static)
    structural_refs = {
        key
        for item in mined.candidates
        if item.candidate_id in refresh_s
        for key in item.rule.evidence_ids
    }
    selected_mined = replace(
        mined,
        candidates=tuple(item for item in mined.candidates if item.candidate_id in refresh_s),
        evidence=tuple(item for item in mined.evidence if item.evidence_id in structural_refs),
    )
    cache = RepoSnapshotSemanticCache(root, cache_root)
    assembly = evaluate_semantic_knowledge(
        baseline.repository_id,
        static,
        selected_mined,
        cache.wrap(structural_model or CodexCliStructuralSemanticModel()),
        history=state.history,
        candidates=state.workflow_candidates,
        selected_workflows=tuple(
            item for item in state.workflow_candidates.candidates if item.candidate_id in refresh_w
        ),
        negative=state.antipattern_candidates,
        selected_anti_patterns=tuple(
            item
            for item in state.antipattern_candidates.candidates
            if item.candidate_id in refresh_a
        ),
        workflow_model=cache.wrap(workflow_model or CodexCliSemanticModel()),
        tendency_model=cache.wrap(tendency_model or CodexCliTendencySemanticModel()),
        anti_pattern_model=cache.wrap(antipattern_model or CodexCliAntiPatternSemanticModel()),
    )
    knowledge = merge_production_knowledge(baseline.production_knowledge, production, assembly)
    result = IncrementalEvolution().apply_production_semantic(
        baseline.source_dna,
        baseline.source_compilation,
        state,
        loaded.plan,
        baseline.production_knowledge,
        knowledge,
    )
    current_head = subprocess.run(
        ("git", "-c", f"safe.directory={root}", "rev-parse", "--verify", "HEAD^{commit}"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if (
        current_head.returncode
        or current_head.stdout.strip() != loaded.plan.target_head_commit
        or fingerprint(PythonStaticAnalyzer().analyze(root)) != production.static_hash
    ):
        raise ValueError("Repository changed during semantic refresh; rerun evolve-plan.")
    next_baseline = make_baseline(
        root,
        loaded.plan.target_head_commit,
        baseline.history_limit,
        result.resulting_project_dna,
        result.resulting_compilation,
        state,
        knowledge.structural,
        knowledge,
    )
    result_path = save_result(
        root, baseline, loaded.plan, knowledge.workflows, knowledge.anti_patterns, result
    )
    next_path = save_baseline(root, next_baseline)
    cache.publish(
        assembly.workflows, assembly.tendencies, assembly.anti_patterns, assembly.structural
    )
    counts = {
        channel: len(production.refresh_ids(channel))
        for channel in ("workflow", "tendency", "anti-pattern", "structural")
    }
    reused = sum(item.reason == "UNCHANGED_REUSED" for item in production.decisions)
    return EvolveApplyResult(
        {
            "repository": str(root),
            "baseline": str(baseline_path),
            "plan": str(plan_path),
            "baseline_id": baseline.baseline_id,
            "plan_id": loaded.plan.plan_id,
            "source_snapshot_id": result.source_snapshot_id,
            "result_snapshot_id": result.result_snapshot_id,
            "target_head_commit": result.target_head_commit,
            "semantic_refresh": {
                "workflow_candidates": counts["workflow"],
                "anti_pattern_candidates": counts["anti-pattern"],
                "tendency_candidates": counts["tendency"],
                "structural_candidates": counts["structural"],
            },
            "full_candidate_count": len(production.inputs),
            "incremental_refresh_count": sum(counts.values()),
            "reused_knowledge_count": reused,
            "semantic_calls_incremental": (
                sum(counts.values())
                + assembly.workflow_retry_count
                + assembly.anti_pattern_retry_count
                - cache.hits
            ),
            "semantic_cache_hits": cache.hits,
            "channel_decisions": {
                "workflow": {
                    "accepted": len(knowledge.workflows.workflows),
                    "rejected": len(knowledge.workflows.rejections),
                },
                "tendency": {
                    "accepted": len(knowledge.tendencies.accepted),
                    "rejected": len(knowledge.tendencies.rejections),
                },
                "anti-pattern": {
                    "accepted": len(knowledge.anti_patterns.anti_patterns),
                    "rejected": len(knowledge.anti_patterns.rejections),
                },
                "structural": {
                    "accepted": len(knowledge.structural.accepted),
                    "rejected": len(knowledge.structural.rejections),
                },
            },
            "refresh_decisions": [
                {"channel": item.channel, "candidate_id": item.candidate_id, "reason": item.reason}
                for item in production.decisions
            ],
            "semantic_failures": assembly.failures,
            "skills": {
                "added": len(result.skill_added_ids),
                "removed": len(result.skill_removed_ids),
                "version_changed": len(result.skill_version_changed_ids),
                "unchanged": len(result.skill_unchanged_ids),
            },
            "result_id": result.result_id,
            "result_artifact": result_path,
            "next_baseline_id": next_baseline.baseline_id,
            "next_baseline": next_path,
            "knowledge": {
                "workflows_refreshed": len(result.refreshed_workflow_ids),
                "workflows_added": len(result.added_workflow_ids),
                "workflows_unresolved": len(result.unresolved_workflow_ids),
                "anti_patterns_refreshed": len(result.refreshed_antipattern_ids),
                "anti_patterns_added": len(result.added_antipattern_ids),
                "anti_patterns_unresolved": len(result.unresolved_antipattern_ids),
            },
            "reviews_remaining": {
                "rules": len(loaded.plan.rule_review_ids),
                "routing_feedback": len(loaded.plan.routing_review_feedback_ids),
                "verify_feedback": len(loaded.plan.verify_review_feedback_ids),
                "unresolved_feedback": len(loaded.plan.unresolved_feedback_ids),
            },
        }
    )
