"""Reconstruct one normal, patch-enriched production workflow request without models."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

from skillforge.domain.serialization import to_json_value
from skillforge.repository.bounded_candidate_selection import scoped_candidates
from skillforge.repository.bounded_git_patch_evidence import validate_git_patch_evidence
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.history_patch_context import enrich_selected_history
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_evaluation import select_semantic_candidates
from skillforge.repository.semantic_policy import ANTIPATTERN_SEMANTIC_LIMIT
from skillforge.repository.semantic_workflows import build_semantic_workflow_request
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def _save_pickle(path: Path, value: object) -> None:
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(to_json_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_current_source_request(
    repo: Path | str,
    output: Path | str,
    *,
    history_limit: int = 50,
    current_module: str | None = None,
    workflow_limit: int = 1,
) -> dict[str, object]:
    """Run production miners, patch enrichment, selection, and request construction."""
    root = Path(repo).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("repo must be an existing repository directory")
    if type(history_limit) is not int or history_limit < 1:
        raise ValueError("history_limit must be a positive integer")
    if type(workflow_limit) is not int or workflow_limit < 1:
        raise ValueError("workflow_limit must be a positive integer")
    destination.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()

    history = GitHistoryMiner().mine(root, limit=history_limit)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    negative_candidates = NegativePatternMiner().mine(history)
    initial_selection = select_semantic_candidates(
        scoped_candidates(candidates, facts, current_module), limit=workflow_limit,
    )
    enriched = enrich_selected_history(
        root, history, facts, candidates, negative_candidates,
        workflow_limit=workflow_limit,
        anti_pattern_limit=ANTIPATTERN_SEMANTIC_LIMIT,
        current_module=current_module,
    )
    initial_history = history
    if enriched.history != initial_history:
        history = enriched.history
        patterns = ChangePatternMiner().mine(history, facts)
        candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    selected = select_semantic_candidates(
        scoped_candidates(candidates, facts, current_module), limit=workflow_limit,
    )

    for name, value in (
        ("history.pkl", history), ("static.pkl", facts), ("patterns.pkl", patterns),
        ("candidates.pkl", candidates), ("selected.pkl", selected),
    ):
        _save_pickle(destination / name, value)
    patch_stats = {
        "history_limit": history_limit,
        "current_module": current_module,
        "workflow_limit": workflow_limit,
        "head": head,
        "initial_selected_ids": [item.candidate_id for item in initial_selection],
        "final_selected_ids": [item.candidate_id for item in selected],
        "history_changed": enriched.history != initial_history,
        "added_git_patch_evidence": len(enriched.evidence),
        "git_reads": enriched.git_reads,
        "added_reference_bytes": enriched.added_reference_bytes,
        "skipped": [to_json_value(item) for item in enriched.skipped],
    }
    _save_json(destination / "patch-enrichment.json", patch_stats)
    if not selected:
        summary = {"status": "no_match", "head": head, "current_module": current_module,
                   "candidate_count": len(candidates.candidates), "request": None}
        _save_json(destination / "summary.json", summary)
        return summary

    request = build_semantic_workflow_request(history, facts, candidates, selected[0])
    evidence_by_id = {item.evidence_id: item for item in history.evidence}
    changes_by_key = {
        (changeset.commit_id, change.path): (changeset, change)
        for changeset in history.changesets for change in changeset.files
    }
    patch_refs = [item for item in request.evidence if item.provenance == "git_patch"]
    if not patch_refs:
        raise RuntimeError("selected workflow request has no git_patch Evidence after enrichment")
    for reference in patch_refs:
        evidence = evidence_by_id[reference.evidence_id]
        key = (evidence.location.commit, evidence.location.path)
        changeset, change = changes_by_key[key]
        if not validate_git_patch_evidence(evidence, changeset, change):
            raise ValueError("git_patch Evidence failed validation")

    _save_json(destination / "request.json", request)
    summary = {
        "status": "ready", "head": head, "current_module": current_module,
        "candidate_count": len(candidates.candidates),
        "selected_candidate_id": selected[0].candidate_id,
        "request_evidence_count": len(request.evidence),
        "request_git_patch_evidence_count": len(patch_refs),
        "request": "request.json",
    }
    _save_json(destination / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--current-module", "--scope", dest="current_module")
    parser.add_argument("--workflow-limit", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_current_source_request(
        args.repo, args.output, history_limit=args.history_limit,
        current_module=args.current_module, workflow_limit=args.workflow_limit,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
