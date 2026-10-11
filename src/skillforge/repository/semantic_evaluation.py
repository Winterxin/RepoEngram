"""Deterministic semantic evaluation and human-review report."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.semantic_workflow import SemanticDecisionResponse
from skillforge.domain.serialization import to_json_value
from skillforge.domain.workflow import WorkflowCandidate, WorkflowCandidateResult
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.codex_cli_semantic import (
    CODEX_SEMANTIC_MODEL,
    CodexCliSemanticError,
    CodexCliSemanticModel,
)
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.semantic_workflows import SemanticWorkflowAbstracter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def select_semantic_candidates(
    result: WorkflowCandidateResult, *, kind: ChangePatternKind | None = None, limit: int
) -> tuple[WorkflowCandidate, ...]:
    """Select support-ranked candidates while preserving the original channel mix."""
    selected, _ = _select_with_reasons(result, kind=kind, limit=limit)
    return selected


def _module_commit_duplicate(left: WorkflowCandidate, right: WorkflowCandidate) -> bool:
    """Collapse only exact co-change equivalents, including unmapped file members."""
    co_change_kinds = {
        ChangePatternKind.FILE_CO_CHANGE,
        ChangePatternKind.MODULE_CO_CHANGE,
    }
    if left.kind not in co_change_kinds or right.kind not in co_change_kinds:
        return False
    left_units = _candidate_units(left)
    right_units = _candidate_units(right)
    if not left_units or not right_units or not left.supporting_commits:
        return False
    return left_units == right_units and set(left.supporting_commits) == set(
        right.supporting_commits
    )


def _candidate_units(candidate: WorkflowCandidate) -> frozenset[tuple[str, str]]:
    """Use canonical modules where known and namespaced original paths otherwise."""
    return frozenset(
        ("module", member.current_module) if member.current_module is not None
        else ("path", member.value)
        for member in candidate.members
    )


def _selection_channel(candidate: WorkflowCandidate) -> str:
    return (
        "tendency"
        if candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
        else "workflow"
    )


def _select_channel_candidates(
    candidates: tuple[WorkflowCandidate, ...], *, limit: int, reasons: dict[str, str]
) -> tuple[WorkflowCandidate, ...]:
    selected: list[WorkflowCandidate] = []
    remaining = list(candidates)
    deferred_duplicates: list[WorkflowCandidate] = []
    covered_units: set[tuple[str, str]] = set()
    miner_rank = {candidate.candidate_id: rank for rank, candidate in enumerate(candidates)}
    while remaining:
        if not selected:
            candidate = remaining.pop(0)
            selected.append(candidate)
            covered_units.update(_candidate_units(candidate))
            reasons[candidate.candidate_id] = (
                "selected as the highest-support channel representative"
            )
            continue

        # Exact equivalents are held for quota refill, with a module representative
        # replacing a file candidate only when its support is at least as strong.
        index = 0
        while index < len(remaining):
            candidate = remaining[index]
            duplicate_index = next((position for position, previous in enumerate(selected)
                                    if _module_commit_duplicate(candidate, previous)), None)
            if duplicate_index is None:
                index += 1
                continue
            previous = selected[duplicate_index]
            prefer_module = (
                previous.kind == ChangePatternKind.FILE_CO_CHANGE
                and candidate.kind == ChangePatternKind.MODULE_CO_CHANGE
                and candidate.support.supporting >= previous.support.supporting
            )
            if prefer_module:
                selected[duplicate_index] = candidate
                deferred_duplicates.append(previous)
                reasons[previous.candidate_id] = (
                    f"exact duplicate of selected candidate {candidate.candidate_id}"
                )
                reasons[candidate.candidate_id] = (
                    "selected as the module-level representative of an exact file equivalent"
                )
            else:
                deferred_duplicates.append(candidate)
                reasons[candidate.candidate_id] = (
                    f"exact duplicate of selected candidate {previous.candidate_id}"
                )
            remaining.pop(index)

        if not remaining:
            break
        if len(selected) >= limit:
            break

        def score(candidate: WorkflowCandidate) -> tuple[Fraction, int, int]:
            units = _candidate_units(candidate)
            novelty = Fraction(len(units - covered_units), len(units)) if units else Fraction(0)
            return novelty, candidate.support.supporting, -miner_rank[candidate.candidate_id]

        candidate = max(remaining, key=score)
        remaining.remove(candidate)
        selected.append(candidate)
        covered_units.update(_candidate_units(candidate))
        reasons[candidate.candidate_id] = (
            "selected for new member coverage; support rank breaks ties"
        )

    # Preserve the old channel quota if only duplicate candidates remain.
    selected_ids = {candidate.candidate_id for candidate in selected}
    for candidate in deferred_duplicates:
        if len(selected) >= limit:
            break
        if candidate.candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
        reasons[candidate.candidate_id] = (
            "selected to preserve the channel quota after distinct candidates were exhausted"
        )
    return tuple(selected)


def _select_with_reasons(
    result: WorkflowCandidateResult,
    *,
    kind: ChangePatternKind | None,
    limit: int,
) -> tuple[tuple[WorkflowCandidate, ...], dict[str, str]]:
    if limit < 1:
        raise ValueError("semantic evaluation limit must be positive")
    candidates = tuple(
        candidate for candidate in result.candidates
        if kind is None or candidate.kind == kind
    )
    reasons: dict[str, str] = {}
    original_head = candidates[:limit]
    quotas: dict[str, int] = {}
    for candidate in original_head:
        channel = _selection_channel(candidate)
        quotas[channel] = quotas.get(channel, 0) + 1

    selected_ids: set[str] = set()
    for channel, quota in quotas.items():
        channel_candidates = tuple(
            candidate for candidate in candidates
            if _selection_channel(candidate) == channel
        )
        selected_ids.update(candidate.candidate_id for candidate in
                             _select_channel_candidates(
                                 channel_candidates, limit=quota, reasons=reasons,
                             ))
    return tuple(candidate for candidate in candidates
                 if candidate.candidate_id in selected_ids), reasons


def semantic_selection_report(result: WorkflowCandidateResult, *, limit: int) -> tuple[dict, ...]:
    selected_items, reasons = _select_with_reasons(result, kind=None, limit=limit)
    selected = {item.candidate_id for item in selected_items}
    return tuple({
        "candidate_id": item.candidate_id, "kind": item.kind.value,
        "rank": rank, "support": item.support.supporting,
        "selected": item.candidate_id in selected,
        "reason": reasons.get(
            item.candidate_id, "outside support-ranked diverse limit",
        ),
    } for rank, item in enumerate(result.candidates, 1))


def evaluate_semantic_workflows(
    path: Path, *, kind: ChangePatternKind = ChangePatternKind.MODULE_CO_CHANGE,
    limit: int = 5, model: CodexCliSemanticModel | None = None,
) -> dict[str, object]:
    """Mine existing evidence and record accepted drafts or explicit failures."""
    history = GitHistoryMiner().mine(path)
    static = PythonStaticAnalyzer().analyze(path)
    patterns = ChangePatternMiner().mine(history, static)
    candidates = WorkflowCandidateMiner().mine(history, static, patterns)
    selected = select_semantic_candidates(candidates, kind=kind, limit=limit)
    semantic_model = model or CodexCliSemanticModel()
    abstracter = SemanticWorkflowAbstracter()
    entries: list[dict[str, object]] = []
    for candidate in selected:
        class RecordingModel:
            response: SemanticDecisionResponse | None = None

            def complete(self, request):
                self.response = semantic_model.complete(request)
                return self.response

        recording_model = RecordingModel()
        entry: dict[str, object] = {
            "candidate": to_json_value(candidate),
            "decision": None,
            "semantic": None,
            "rejection": None,
            "model_response": None,
            "evidence_ids": list(candidate.evidence_ids),
            "status": "failed",
        }
        try:
            result = abstracter.abstract(
                history, static, candidates, recording_model, selected_candidates=(candidate,)
            )
        except CodexCliSemanticError as error:
            entry["error"] = {"stage": "codex_cli", "message": str(error)}
        except ValueError as error:
            entry["error"] = {"stage": "semantic_validator", "message": str(error)}
        else:
            if result.workflows:
                entry["decision"] = "WORKFLOW"
                entry["semantic"] = to_json_value(result.workflows[0])
                entry["evidence_ids"] = list(result.workflows[0].evidence_ids)
                entry["status"] = "accepted_by_validator"
            else:
                rejection = result.rejections[0]
                entry["decision"] = rejection.decision.value
                entry["rejection"] = to_json_value(rejection)
                entry["evidence_ids"] = list(rejection.evidence_ids)
                entry["status"] = "semantic_rejected"
        if recording_model.response is not None:
            entry["model_response"] = recording_model.response.to_dict()
            if entry["decision"] is None:
                entry["decision"] = recording_model.response.decision.value
        entries.append(entry)
    return {
        "repository_id": candidates.repository_id,
        "provider": "codex_cli",
        "model": CODEX_SEMANTIC_MODEL,
        "kind": kind.value,
        "requested_limit": limit,
        "selected_count": len(selected),
        "entries": entries,
    }


def write_semantic_evaluation_report(report: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
