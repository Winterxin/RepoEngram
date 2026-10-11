"""Real Codex evaluation of mined negative-pattern candidates."""

from __future__ import annotations

import json
from pathlib import Path

from skillforge.domain.negative_patterns import (
    NegativePatternCandidate,
    NegativePatternCandidateResult,
    NegativeSignalKind,
)
from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionResponse
from skillforge.domain.serialization import to_json_value
from skillforge.repository.codex_cli_antipattern import (
    CODEX_ANTIPATTERN_MODEL,
    CodexCliAntiPatternError,
    CodexCliAntiPatternSemanticModel,
)
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_antipatterns import (
    AntiPatternSemanticModel,
    SemanticAntiPatternAbstracter,
    build_semantic_antipattern_request,
)


def select_semantic_antipattern_candidates(
    result: NegativePatternCandidateResult, *, limit: int
) -> tuple[NegativePatternCandidate, ...]:
    """Spend the fixed quota on evidence signals, not alphabetical file names.

    Ranking is only a heuristic. The existing semantic/evidence gates still
    decide whether a negative event and reusable guidance are established.
    """
    if limit < 1:
        raise ValueError("semantic candidate limit must be positive")
    signals = {item.signal_id: item for item in result.signals}
    evidence = {item.evidence_id: item for item in result.evidence}
    features = {}
    for candidate in result.candidates:
        related = tuple(signals[key] for key in candidate.signal_ids)
        verified = any(item.kind == NegativeSignalKind.REVERT_SHA_REFERENCE
                       for item in related)
        corrections = tuple(item for item in related if item.kind in {
            NegativeSignalKind.CORRECTION_MESSAGE_HINT,
            NegativeSignalKind.EXPLICIT_REVERT_HINT,
            NegativeSignalKind.REVERT_SHA_REFERENCE,
        })
        # Use only already bounded Python patch observations. A changed guard,
        # return or exception is a ranking hint, never proof of a historical bug.
        control_change = any(
            item.provenance == "git_patch"
            and Path(item.location.path).suffix == ".py"
            and any(line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
                    and line[1:].lstrip().startswith(("if ", "elif ", "return ", "raise "))
                    for line in item.description.splitlines())
            for item in (evidence[key] for key in candidate.evidence_ids)
        )
        commits = frozenset(key for item in corrections for key in item.commit_ids)
        features[candidate.candidate_id] = (
            verified, bool(corrections) and control_change, bool(corrections), commits
        )
    remaining = list(result.candidates)
    selected = []
    covered = set()
    while remaining and len(selected) < limit:
        def rank(candidate):
            verified, control, correction, commits = features[candidate.candidate_id]
            return (not verified, not control, not correction,
                    -len(commits - covered), -len(commits),
                    candidate.affected_paths, candidate.candidate_id)
        candidate = min(remaining, key=rank)
        selected.append(candidate)
        covered.update(features[candidate.candidate_id][3])
        remaining.remove(candidate)
    return tuple(selected)


def evaluate_semantic_antipatterns(
    repo_path: Path, *, history_limit: int = 50, limit: int = 5,
    model: AntiPatternSemanticModel | None = None,
) -> dict[str, object]:
    """Mine Git history and record each candidate's validated decision or failure."""
    if history_limit < 0:
        raise ValueError("history limit must be nonnegative")
    if limit < 1:
        raise ValueError("semantic candidate limit must be positive")
    history = GitHistoryMiner().mine(repo_path, limit=history_limit)
    candidates = NegativePatternMiner().mine(history)
    selected = select_semantic_antipattern_candidates(candidates, limit=limit)
    semantic_model = model if model is not None else CodexCliAntiPatternSemanticModel()
    abstracter = SemanticAntiPatternAbstracter()
    entries: list[dict[str, object]] = []
    for candidate in selected:
        class RecordingModel:
            response: SemanticAntiPatternDecisionResponse | None = None

            def complete(self, request):
                self.response = semantic_model.complete(request)
                return self.response

        recording_model = RecordingModel()
        entry: dict[str, object] = {
            "candidate": to_json_value(candidate),
            "request": None,
            "model_response": None,
            "status": "validation_error",
            "draft": None,
            "rejection": None,
            "validation_error": None,
        }
        try:
            request = build_semantic_antipattern_request(history, candidates, candidate)
            entry["request"] = request.to_dict()
            result = abstracter.abstract(
                history, candidates, recording_model, selected_candidates=(candidate,)
            )
        except CodexCliAntiPatternError as error:
            entry["status"] = "model_error"
            entry["validation_error"] = {"stage": "codex_cli", "message": str(error)}
        except ValueError as error:
            entry["validation_error"] = {"stage": "semantic_validator", "message": str(error)}
        else:
            if result.anti_patterns:
                entry["status"] = "accepted_by_validator"
                entry["draft"] = to_json_value(result.anti_patterns[0])
            else:
                entry["status"] = "semantic_rejected"
                entry["rejection"] = to_json_value(result.rejections[0])
        if recording_model.response is not None:
            entry["model_response"] = recording_model.response.to_dict()
        entries.append(entry)
    return {
        "repository_id": candidates.repository_id,
        "provider": "codex-cli",
        "model": CODEX_ANTIPATTERN_MODEL,
        "history_limit": history_limit,
        "requested_limit": limit,
        "selected_count": len(selected),
        "entries": entries,
    }


def write_semantic_antipattern_evaluation_report(
    report: dict[str, object], path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
