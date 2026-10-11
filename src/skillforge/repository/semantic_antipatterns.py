"""Build candidate-scoped Anti-pattern context and gate untrusted model output."""

from __future__ import annotations

import hashlib
import re
from math import isfinite
from typing import Protocol

from skillforge.domain.change import GitHistoryResult
from skillforge.domain.evidence import Confidence, Evidence, EvidenceType
from skillforge.domain.negative_patterns import (
    NegativePatternCandidate,
    NegativePatternCandidateResult,
    NegativeSignal,
    NegativeSignalKind,
)
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
    SemanticAntiPatternDraft,
    SemanticAntiPatternEvidenceRef,
    SemanticAntiPatternException,
    SemanticAntiPatternRejection,
    SemanticAntiPatternRequest,
    SemanticAntiPatternResponse,
    SemanticAntiPatternResult,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
    SemanticNegativeChangedFile,
    SemanticNegativeCommit,
    SemanticNegativeSignal,
)
from skillforge.errors import SemanticDecisionValidationError

CONFIDENCE_METHOD = "llm_semantic_antipattern_v1"
_ALLOWED_PROVENANCE = frozenset({"git_log", "git_diff", "git_numstat", "git_patch"})
_COMMIT_TOKEN = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{7,64}(?![A-Za-z0-9])")
_PATH_TOKEN = re.compile(
    r"(?<![\w.])(?:[\w.-]+/)+[\w.-]+"
    r"|(?<![\w/])[\w.-]+\.[A-Za-z0-9]{1,12}(?!\w)",
    re.UNICODE,
)


class AntiPatternSemanticModel(Protocol):
    """Synchronous structured model port; implementations may be local fakes."""

    def complete(
        self, request: SemanticAntiPatternRequest
    ) -> SemanticAntiPatternDecisionResponse:
        """Return an untrusted interpretation of exactly one candidate."""


def _index_unique(items: tuple, key, label: str) -> dict:
    result = {}
    for item in items:
        item_key = key(item)
        if item_key in result:
            if result[item_key] != item:
                raise ValueError(f"Conflicting {label}: {item_key}")
            raise ValueError(f"Duplicate {label}: {item_key}")
        result[item_key] = item
    return result


def _evidence_reference(evidence: Evidence) -> SemanticAntiPatternEvidenceRef:
    return SemanticAntiPatternEvidenceRef(
        evidence_id=evidence.evidence_id,
        kind=evidence.kind.value,
        path=evidence.location.path,
        commit_id=evidence.location.commit,
        line=evidence.location.line,
        description=evidence.description,
        provenance=evidence.provenance,
    )


def build_semantic_antipattern_request(
    history: GitHistoryResult,
    candidate_result: NegativePatternCandidateResult,
    candidate: NegativePatternCandidate,
) -> SemanticAntiPatternRequest:
    """Build one request using only this candidate's signal and Evidence closure."""

    repository_id = candidate_result.repository_id
    if history.repository_id != repository_id or candidate.repository_id != repository_id:
        raise ValueError("all inputs must describe the same repository_id")
    if candidate_result.examined_commits != len(history.changesets):
        raise ValueError("candidate examined_commits mismatch with Git history")
    matching_candidates = [
        item for item in candidate_result.candidates
        if item.candidate_id == candidate.candidate_id
    ]
    if len(matching_candidates) != 1 or matching_candidates[0] != candidate:
        raise ValueError("candidate is not present unchanged in NegativePatternCandidateResult")
    if any(item.repository_id != repository_id for item in history.changesets):
        raise ValueError("Git history changeset repository_id mismatch")

    changesets = _index_unique(
        history.changesets, lambda item: item.commit_id, "Git history commit_id"
    )
    history_evidence = _index_unique(
        history.evidence, lambda item: item.evidence_id, "Git history Evidence ID"
    )
    candidate_evidence = _index_unique(
        candidate_result.evidence,
        lambda item: item.evidence_id,
        "Negative candidate Evidence ID",
    )
    signals_by_id = _index_unique(
        candidate_result.signals, lambda item: item.signal_id, "NegativeSignal ID"
    )
    selected_signals: list[NegativeSignal] = []
    for signal_id in candidate.signal_ids:
        signal = signals_by_id.get(signal_id)
        if signal is None:
            raise ValueError(f"Missing candidate signal: {signal_id}")
        if signal.repository_id != repository_id:
            raise ValueError(f"NegativeSignal repository_id mismatch: {signal_id}")
        selected_signals.append(signal)
    if tuple(sorted(signal.signal_id for signal in selected_signals)) != candidate.signal_ids:
        raise ValueError("candidate signal_ids do not match its signal closure")

    signal_commits = tuple(sorted({
        commit_id for signal in selected_signals for commit_id in signal.commit_ids
    }))
    if candidate.commit_ids != signal_commits:
        raise ValueError("candidate commit_ids do not match its signal closure")
    if candidate.signal_count != len(selected_signals):
        raise ValueError("candidate signal_count does not match its signal closure")
    if candidate.involved_commit_count != len(signal_commits):
        raise ValueError("candidate involved_commit_count does not match its signal closure")
    expected_evidence_ids = {
        evidence_id for signal in selected_signals for evidence_id in signal.evidence_ids
    }
    if candidate.evidence_ids != tuple(sorted(expected_evidence_ids)):
        raise ValueError("candidate evidence_ids do not match its signal closure")
    missing = expected_evidence_ids - candidate_evidence.keys()
    if missing:
        raise ValueError("Missing candidate Evidence: " + ", ".join(sorted(missing)))

    allowed_paths = set(candidate.affected_paths)
    if candidate.affected_paths:
        if len(candidate.affected_paths) != 1 or any(
            candidate.affected_paths[0] not in signal.affected_paths
            or not set(signal.affected_paths) <= allowed_paths
            for signal in selected_signals
        ):
            raise ValueError("candidate signals must share its exact affected path")
    elif len(selected_signals) != 1 or selected_signals[0].affected_paths:
        raise ValueError("pathless signals must remain separate candidates")

    selected_evidence: dict[str, Evidence] = {}
    for evidence_id in sorted(expected_evidence_ids):
        item = candidate_evidence[evidence_id]
        if item.repository_id != repository_id:
            raise ValueError(f"Evidence repository_id mismatch: {evidence_id}")
        if item.kind != EvidenceType.GIT or item.provenance not in _ALLOWED_PROVENANCE:
            raise ValueError(f"Invalid candidate Git Evidence: {evidence_id}")
        if item.evidence_id not in history_evidence:
            raise ValueError(f"Candidate Evidence missing from Git history: {evidence_id}")
        if history_evidence[evidence_id] != item:
            raise ValueError(f"Candidate/input Evidence mismatch: {evidence_id}")
        selected_evidence[evidence_id] = item

    for signal in selected_signals:
        signal_commit_set = set(signal.commit_ids)
        linked_evidence = tuple(
            selected_evidence[evidence_id] for evidence_id in signal.evidence_ids
        )
        for item in linked_evidence:
            if item.location.commit not in signal_commit_set:
                raise ValueError(
                    f"NegativeSignal Evidence references an unrelated commit: "
                    f"{signal.signal_id}"
                )
            if item.provenance != "git_log" and item.location.path not in signal.affected_paths:
                raise ValueError(
                    f"NegativeSignal Evidence references an unrelated path: {signal.signal_id}"
                )
        log_commits = {
            item.location.commit for item in linked_evidence
            if item.provenance == "git_log"
        }
        if signal.kind in {
            NegativeSignalKind.EXPLICIT_REVERT_HINT,
            NegativeSignalKind.CORRECTION_MESSAGE_HINT,
        }:
            if not set(signal.commit_ids) <= log_commits:
                raise ValueError(
                    f"Message-based NegativeSignal is missing git_log Evidence: "
                    f"{signal.signal_id}"
                )
            if any(
                not any(
                    item.provenance == "git_diff"
                    and item.location.commit == signal.commit_ids[0]
                    and item.location.path == path
                    for item in linked_evidence
                )
                for path in signal.affected_paths
            ):
                raise ValueError(
                    f"Message-based NegativeSignal is missing related git_diff Evidence: "
                    f"{signal.signal_id}"
                )
        elif signal.kind == NegativeSignalKind.REVERT_SHA_REFERENCE:
            if not set(signal.commit_ids) <= log_commits:
                raise ValueError(
                    f"Revert reference is missing related git_log Evidence: {signal.signal_id}"
                )
            revert_commit = next(
                commit_id for commit_id in signal.commit_ids
                if commit_id != signal.referenced_commit_id
            )
            if signal.affected_paths and not any(
                item.provenance == "git_diff"
                and item.location.commit == revert_commit
                and item.location.path in signal.affected_paths
                for item in linked_evidence
            ):
                raise ValueError(
                    f"Revert reference is missing revert-path git_diff Evidence: "
                    f"{signal.signal_id}"
                )
        elif signal.kind == NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT:
            if any(
                not any(
                    item.provenance == "git_diff"
                    and item.location.commit == commit_id
                    and item.location.path == path
                    for item in linked_evidence
                )
                for commit_id in signal.commit_ids
                for path in signal.affected_paths
            ):
                raise ValueError(
                    f"Repeated-edit signal is missing related git_diff Evidence: "
                    f"{signal.signal_id}"
                )

    missing_commits = set(signal_commits) - changesets.keys()
    if missing_commits:
        raise ValueError(
            "Missing candidate commits: " + ", ".join(sorted(missing_commits))
        )

    selected_by_commit: dict[str, list[Evidence]] = {item: [] for item in signal_commits}
    for item in selected_evidence.values():
        commit_id = item.location.commit
        if commit_id not in selected_by_commit:
            raise ValueError(f"Candidate Evidence references unknown commit: {item.evidence_id}")
        changeset = changesets[commit_id]
        if item.evidence_id not in changeset.evidence_ids:
            raise ValueError(f"Candidate Evidence is not linked to its commit: {item.evidence_id}")
        if item.provenance == "git_log":
            if item.location.path != ".":
                raise ValueError(f"git_log Evidence has an invalid path: {item.evidence_id}")
        else:
            if item.location.path not in allowed_paths:
                raise ValueError(f"Candidate Evidence references unknown path: {item.evidence_id}")
            matching_files = [
                changed_file for changed_file in changeset.files
                if changed_file.path == item.location.path
                and item.evidence_id in changed_file.evidence_ids
            ]
            if len(matching_files) != 1:
                raise ValueError(
                    f"Candidate file Evidence has no unique Git change: {item.evidence_id}"
                )
            if item.evidence_id not in matching_files[0].evidence_ids:
                raise ValueError(f"Candidate file Evidence mismatch: {item.evidence_id}")
            if item.provenance == "git_patch":
                from skillforge.repository.bounded_git_patch_evidence import (
                    validate_git_patch_evidence,
                )

                if not validate_git_patch_evidence(item, changeset, matching_files[0]):
                    raise ValueError("Invalid historical patch Evidence")
        selected_by_commit[commit_id].append(item)
    if any(not selected_by_commit[item] for item in signal_commits):
        raise ValueError("candidate commit is missing linked Evidence")

    semantic_signals = tuple(sorted((
        SemanticNegativeSignal(
            signal_id=item.signal_id,
            kind=item.kind,
            commit_ids=item.commit_ids,
            affected_paths=item.affected_paths,
            method=item.method,
            heuristic=item.heuristic,
            matched_terms=item.matched_terms,
            referenced_commit_id=item.referenced_commit_id,
            evidence_ids=item.evidence_ids,
        )
        for item in selected_signals
    ), key=lambda item: item.signal_id))

    semantic_commits: list[SemanticNegativeCommit] = []
    for commit_id in signal_commits:
        changeset = changesets[commit_id]
        linked = selected_by_commit[commit_id]
        log_evidence = [item for item in linked if item.provenance == "git_log"]
        if len(log_evidence) > 1:
            raise ValueError(f"Duplicate candidate git_log Evidence: {commit_id}")
        message = None
        if log_evidence:
            if not isinstance(changeset.message, str):
                raise ValueError(f"Invalid Git message for evidenced commit: {commit_id}")
            message = changeset.message

        changed_files: list[SemanticNegativeChangedFile] = []
        for changed_file in changeset.files:
            if changed_file.path not in allowed_paths:
                continue
            linked_ids = set(changed_file.evidence_ids) & expected_evidence_ids
            if not linked_ids:
                continue
            diff_ids = {
                evidence_id for evidence_id in linked_ids
                if selected_evidence[evidence_id].provenance == "git_diff"
            }
            numstat_ids = {
                evidence_id for evidence_id in linked_ids
                if selected_evidence[evidence_id].provenance == "git_numstat"
            }
            if not diff_ids and not numstat_ids:
                continue
            changed_files.append(SemanticNegativeChangedFile(
                path=changed_file.path,
                kind=changed_file.kind.value if diff_ids else None,
                previous_path=changed_file.previous_path if diff_ids else None,
                additions=changed_file.additions if numstat_ids else None,
                deletions=changed_file.deletions if numstat_ids else None,
                evidence_ids=tuple(sorted(diff_ids | numstat_ids | {
                    key for key in linked_ids if selected_evidence[key].provenance == "git_patch"
                })),
            ))
        changed_files.sort(key=lambda item: (item.path, item.kind or "", item.previous_path or ""))
        semantic_commits.append(SemanticNegativeCommit(
            commit_id=commit_id,
            message=message,
            relevant_changed_files=tuple(changed_files),
            evidence_ids=tuple(sorted(item.evidence_id for item in linked)),
        ))

    return SemanticAntiPatternRequest(
        repository_id=repository_id,
        candidate_id=candidate.candidate_id,
        affected_paths=candidate.affected_paths,
        signal_count=candidate.signal_count,
        involved_commit_count=candidate.involved_commit_count,
        examined_commits=candidate.examined_commits,
        signals=semantic_signals,
        commits=tuple(semantic_commits),
        evidence=tuple(
            _evidence_reference(selected_evidence[key])
            for key in sorted(selected_evidence)
        ),
    )


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_evidence_ids(
    values: object, allowed: set[str], label: str
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} must cite at least one Evidence ID")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{label} contains an invalid Evidence ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate Evidence IDs")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{label} cites unknown Candidate Evidence: {', '.join(sorted(unknown))}")
    return tuple(sorted(values))


def _request_paths(request: SemanticAntiPatternRequest) -> set[str]:
    paths = set(request.affected_paths)
    for commit in request.commits:
        for changed_file in commit.relevant_changed_files:
            paths.add(changed_file.path)
            if changed_file.previous_path is not None:
                paths.add(changed_file.previous_path)
    return paths - {"."}


def _request_commits(request: SemanticAntiPatternRequest) -> set[str]:
    return {item.commit_id for item in request.commits}


def _text_claims(anti_pattern: SemanticAntiPatternResponse) -> tuple[str, ...]:
    claims = [
        anti_pattern.name,
        anti_pattern.description,
        anti_pattern.avoid_behavior,
        anti_pattern.negative_evidence_summary,
        anti_pattern.scope.description,
        anti_pattern.scope.reason,
        *anti_pattern.scope.selectors,
    ]
    claims.extend(item.description for item in anti_pattern.exceptions)
    return tuple(claims)


def _validate_text_references(
    claims: tuple[str, ...], allowed_commits: set[str], allowed_paths: set[str],
    rejection_symbols: frozenset[str] = frozenset(),
) -> None:
    normalized_commits = {item.lower() for item in allowed_commits}
    for claim in claims:
        if not isinstance(claim, str):
            raise ValueError("semantic claims must be strings")
        for token in _COMMIT_TOKEN.findall(claim):
            matches = [
                commit for commit in normalized_commits if commit.startswith(token.lower())
            ]
            if len(matches) != 1:
                raise ValueError(f"model text references unknown commit: {token}")
        path_free_claim = claim
        for allowed_path in sorted(allowed_paths, key=len, reverse=True):
            escaped_path = re.escape(allowed_path)
            exact_path = re.compile(rf"(?<![\w./-]){escaped_path}(?![\w./-])")
            sentence_path = re.compile(
                rf"(?<![\w./-]){escaped_path}\.(?=$|[\s,;:!?)]|['\"])"
            )
            path_free_claim = exact_path.sub(" ", path_free_claim)
            path_free_claim = sentence_path.sub(" ", path_free_claim)
        for token in _PATH_TOKEN.findall(path_free_claim):
            path = token.rstrip(".,;:!?)]}\"'")
            if path in rejection_symbols:
                continue
            if path and path not in allowed_paths:
                if re.fullmatch(r"v?\d+(?:\.\d+)+", path, re.IGNORECASE):
                    continue
                raise ValueError(
                    f"model text references unknown path: {path}; use only exact supplied "
                    "paths. Write operation alternatives with 'or' and function names "
                    "without dotted prefixes; do not invent a replacement path.")


def _cited_rejection_symbols(request, evidence_ids):
    """A rejection can quote class members from its cited historical log.

    These names describe why no knowledge is emitted. They never exempt an
    accepted draft, known file extension, scope, or uncited source claim.
    """
    from skillforge.repository.semantic_references import cited_rejection_symbols
    from skillforge.repository.semantic_workflows import _PATH_TOKEN as file_token

    return cited_rejection_symbols(request.evidence, request.commits, evidence_ids, file_token)


def _validate_scope(
    scope: SemanticAntiPatternScope, request: SemanticAntiPatternRequest
) -> SemanticAntiPatternScope:
    if not isinstance(scope, SemanticAntiPatternScope):
        raise ValueError("model Anti-pattern scope has an invalid structure")
    if not isinstance(scope.kind, SemanticAntiPatternScopeKind):
        raise ValueError("model Anti-pattern scope kind is invalid")
    selectors = scope.selectors
    if not isinstance(selectors, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in selectors
    ):
        raise ValueError("Anti-pattern scope selectors must be non-empty strings")
    if len(selectors) != len(set(selectors)):
        raise ValueError("Anti-pattern scope selectors contain duplicates")
    description = _nonempty(scope.description, "scope description")
    reason = _nonempty(scope.reason, "scope reason")
    if scope.kind in {
        SemanticAntiPatternScopeKind.UNKNOWN,
        SemanticAntiPatternScopeKind.REPOSITORY,
    }:
        if selectors:
            raise ValueError(f"{scope.kind.value} scope must not have selectors")
    elif scope.kind == SemanticAntiPatternScopeKind.FILE:
        if len(request.affected_paths) != 1:
            raise ValueError("FILE scope requires a path-bound candidate")
        if selectors != request.affected_paths:
            raise ValueError("FILE scope selector must exactly equal the candidate path")
    else:
        raise ValueError("Anti-pattern scope kind is unsupported")
    if request.affected_paths and scope.kind == SemanticAntiPatternScopeKind.REPOSITORY:
        raise ValueError("path-bound candidates cannot be promoted to repository scope")
    return SemanticAntiPatternScope(
        scope.kind,
        tuple(item.strip() for item in selectors),
        description,
        reason,
        scope.evidence_ids,
    )


def _validate_anti_pattern(
    request: SemanticAntiPatternRequest,
    value: SemanticAntiPatternResponse,
    allowed_evidence: set[str],
) -> tuple[SemanticAntiPatternResponse, set[str]]:
    if not isinstance(value, SemanticAntiPatternResponse):
        raise ValueError("ANTI_PATTERN decision requires a complete Anti-pattern object")
    name = _nonempty(value.name, "Anti-pattern name")
    description = _nonempty(value.description, "Anti-pattern description")
    avoid_behavior = _nonempty(value.avoid_behavior, "avoid_behavior")
    negative_summary = _nonempty(
        value.negative_evidence_summary, "negative_evidence_summary"
    )
    name_ids = _validate_evidence_ids(value.name_evidence_ids, allowed_evidence, "name")
    description_ids = _validate_evidence_ids(
        value.description_evidence_ids, allowed_evidence, "description"
    )
    avoid_ids = _validate_evidence_ids(
        value.avoid_behavior_evidence_ids, allowed_evidence, "avoid_behavior"
    )
    negative_ids = _validate_evidence_ids(
        value.negative_evidence_ids, allowed_evidence, "negative evidence summary"
    )
    scope = _validate_scope(value.scope, request)
    scope_ids = _validate_evidence_ids(
        scope.evidence_ids, allowed_evidence, "Anti-pattern scope"
    )
    scope = SemanticAntiPatternScope(
        scope.kind, scope.selectors, scope.description, scope.reason, scope_ids
    )

    if not isinstance(value.exceptions, tuple):
        raise ValueError("Anti-pattern exceptions must be a tuple")
    exceptions: list[SemanticAntiPatternException] = []
    exception_evidence: set[str] = set()
    for item in value.exceptions:
        if not isinstance(item, SemanticAntiPatternException):
            raise ValueError("Anti-pattern exception has an invalid structure")
        normalized = SemanticAntiPatternException(
            _nonempty(item.description, "exception description"),
            _validate_evidence_ids(item.evidence_ids, allowed_evidence, "exception"),
        )
        exception_evidence.update(normalized.evidence_ids)
        exceptions.append(normalized)

    referenced = (
        set(name_ids) | set(description_ids) | set(avoid_ids) | set(negative_ids)
        | set(scope_ids) | exception_evidence
    )
    aggregate = _validate_evidence_ids(
        value.evidence_ids, allowed_evidence, "Anti-pattern"
    )
    if set(aggregate) != referenced:
        raise ValueError("Anti-pattern Evidence IDs must match its claim reference closure")
    normalized = SemanticAntiPatternResponse(
        name=name,
        name_evidence_ids=name_ids,
        description=description,
        description_evidence_ids=description_ids,
        avoid_behavior=avoid_behavior,
        avoid_behavior_evidence_ids=avoid_ids,
        negative_evidence_summary=negative_summary,
        negative_evidence_ids=negative_ids,
        scope=scope,
        exceptions=tuple(exceptions),
        evidence_ids=tuple(sorted(referenced)),
    )
    _validate_text_references(
        _text_claims(normalized), _request_commits(request), _request_paths(request)
    )
    return normalized, referenced


def _validate_decision(
    request: SemanticAntiPatternRequest,
    response: SemanticAntiPatternDecisionResponse,
) -> tuple[
    SemanticAntiPatternDraft | None, SemanticAntiPatternRejection | None, set[str]
]:
    if not isinstance(response, SemanticAntiPatternDecisionResponse):
        raise ValueError("semantic model returned an invalid decision response")
    if response.repository_id != request.repository_id:
        raise ValueError("semantic response repository_id mismatch")
    if response.candidate_id != request.candidate_id:
        raise ValueError("semantic response candidate_id mismatch")
    if not isinstance(response.decision, SemanticAntiPatternDecisionKind):
        raise ValueError("semantic decision is invalid")
    if isinstance(response.confidence_score, bool) or not isinstance(
        response.confidence_score, (int, float)
    ):
        raise ValueError("Confidence score must be numeric")
    score = float(response.confidence_score)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0 and 1")
    reason = _nonempty(response.reason, "semantic decision reason")
    allowed_evidence = {item.evidence_id for item in request.evidence}
    decision_evidence = _validate_evidence_ids(
        response.decision_evidence_ids, allowed_evidence, "semantic decision"
    )
    _validate_text_references(
        (reason,), _request_commits(request), _request_paths(request),
        _cited_rejection_symbols(request, decision_evidence)
        if response.decision != SemanticAntiPatternDecisionKind.ANTI_PATTERN else frozenset(),
    )
    confidence = Confidence(score, CONFIDENCE_METHOD, None)

    if response.decision == SemanticAntiPatternDecisionKind.ANTI_PATTERN:
        if response.anti_pattern is None:
            raise ValueError("ANTI_PATTERN decision requires an Anti-pattern object")
        anti_pattern, cited = _validate_anti_pattern(
            request, response.anti_pattern, allowed_evidence
        )
        evidence_ids = tuple(sorted(set(decision_evidence) | cited))
        draft_id = hashlib.sha256(
            f"{CONFIDENCE_METHOD}\0{request.repository_id}\0{request.candidate_id}".encode()
        ).hexdigest()[:24]
        draft = SemanticAntiPatternDraft(
            draft_id=draft_id,
            repository_id=request.repository_id,
            candidate_id=request.candidate_id,
            name=anti_pattern.name,
            name_evidence_ids=anti_pattern.name_evidence_ids,
            description=anti_pattern.description,
            description_evidence_ids=anti_pattern.description_evidence_ids,
            avoid_behavior=anti_pattern.avoid_behavior,
            avoid_behavior_evidence_ids=anti_pattern.avoid_behavior_evidence_ids,
            negative_evidence_summary=anti_pattern.negative_evidence_summary,
            negative_evidence_ids=anti_pattern.negative_evidence_ids,
            scope=anti_pattern.scope,
            exceptions=anti_pattern.exceptions,
            decision_reason=reason,
            decision_evidence_ids=decision_evidence,
            evidence_ids=evidence_ids,
            confidence=confidence,
        )
        return draft, None, set(evidence_ids)

    if response.anti_pattern is not None:
        raise ValueError(f"{response.decision.value} decision must not include an Anti-pattern")
    cited_evidence = set(decision_evidence)
    if not any(
        set(signal.evidence_ids) <= cited_evidence for signal in request.signals
    ):
        raise ValueError(
            "semantic rejection must cite a complete NegativeSignal Evidence closure"
        )
    rejection = SemanticAntiPatternRejection(
        repository_id=request.repository_id,
        candidate_id=request.candidate_id,
        decision=response.decision,
        reason=reason,
        evidence_ids=decision_evidence,
        confidence=confidence,
    )
    return None, rejection, set(decision_evidence)


class SemanticAntiPatternAbstracter:
    """Interpret each selected candidate once and retain only validated outputs."""

    def abstract(
        self,
        history: GitHistoryResult,
        candidate_result: NegativePatternCandidateResult,
        model: AntiPatternSemanticModel,
        *,
        selected_candidates: tuple[NegativePatternCandidate, ...] | None = None,
    ) -> SemanticAntiPatternResult:
        repository_id = candidate_result.repository_id
        if history.repository_id != repository_id:
            raise ValueError("all inputs must describe the same repository_id")
        available = _index_unique(
            candidate_result.candidates,
            lambda item: item.candidate_id,
            "NegativePatternCandidate ID",
        )
        candidates = (
            candidate_result.candidates if selected_candidates is None else selected_candidates
        )
        selected_ids = tuple(item.candidate_id for item in candidates)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("candidate can receive only one semantic decision")
        for item in candidates:
            if available.get(item.candidate_id) != item:
                raise ValueError(
                    "candidate is not present unchanged in NegativePatternCandidateResult"
                )

        drafts: list[SemanticAntiPatternDraft] = []
        rejections: list[SemanticAntiPatternRejection] = []
        used_evidence: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item.candidate_id):
            request = build_semantic_antipattern_request(history, candidate_result, candidate)
            response = model.complete(request)
            try:
                draft, rejection, cited = _validate_decision(request, response)
            except TypeError as error:
                from skillforge.errors import SemanticResponseValidationError

                raise SemanticResponseValidationError(str(error)) from error
            except ValueError as error:
                if not isinstance(response, SemanticAntiPatternDecisionResponse):
                    from skillforge.errors import SemanticResponseValidationError

                    raise SemanticResponseValidationError(str(error)) from error
                raise SemanticDecisionValidationError(str(error)) from error
            if draft is not None:
                drafts.append(draft)
            if rejection is not None:
                rejections.append(rejection)
            used_evidence.update(cited)

        evidence_by_id = _index_unique(
            candidate_result.evidence,
            lambda item: item.evidence_id,
            "Negative candidate Evidence ID",
        )
        evidence = tuple(evidence_by_id[key] for key in sorted(used_evidence))
        return SemanticAntiPatternResult(
            repository_id,
            tuple(drafts),
            tuple(rejections),
            evidence,
        )
