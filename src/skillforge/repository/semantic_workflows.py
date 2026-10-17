"""Build candidate-scoped context and validate semantic workflow model output."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from math import isfinite
from typing import Protocol

from skillforge.domain.change import ChangePattern, ChangePatternKind, GitHistoryResult
from skillforge.domain.evidence import Confidence, Evidence
from skillforge.domain.semantic_workflow import (
    SemanticChangedFile,
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticEntityReferences,
    SemanticEvidenceRef,
    SemanticSourcePattern,
    SemanticStepClassification,
    SemanticSupportingCommit,
    SemanticWorkflowDraft,
    SemanticWorkflowException,
    SemanticWorkflowRejection,
    SemanticWorkflowRequest,
    SemanticWorkflowResponse,
    SemanticWorkflowResult,
    SemanticWorkflowRole,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.static_analysis import StaticAnalysisResult
from skillforge.domain.workflow import WorkflowCandidate, WorkflowCandidateResult
from skillforge.errors import SemanticDecisionValidationError
from skillforge.repository.semantic_references import cited_prose_filenames, cited_rejection_symbols

CONFIDENCE_METHOD = "llm_semantic_workflow_v1"

_COMMIT_TOKEN = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{7,40}(?![A-Za-z0-9])")
_PATH_TOKEN = re.compile(
    r"(?<![\w.])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]{1,8})?"
    r"|(?<![\w/])[A-Za-z0-9_.-]+\.(?:py|pyi|toml|yaml|yml|json|md|txt|rst|html|"
    r"css|js|ts|tsx|jsx|sql|sh|bat|ps1|xml|ini|cfg|csv|java|go|rs|cs|cpp|c|h|lock)"
    r"(?!\w)",
    re.IGNORECASE,
)
_MODULE_TOKEN = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")


class WorkflowSemanticModel(Protocol):
    """Thin synchronous port for a structured semantic model implementation."""

    def complete(self, request: SemanticWorkflowRequest) -> SemanticDecisionResponse:
        """Return an untrusted structured interpretation of the supplied context."""


def _index_unique(items: tuple, key, label: str) -> dict:
    indexed = {}
    for item in items:
        item_key = key(item)
        if item_key in indexed:
            raise ValueError(f"duplicate {label}: {item_key}")
        indexed[item_key] = item
    return indexed


def _evidence_reference(evidence: Evidence) -> SemanticEvidenceRef:
    return SemanticEvidenceRef(
        evidence_id=evidence.evidence_id,
        kind=evidence.kind.value,
        path=evidence.location.path,
        commit_id=evidence.location.commit,
        line=evidence.location.line,
        description=evidence.description,
        provenance=evidence.provenance,
    )


def _is_relevant_file(
    kind: ChangePatternKind,
    members: set[str],
    path: str,
    previous_path: str | None,
    module_by_path: dict[str, str],
) -> bool:
    if kind == ChangePatternKind.FILE_CO_CHANGE:
        return path in members or previous_path in members
    return (
        module_by_path.get(path) in members
        or (previous_path is not None and module_by_path.get(previous_path) in members)
    )


def build_semantic_workflow_request(
    history: GitHistoryResult,
    static_analysis: StaticAnalysisResult,
    candidate_result: WorkflowCandidateResult,
    candidate: WorkflowCandidate,
) -> SemanticWorkflowRequest:
    """Package only one Candidate, its source family, commits, and closed Evidence."""

    if candidate.kind == ChangePatternKind.STATIC_DELEGATION:
        from skillforge.repository.static_delegation import build_delegation_request

        if history.repository_id != static_analysis.repository_id:
            raise ValueError("all inputs must describe the same repository_id")
        return build_delegation_request(static_analysis, candidate_result, candidate)

    repository_id = candidate_result.repository_id
    if history.repository_id != repository_id or static_analysis.repository_id != repository_id:
        raise ValueError("all inputs must describe the same repository_id")
    if candidate.repository_id != repository_id:
        raise ValueError("candidate repository_id mismatch")
    matching_candidates = [
        item for item in candidate_result.candidates
        if item.candidate_id == candidate.candidate_id
    ]
    if len(matching_candidates) != 1 or matching_candidates[0] != candidate:
        raise ValueError("candidate is not present unchanged in WorkflowCandidateResult")
    if any(item.repository_id != repository_id for item in history.changesets):
        raise ValueError("Git history changeset repository_id mismatch")
    if any(item.repository_id != repository_id for item in static_analysis.evidence):
        raise ValueError("Static Analysis Evidence repository_id mismatch")

    changesets = _index_unique(
        history.changesets, lambda item: item.commit_id, "Git history commit_id"
    )
    patterns = _index_unique(
        candidate_result.source_patterns, lambda item: item.pattern_id, "source pattern ID"
    )
    missing_patterns = set(candidate.source_pattern_ids) - patterns.keys()
    if missing_patterns:
        missing = ", ".join(sorted(missing_patterns))
        raise ValueError(f"Missing Candidate source patterns: {missing}")
    source_patterns: list[ChangePattern] = []
    for pattern_id in sorted(candidate.source_pattern_ids):
        pattern = patterns[pattern_id]
        if pattern.repository_id != repository_id or pattern.kind != candidate.kind:
            raise ValueError(f"Candidate source pattern context mismatch: {pattern_id}")
        source_patterns.append(pattern)

    supporting_ids = tuple(sorted(set(candidate.supporting_commits)))
    if len(supporting_ids) != len(candidate.supporting_commits):
        raise ValueError("Candidate supporting commits must be unique")
    missing_commits = set(supporting_ids) - changesets.keys()
    if missing_commits:
        raise ValueError(
            "Missing supporting commits for Candidate "
            f"{candidate.candidate_id}: {', '.join(sorted(missing_commits))}"
        )

    evidence_by_id = _index_unique(
        candidate_result.evidence, lambda item: item.evidence_id, "Candidate Evidence ID"
    )
    candidate_evidence_ids = set(candidate.evidence_ids)
    missing_evidence = candidate_evidence_ids - evidence_by_id.keys()
    if missing_evidence:
        raise ValueError(
            f"Missing Candidate Evidence: {', '.join(sorted(missing_evidence))}"
        )
    for evidence_id in candidate_evidence_ids:
        evidence = evidence_by_id[evidence_id]
        if evidence.repository_id != repository_id:
            raise ValueError(f"Evidence repository_id mismatch: {evidence_id}")

    selected_input_evidence: dict[str, Evidence] = {}
    for evidence in (*history.evidence, *static_analysis.evidence):
        if evidence.evidence_id not in candidate_evidence_ids:
            continue
        previous = selected_input_evidence.setdefault(evidence.evidence_id, evidence)
        if previous != evidence:
            raise ValueError(f"Conflicting input Evidence ID: {evidence.evidence_id}")
        if evidence_by_id[evidence.evidence_id] != evidence:
            raise ValueError(f"Candidate/input Evidence mismatch: {evidence.evidence_id}")
    missing_input_evidence = candidate_evidence_ids - selected_input_evidence.keys()
    if missing_input_evidence:
        raise ValueError(
            "Missing input Evidence: " + ", ".join(sorted(missing_input_evidence))
        )

    modules_by_path = _index_unique(
        static_analysis.modules, lambda item: item.path, "Static module path"
    )
    module_by_path = {
        path: module.module
        for path, module in modules_by_path.items()
        if module.module is not None
    }

    family_ids = tuple(sorted({
        commit_id for pattern in source_patterns for commit_id in pattern.supporting_commits
    }))
    missing_family_commits = set(family_ids) - changesets.keys()
    if missing_family_commits:
        raise ValueError(
            "Missing source family supporting commits: "
            + ", ".join(sorted(missing_family_commits))
        )
    if not set(supporting_ids) <= set(family_ids):
        raise ValueError("Candidate supporting commits are outside source pattern family")

    family_supporting_commits: list[SemanticSupportingCommit] = []
    for commit_id in family_ids:
        changeset = changesets[commit_id]
        if changeset.repository_id != repository_id:
            raise ValueError(f"Supporting commit repository_id mismatch: {commit_id}")
        matching_patterns = tuple(
            pattern for pattern in source_patterns if commit_id in pattern.supporting_commits
        )
        relevant_members = {
            item.value for pattern in matching_patterns for item in pattern.items
        }
        relevant_files: list[SemanticChangedFile] = []
        for change in changeset.files:
            role_match = candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE and any(
                event.commit_id == commit_id and event.path == change.path
                for event in candidate.structural_events
            )
            if not role_match and not _is_relevant_file(
                candidate.kind, relevant_members, change.path, change.previous_path,
                module_by_path,
            ):
                continue
            missing_file_evidence = set(change.evidence_ids) - candidate_evidence_ids
            if missing_file_evidence:
                raise ValueError(
                    f"Supporting file Evidence missing from Candidate: "
                    f"{commit_id}:{change.path}"
                )
            relevant_files.append(SemanticChangedFile(
                path=change.path,
                kind=change.kind.value,
                previous_path=change.previous_path,
                additions=change.additions,
                deletions=change.deletions,
                evidence_ids=tuple(sorted(change.evidence_ids)),
            ))
        relevant_files.sort(key=lambda item: (item.path, item.kind, item.previous_path or ""))
        commit_evidence_ids = tuple(sorted(
            evidence_id for evidence_id in candidate_evidence_ids
            if evidence_by_id[evidence_id].location.commit == commit_id
        ))
        family_supporting_commits.append(SemanticSupportingCommit(
            commit_id=commit_id,
            message=changeset.message,
            relevant_changed_files=tuple(relevant_files),
            evidence_ids=commit_evidence_ids,
            source_pattern_ids=tuple(pattern.pattern_id for pattern in matching_patterns),
            supports_seed=commit_id in supporting_ids,
        ))

    evidence = tuple(
        _evidence_reference(evidence_by_id[evidence_id])
        for evidence_id in sorted(candidate_evidence_ids)
    )
    selected_evidence_ids = {item.evidence_id for item in evidence}
    semantic_patterns = tuple(SemanticSourcePattern(
        pattern_id=pattern.pattern_id,
        kind=pattern.kind,
        items=tuple(item.value for item in pattern.items),
        support=pattern.support,
        support_ratio=pattern.support_ratio,
        item_occurrences=pattern.item_occurrences,
        evidence_ids=tuple(sorted(set(pattern.evidence_ids) & selected_evidence_ids)),
        supporting_commits=pattern.supporting_commits,
    ) for pattern in source_patterns)

    return SemanticWorkflowRequest(
        repository_id=repository_id,
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        members=tuple(sorted(candidate.members, key=lambda item: item.value)),
        current_modules=tuple(sorted(set(candidate.current_modules) | {
            module_by_path[event.path] for event in candidate.structural_events
            if event.path in module_by_path
        })),
        support=candidate.support,
        support_ratio=candidate.support_ratio,
        pair_associations=tuple(sorted(
            candidate.pair_associations,
            key=lambda item: (item.first, item.second, item.source_pattern_id),
        )),
        structural_links=tuple(sorted(
            candidate.structural_links,
            key=lambda item: (item.kind.value, item.source, item.target),
        )),
        structural_coherence=candidate.structural_coherence,
        noise_hints=tuple(sorted(
            candidate.noise_signals,
            key=lambda item: (item.kind.value, item.target, item.method, item.evidence_ids),
        )),
        source_patterns=semantic_patterns,
        supporting_commits=tuple(
            item for item in family_supporting_commits if item.supports_seed
        ),
        family_supporting_commits=tuple(family_supporting_commits),
        evidence=evidence,
    )


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_evidence_ids(
    evidence_ids: object,
    allowed_evidence_ids: set[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(evidence_ids, tuple) or not evidence_ids:
        raise ValueError(f"{label} must cite at least one Evidence ID")
    if any(not isinstance(item, str) or not item for item in evidence_ids):
        raise ValueError(f"{label} contains an invalid Evidence ID")
    unknown = set(evidence_ids) - allowed_evidence_ids
    if unknown:
        raise ValueError(f"{label} cites unknown Candidate Evidence: {', '.join(sorted(unknown))}")
    return tuple(sorted(set(evidence_ids)))


def _validate_reference_list(values: object, allowed: set[str], label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"model {label} references must be a tuple")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"model {label} reference is invalid")
    if len(values) != len(set(values)):
        raise ValueError(f"model {label} references contain duplicates")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"model referenced unknown {label}: {', '.join(sorted(unknown))}")
    return values


def _text_claims(response: SemanticWorkflowResponse) -> tuple[str, ...]:
    claims = [response.name, response.intent]
    for role in response.roles:
        claims.extend((role.name, role.responsibility))
    for step in response.ordered_steps:
        claims.append(step.action)
    for group in (
        response.required_steps,
        response.optional_steps,
        response.conditional_steps,
    ):
        for item in group:
            claims.append(item.reason)
            if item.condition is not None:
                claims.append(item.condition)
    claims.extend((response.scope.description, response.scope.reason))
    claims.extend(response.scope.selectors)
    claims.extend(item.description for item in response.exceptions)
    return tuple(claims)


def _validate_text_claims(
    claims: tuple[str, ...],
    allowed_commits: set[str],
    allowed_modules: set[str],
    allowed_paths: set[str],
    rejection_symbols: frozenset[str] = frozenset(),
    mentioned_paths: frozenset[str] = frozenset(),
    cited_evidence_ids: frozenset[str] = frozenset(),
) -> None:
    for claim in claims:
        if not isinstance(claim, str):
            raise ValueError("semantic claims must be strings")
        for commit_token in _COMMIT_TOKEN.findall(claim):
            if commit_token in cited_evidence_ids:
                continue  # Exact, validated Evidence IDs are not Git commit references.
            normalized = commit_token.lower()
            if not any(commit.lower().startswith(normalized) for commit in allowed_commits):
                raise ValueError(f"model text references unknown commit: {commit_token}")
        path_tokens = _PATH_TOKEN.findall(claim)
        for path_token in path_tokens:
            # A dotted Python module can resemble a file extension (pkg.c).
            # Preserve exact known references before stripping prose punctuation.
            if path_token not in allowed_paths and path_token not in allowed_modules:
                path_token = path_token.rstrip(".")
            if (path_token not in allowed_paths and path_token not in allowed_modules
                    and path_token not in mentioned_paths):
                raise ValueError(
                    f"model text references unknown path: {path_token}; use only exact "
                    "supplied paths. Write operation alternatives with 'or' and function "
                    "names without dotted prefixes; do not invent a replacement path.")
        without_paths = _PATH_TOKEN.sub(" ", claim)
        for module_token in _MODULE_TOKEN.findall(without_paths):
            if module_token in rejection_symbols:
                continue
            if module_token.lower() in {"e.g", "i.e"}:
                continue
            if re.fullmatch(r"v?\d+(?:\.\d+)+", module_token, re.IGNORECASE):
                continue
            if module_token not in allowed_modules:
                raise ValueError(f"model text references unknown module: {module_token}")


def _validate_text_references(
    response: SemanticWorkflowResponse,
    allowed_commits: set[str],
    allowed_modules: set[str],
    allowed_paths: set[str],
    mentioned_paths: frozenset[str] = frozenset(),
    cited_evidence_ids: frozenset[str] = frozenset(),
) -> None:
    _validate_text_claims(
        _text_claims(response), allowed_commits, allowed_modules, allowed_paths,
        mentioned_paths=mentioned_paths,
        cited_evidence_ids=cited_evidence_ids,
    )


def _allowed_directories(paths: set[str]) -> set[str]:
    directories = {"."}
    for path in paths:
        if path == ".":
            continue
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _request_allowed_paths(request: SemanticWorkflowRequest) -> set[str]:
    paths = {evidence.path for evidence in request.evidence}
    if request.kind == ChangePatternKind.FILE_CO_CHANGE:
        paths.update(member.value for member in request.members)
    paths.update(
        changed_file.path
        for commit in request.family_supporting_commits
        for changed_file in commit.relevant_changed_files
    )
    paths.update(
        changed_file.previous_path
        for commit in request.family_supporting_commits
        for changed_file in commit.relevant_changed_files
        if changed_file.previous_path is not None
    )
    return paths


def _validate_scope(
    scope: SemanticWorkflowScope,
    allowed_modules: set[str],
    allowed_paths: set[str],
) -> None:
    if not isinstance(scope, SemanticWorkflowScope):
        raise ValueError("model scope has an invalid structure")
    if not isinstance(scope.kind, SemanticWorkflowScopeKind):
        raise ValueError("model scope kind is invalid")
    _nonempty(scope.description, "scope description")
    _nonempty(scope.reason, "scope reason")
    if not isinstance(scope.selectors, tuple):
        raise ValueError("scope selectors must be a tuple")
    if scope.kind in {SemanticWorkflowScopeKind.UNKNOWN, SemanticWorkflowScopeKind.REPOSITORY}:
        if scope.selectors:
            raise ValueError(f"{scope.kind.value} scope must not have selectors")
        return
    if not scope.selectors:
        raise ValueError("scope selectors must contain at least one entity")
    selectors = tuple(_nonempty(item, "scope selector") for item in scope.selectors)
    if len(selectors) != len(set(selectors)):
        raise ValueError("scope selectors contain duplicates")
    for selector in selectors:
        if scope.kind == SemanticWorkflowScopeKind.MODULE and selector not in allowed_modules:
            raise ValueError(f"scope references unknown module: {selector}")
        if scope.kind == SemanticWorkflowScopeKind.FILE and selector not in allowed_paths:
            raise ValueError(f"scope references unknown path: {selector}")
        if scope.kind == SemanticWorkflowScopeKind.DIRECTORY:
            if selector not in _allowed_directories(allowed_paths):
                raise ValueError(f"scope references unknown directory: {selector}")


def _validate_response(
    request: SemanticWorkflowRequest,
    response: SemanticWorkflowResponse,
) -> tuple[SemanticWorkflowDraft, set[str]]:
    if not isinstance(response, SemanticWorkflowResponse):
        raise ValueError("semantic model returned an invalid structured response")
    if response.repository_id != request.repository_id:
        raise ValueError("semantic response repository_id mismatch")
    if response.candidate_id != request.candidate_id:
        raise ValueError("semantic response candidate_id mismatch")

    name = _nonempty(response.name, "workflow name")
    intent = _nonempty(response.intent, "workflow intent")
    allowed_evidence_ids = {item.evidence_id for item in request.evidence}
    name_evidence_ids = _validate_evidence_ids(
        response.name_evidence_ids, allowed_evidence_ids, "workflow name"
    )
    intent_evidence_ids = _validate_evidence_ids(
        response.intent_evidence_ids, allowed_evidence_ids, "workflow intent"
    )

    if not isinstance(response.roles, tuple):
        raise ValueError("workflow roles must be a tuple")
    role_ids = []
    roles: list[SemanticWorkflowRole] = []
    referenced_evidence = set(name_evidence_ids) | set(intent_evidence_ids)
    for role in response.roles:
        if not isinstance(role, SemanticWorkflowRole):
            raise ValueError("workflow role has an invalid structure")
        role_id = _nonempty(role.role_id, "role ID")
        role_ids.append(role_id)
        normalized = SemanticWorkflowRole(
            role_id,
            _nonempty(role.name, "role name"),
            _nonempty(role.responsibility, "role responsibility"),
            _validate_evidence_ids(role.evidence_ids, allowed_evidence_ids, "workflow role"),
        )
        referenced_evidence.update(normalized.evidence_ids)
        roles.append(normalized)
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("duplicate workflow role ID")

    if not isinstance(response.ordered_steps, tuple) or not response.ordered_steps:
        raise ValueError("ordered_steps must be a non-empty tuple")
    steps: list[SemanticWorkflowStep] = []
    step_ids = []
    for step in response.ordered_steps:
        if not isinstance(step, SemanticWorkflowStep):
            raise ValueError("workflow step has an invalid structure")
        step_id = _nonempty(step.step_id, "step ID")
        step_ids.append(step_id)
        normalized = SemanticWorkflowStep(
            step_id,
            _nonempty(step.action, "step action"),
            _validate_evidence_ids(step.evidence_ids, allowed_evidence_ids, "workflow step"),
        )
        referenced_evidence.update(normalized.evidence_ids)
        steps.append(normalized)
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("duplicate step ID")
    known_steps = set(step_ids)

    classified: set[str] = set()

    def validate_classifications(
        items: object, label: str
    ) -> tuple[SemanticStepClassification, ...]:
        if not isinstance(items, tuple):
            raise ValueError(f"{label} must be a tuple")
        normalized_items = []
        local_step_ids = []
        for item in items:
            if not isinstance(item, SemanticStepClassification):
                raise ValueError(f"{label} item has an invalid structure")
            step_id = _nonempty(item.step_id, f"{label} step ID")
            if step_id not in known_steps:
                raise ValueError(f"{label} references unknown step: {step_id}")
            local_step_ids.append(step_id)
            if step_id in classified:
                raise ValueError(
                    "step appears in multiple required/optional/conditional groups: "
                    f"{step_id}"
                )
            if label == "conditional_steps":
                condition = _nonempty(item.condition, f"{label} condition")
            elif item.condition is not None:
                raise ValueError(f"{label} condition must be None")
            else:
                condition = None
            normalized = SemanticStepClassification(
                step_id,
                _nonempty(item.reason, f"{label} reason"),
                _validate_evidence_ids(item.evidence_ids, allowed_evidence_ids, label),
                condition,
            )
            referenced_evidence.update(normalized.evidence_ids)
            normalized_items.append(normalized)
        if len(local_step_ids) != len(set(local_step_ids)):
            raise ValueError(f"duplicate step in {label}")
        classified.update(local_step_ids)
        return tuple(normalized_items)

    required_steps = validate_classifications(response.required_steps, "required_steps")
    optional_steps = validate_classifications(response.optional_steps, "optional_steps")
    conditional_steps = validate_classifications(
        response.conditional_steps, "conditional_steps"
    )
    if classified != known_steps:
        raise ValueError(
            "ordered_steps missing classification: "
            + ", ".join(sorted(known_steps - classified))
        )

    scope = response.scope
    _validate_scope(scope, set(request.current_modules), _request_allowed_paths(request))
    scope_evidence_ids = _validate_evidence_ids(
        scope.evidence_ids, allowed_evidence_ids, "workflow scope"
    )
    scope = SemanticWorkflowScope(
        scope.kind,
        tuple(sorted(item.strip() for item in scope.selectors)),
        _nonempty(scope.description, "scope description"),
        _nonempty(scope.reason, "scope reason"),
        scope_evidence_ids,
    )
    referenced_evidence.update(scope.evidence_ids)

    if not isinstance(response.exceptions, tuple):
        raise ValueError("workflow exceptions must be a tuple")
    exceptions: list[SemanticWorkflowException] = []
    for exception in response.exceptions:
        if not isinstance(exception, SemanticWorkflowException):
            raise ValueError("workflow exception has an invalid structure")
        normalized = SemanticWorkflowException(
            _nonempty(exception.description, "exception description"),
            _validate_evidence_ids(
                exception.evidence_ids, allowed_evidence_ids, "workflow exception"
            ),
        )
        referenced_evidence.update(normalized.evidence_ids)
        exceptions.append(normalized)

    if isinstance(response.confidence_score, bool) or not isinstance(
        response.confidence_score, (int, float)
    ):
        raise ValueError("Confidence score must be numeric")
    score = float(response.confidence_score)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0 and 1")

    if not isinstance(response.references, SemanticEntityReferences):
        raise ValueError("semantic entity references have an invalid structure")
    allowed_paths = _request_allowed_paths(request)
    allowed_modules = set(request.current_modules) | {
        member.current_module for member in request.members
        if member.current_module is not None
    }
    allowed_commits = {item.commit_id for item in request.family_supporting_commits} | {
        item.commit_id for item in request.evidence if item.commit_id is not None
    }
    _validate_reference_list(response.references.commit_ids, allowed_commits, "commit")
    _validate_reference_list(response.references.module_names, allowed_modules, "module")
    _validate_reference_list(
        response.references.paths,
        allowed_paths | _allowed_directories(allowed_paths),
        "path",
    )
    _validate_text_references(
        response, allowed_commits, allowed_modules, allowed_paths,
        cited_prose_filenames(request.evidence, request.family_supporting_commits,
                              referenced_evidence, _PATH_TOKEN),
        cited_evidence_ids=frozenset(referenced_evidence),
    )

    draft_id = hashlib.sha256(
        f"{CONFIDENCE_METHOD}\0{request.repository_id}\0{request.candidate_id}".encode()
    ).hexdigest()[:24]
    draft = SemanticWorkflowDraft(
        draft_id=draft_id,
        repository_id=request.repository_id,
        candidate_id=request.candidate_id,
        name=name,
        name_evidence_ids=name_evidence_ids,
        intent=intent,
        intent_evidence_ids=intent_evidence_ids,
        roles=tuple(roles),
        ordered_steps=tuple(steps),
        required_steps=required_steps,
        optional_steps=optional_steps,
        conditional_steps=conditional_steps,
        scope=scope,
        exceptions=tuple(exceptions),
        evidence_ids=tuple(sorted(referenced_evidence)),
        support=request.support,
        confidence=Confidence(score, CONFIDENCE_METHOD, request.support),
    )
    return draft, referenced_evidence


def _validate_decision(
    request: SemanticWorkflowRequest,
    response: SemanticDecisionResponse,
) -> tuple[SemanticWorkflowDraft | None, SemanticWorkflowRejection | None, set[str]]:
    if not isinstance(response, SemanticDecisionResponse):
        raise ValueError("semantic model returned an invalid decision response")
    if response.repository_id != request.repository_id:
        raise ValueError("semantic response repository_id mismatch")
    if response.candidate_id != request.candidate_id:
        raise ValueError("semantic response candidate_id mismatch")
    if not isinstance(response.decision, SemanticDecisionKind):
        raise ValueError("semantic decision is invalid")
    if response.support != request.support:
        raise ValueError("semantic decision support mismatch")
    if isinstance(response.confidence_score, bool) or not isinstance(
        response.confidence_score, (int, float)
    ):
        raise ValueError("Confidence score must be numeric")
    score = float(response.confidence_score)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0 and 1")
    reason = _nonempty(response.reason, "semantic decision reason")
    evidence_ids = _validate_evidence_ids(
        response.decision_evidence_ids,
        {item.evidence_id for item in request.evidence},
        "semantic decision",
    )
    allowed_paths = _request_allowed_paths(request)
    allowed_commits = {item.commit_id for item in request.family_supporting_commits} | {
        item.commit_id for item in request.evidence if item.commit_id is not None
    }
    _validate_text_claims(
        (reason,), allowed_commits, set(request.current_modules), allowed_paths,
        cited_rejection_symbols(
            request.evidence, request.family_supporting_commits, evidence_ids, _PATH_TOKEN
        ) if response.decision != SemanticDecisionKind.WORKFLOW else frozenset(),
        mentioned_paths=cited_prose_filenames(
            request.evidence, request.family_supporting_commits, evidence_ids, _PATH_TOKEN,
        ),
        cited_evidence_ids=frozenset(evidence_ids),
    )
    if response.decision == SemanticDecisionKind.WORKFLOW:
        if response.workflow is None:
            raise ValueError("WORKFLOW decision requires a complete workflow")
        if response.workflow.confidence_score != score:
            raise ValueError("workflow and decision confidence mismatch")
        draft, cited = _validate_response(request, response.workflow)
        all_evidence = set(evidence_ids) | cited
        draft = replace(draft, evidence_ids=tuple(sorted(all_evidence)))
        return draft, None, all_evidence
    if response.workflow is not None:
        raise ValueError(f"{response.decision.value} decision must not include a workflow")
    rejection = SemanticWorkflowRejection(
        repository_id=request.repository_id,
        candidate_id=request.candidate_id,
        decision=response.decision,
        reason=reason,
        evidence_ids=evidence_ids,
        confidence=Confidence(score, CONFIDENCE_METHOD, request.support),
        support=request.support,
    )
    return None, rejection, set(evidence_ids)


class SemanticWorkflowAbstracter:
    """Ask a model to interpret each Candidate, then accept only validated drafts."""

    def abstract(
        self,
        history: GitHistoryResult,
        static_analysis: StaticAnalysisResult,
        candidate_result: WorkflowCandidateResult,
        model: WorkflowSemanticModel,
        *,
        selected_candidates: tuple[WorkflowCandidate, ...] | None = None,
    ) -> SemanticWorkflowResult:
        repository_id = candidate_result.repository_id
        if history.repository_id != repository_id or static_analysis.repository_id != repository_id:
            raise ValueError("all inputs must describe the same repository_id")

        drafts: list[SemanticWorkflowDraft] = []
        rejections: list[SemanticWorkflowRejection] = []
        evidence_ids: set[str] = set()
        candidates = (
            candidate_result.candidates if selected_candidates is None else selected_candidates
        )
        for candidate in sorted(candidates, key=lambda item: item.candidate_id):
            request = build_semantic_workflow_request(
                history, static_analysis, candidate_result, candidate
            )
            response = model.complete(request)
            try:
                draft, rejection, cited = _validate_decision(request, response)
            except TypeError as error:
                from skillforge.errors import SemanticResponseValidationError

                raise SemanticResponseValidationError(str(error)) from error
            except ValueError as error:
                if not isinstance(response, SemanticDecisionResponse):
                    from skillforge.errors import SemanticResponseValidationError

                    raise SemanticResponseValidationError(str(error)) from error
                raise SemanticDecisionValidationError(str(error)) from error
            if draft is not None:
                drafts.append(draft)
            if rejection is not None:
                rejections.append(rejection)
            evidence_ids.update(cited)

        evidence_by_id = _index_unique(
            candidate_result.evidence,
            lambda item: item.evidence_id,
            "Candidate Evidence ID",
        )
        evidence = tuple(evidence_by_id[key] for key in sorted(evidence_ids))
        return SemanticWorkflowResult(repository_id, tuple(drafts), evidence, tuple(rejections))
