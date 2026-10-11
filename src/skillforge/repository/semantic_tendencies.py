"""Thin history convention gate using the existing candidate/Evidence adapter."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Protocol

from skillforge.domain.change import ChangePatternKind, FileChangeKind
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.semantic_tendency import (
    ScopedTendencyDecision,
    SemanticTendencyAcceptance,
    SemanticTendencyRequest,
    SemanticTendencyResponse,
    SemanticTendencyResult,
    TendencyDecisionKind,
)
from skillforge.domain.semantic_workflow import SemanticChangedFile, SemanticSupportingCommit
from skillforge.errors import SemanticResponseValidationError
from skillforge.repository.semantic_structural_rules import _POLICY
from skillforge.repository.semantic_workflows import (
    _evidence_reference,
    _validate_text_claims,
    build_semantic_workflow_request,
)


class TendencySemanticModel(Protocol):
    def complete(self, request: SemanticTendencyRequest
                 ) -> ScopedTendencyDecision | SemanticTendencyResponse: ...


def tendency_request_fingerprint(request: SemanticTendencyRequest) -> str:
    return hashlib.sha256(json.dumps(
        request.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def bind_tendency_decision(request: SemanticTendencyRequest, result):
    if isinstance(result, SemanticTendencyResponse):
        # Existing programmatic model implementations already supply bound responses.
        # Their identity remains strictly validated; model echoes use the scoped port.
        return result
    if (not isinstance(result, ScopedTendencyDecision)
            or result.request_fingerprint != tendency_request_fingerprint(request)):
        raise ValueError("tendency response belongs to a different request")
    decision = result.decision
    refs = decision.evidence_ids
    if decision.citation_choices is not None:
        choices = dict(decision.citation_choices)
        expected = {f"e{i}_id" for i in range(len(request.structural_events))}
        if len(choices) != len(decision.citation_choices) or set(choices) != expected:
            raise ValueError("tendency citations do not match the original event slots")
        sources = {e.evidence_id: e for e in request.evidence}
        selected = []
        for i, event in enumerate(request.structural_events):
            key = choices[f"e{i}_id"]
            if key is None:
                if decision.decision == TendencyDecisionKind.SUPPORTED_TENDENCY:
                    raise ValueError("supported tendency requires an explicit citation for every event")
                continue
            if (key not in event.evidence_ids or key not in sources or sources[key].provenance != "git_diff"
                    or sources[key].path != event.path or sources[key].commit_id != event.commit_id):
                raise ValueError("tendency citation does not belong to its original event")
            selected.append(key)
        if len(refs) != 1 or refs[0] not in sources:
            raise ValueError("tendency decision anchor must be original Evidence")
        # Canonical serialization of a newly selected unordered map, never a
        # repair of legacy arrays or saved failed model responses.
        refs = tuple(sorted(set((*refs, *selected))))
    warnings = []
    for label, echoed, expected in (
        ("repository_id", decision.echoed_repository_id, request.context.repository_id),
        ("candidate_id", decision.echoed_candidate_id, request.context.candidate_id),
    ):
        if echoed is not None and echoed != expected:
            warnings.append(f"Ignored model {label} echo {echoed!r}; caller bound {expected!r}")
    return SemanticTendencyResponse(
        request.context.repository_id, request.context.candidate_id,
        decision.decision, decision.reason, refs, tuple(warnings),
    )


def build_semantic_tendency_request(history, facts, candidates, candidate):
    if candidate.kind != ChangePatternKind.ADDED_PYTHON_MODULE_ROLE:
        raise ValueError("history tendency requires structural-role candidate")
    context = build_semantic_workflow_request(history, facts, candidates, candidate)
    events = candidate.structural_events
    directories = {str(PurePosixPath(event.path).parent) for event in events}
    if len(directories) != 1:
        raise ValueError("structural events require a common parent directory")
    directory, = directories
    initializer = directory + "/__init__.py"
    modules = [item.module for item in facts.modules
               if item.path == initializer and item.module is not None]
    if len(modules) != 1:
        raise ValueError("parent package is not supported by current Python facts")
    package, = modules
    role = f"ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER({package})"
    if tuple(item.value for item in candidate.members) != (role,):
        raise ValueError("candidate normalized role differs from parent package")
    commits = {item.commit_id: item for item in history.changesets}
    sources = {item.evidence_id: item for item in (*history.evidence, *facts.evidence)}
    for event in events:
        change = next((item for item in commits[event.commit_id].files
                       if item.path == event.path), None)
        if change is None or change.kind != FileChangeKind.ADDED:
            raise ValueError("structural event is not a concrete Git addition")
        if (not event.path.endswith(".py") or event.path == initializer
                or not set(change.evidence_ids) <= set(event.evidence_ids)):
            raise ValueError("structural event lacks added module Evidence")
        if not any(sources[key].provenance == "git_parent_package"
                   and sources[key].location.path == initializer
                   and sources[key].location.commit == event.commit_id
                   for key in event.evidence_ids):
            raise ValueError("structural event lacks parent package Evidence")
    # Include every change in the observed package, not just the positive events.
    # Existing module edits/deletions are visible; they are not automatically exceptions.
    examined = []
    refs = set(candidate.evidence_ids)
    supporting = set(candidate.supporting_commits)
    for commit in history.changesets:
        files = []
        for change in commit.files:
            if not any(PurePosixPath(path).is_relative_to(directory)
                       for path in (change.path, change.previous_path) if path):
                continue
            refs.update(change.evidence_ids)
            files.append(SemanticChangedFile(
                change.path, change.kind.value, change.previous_path,
                change.additions, change.deletions, tuple(sorted(change.evidence_ids)),
            ))
        examined.append(SemanticSupportingCommit(
            commit.commit_id, commit.message, tuple(sorted(files, key=lambda item: item.path)),
            tuple(sorted({key for item in files for key in item.evidence_ids})),
            (), commit.commit_id in supporting,
        ))
    return SemanticTendencyRequest(
        context, role, package, directory, events,
        tuple(sorted(examined, key=lambda item: item.commit_id)),
        tuple(_evidence_reference(sources[key]) for key in sorted(refs)),
    )


def validate_semantic_tendency_response(request, response):
    context = request.context
    if not isinstance(response, SemanticTendencyResponse):
        raise ValueError("invalid tendency response")
    if (response.repository_id != context.repository_id
            or response.candidate_id != context.candidate_id):
        raise ValueError("tendency response identity mismatch")
    if not isinstance(response.decision, TendencyDecisionKind):
        raise ValueError("invalid tendency decision")
    allowed = {item.evidence_id for item in request.evidence}
    if (not response.evidence_ids
            or response.evidence_ids != tuple(sorted(set(response.evidence_ids)))
            or not set(response.evidence_ids) <= allowed):
        raise ValueError("tendency decision requires known sorted Evidence IDs")
    if not isinstance(response.reason, str) or not response.reason.strip():
        raise ValueError("tendency reason is required")
    if _POLICY.search(response.reason):
        raise ValueError("tendency decision attempts to introduce policy")
    _validate_text_claims(
        (response.reason,), {item.commit_id for item in request.examined_changes},
        set(context.current_modules) | {request.parent_package},
        {item.path for item in request.evidence} | {request.structural_scope},
    )
    if response.decision != TendencyDecisionKind.SUPPORTED_TENDENCY:
        return response
    events = request.structural_events
    if (len({item.commit_id for item in events}) < 3
            or len({item.path for item in events}) < 3
            or len(context.supporting_commits) < 3):
        raise ValueError("tendency requires multiple independent commits and paths")
    if context.support.supporting != len(context.supporting_commits) or {
        item.commit_id for item in context.supporting_commits
    } != {item.commit_id for item in events}:
        raise ValueError("tendency support and events mismatch")
    if context.kind != ChangePatternKind.ADDED_PYTHON_MODULE_ROLE or not request.parent_package:
        raise ValueError("tendency requires structural role and local scope")
    if request.normalized_role != (
        f"ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER({request.parent_package})"
    ) or any(str(PurePosixPath(item.path).parent) != request.structural_scope for item in events):
        raise ValueError("tendency events have different structural roles")
    cited = set(response.evidence_ids)
    evidence = {item.evidence_id: item for item in request.evidence}
    for event in events:
        if not set(event.evidence_ids) <= allowed or not any(
            evidence[key].provenance == "git_diff" and evidence[key].path == event.path
            and evidence[key].commit_id == event.commit_id for key in event.evidence_ids
        ):
            raise ValueError("tendency event lacks concrete diff Evidence")
        if not any(evidence[key].provenance == "git_parent_package"
                   and evidence[key].path == request.structural_scope + "/__init__.py"
                   and evidence[key].commit_id == event.commit_id for key in event.evidence_ids):
            raise ValueError("tendency scope lacks parent package Evidence")
        if not any(key in cited and evidence[key].provenance == "git_diff"
                   and evidence[key].path == event.path
                   and evidence[key].commit_id == event.commit_id for key in event.evidence_ids):
            raise ValueError("tendency decision must cite every supporting event")
    refs = tuple(sorted(allowed))
    # Canonical wording prevents the model from widening scope or inventing obligations.
    statement = (
        f"Observed historical tendency in {request.structural_scope}: "
        "multiple independent changes added distinct direct child Python modules. "
        "This observation is limited to the supplied history; it does not establish "
        "a universal choice for new capabilities. Other package changes are retained "
        "in the examined Evidence."
    )
    rule = ProjectRule(
        context.candidate_id, context.repository_id, RuleType.CONVENTION,
        statement, ProjectScope(ProjectScopeKind.DIRECTORY, (request.structural_scope,)),
        ProjectKnowledgeStatus.CANDIDATE, refs, context.support,
        None,
    )
    return SemanticTendencyAcceptance(request, response, rule)


class SemanticTendencyEvaluator:
    def evaluate(self, history, facts, candidates, model, *, selected_candidates):
        accepted, rejected = [], []
        sources = {item.evidence_id: item for item in (*history.evidence, *facts.evidence)}
        for candidate in selected_candidates:
            request = build_semantic_tendency_request(history, facts, candidates, candidate)
            raw_response = model.complete(request)
            try:
                response = bind_tendency_decision(request, raw_response)
                decision = validate_semantic_tendency_response(request, response)
            except (ValueError, TypeError) as error:
                raise SemanticResponseValidationError(str(error)) from error
            if isinstance(decision, SemanticTendencyAcceptance):
                accepted.append(decision)
            else:
                rejected.append(decision)
        refs = {key for item in accepted for key in item.rule.evidence_ids}
        refs.update(key for item in rejected for key in item.evidence_ids)
        return SemanticTendencyResult(
            candidates.repository_id, tuple(accepted), tuple(rejected),
            tuple(sources[key] for key in sorted(refs)),
        )
