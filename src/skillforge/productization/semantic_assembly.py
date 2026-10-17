"""Assemble independent validated semantic results without weakening their gates."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.evidence_merge import merge_evidence as _merge_evidence
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_tendency import SemanticTendencyResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.domain.serialization import to_json_value
from skillforge.errors import SemanticDecisionValidationError, SemanticResponseValidationError
from skillforge.repository.codex_cli_semantic import CodexCliSemanticError
from skillforge.repository.semantic_antipatterns import (
    SemanticAntiPatternAbstracter,
    build_semantic_antipattern_request,
)
from skillforge.repository.semantic_batch import (
    PrefetchedSemanticModel,
    prefetch_semantic_batch,
    request_id,
)
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
)
from skillforge.repository.semantic_tendencies import (
    SemanticTendencyEvaluator,
    build_semantic_tendency_request,
)
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    build_semantic_workflow_request,
)

ResultT = TypeVar("ResultT")


def _retry_invalid_decision(evaluate: Callable[[], ResultT], observed=None) -> tuple[ResultT, int]:
    """Preserve the original one retry for typed structured decision failures only."""
    try:
        return evaluate(), 0
    except SemanticDecisionValidationError as error:
        if observed is not None:
            observed.note_validation_failure(error)
        return evaluate(), 1


class _ModelFailure(RuntimeError):
    def __init__(self, error, status):
        super().__init__(str(error))
        self.status = status


class _ObservedModel:
    """Bound model-only failures; never catch request construction or assembly errors."""

    def __init__(self, model):
        self.model = model
        self.attempts = []
        self.status = "INVALID"
        self.used_batch = False
        self.validation_feedback = ""
        self.last_request = None

    def note_validation_failure(self, error):
        self.validation_feedback = str(error)
        model, seen = self.model, set()
        while model is not None and id(model) not in seen:
            seen.add(id(model))
            remember = getattr(model, "remember_validation_feedback", None)
            if callable(remember) and self.last_request is not None:
                remember(self.last_request, error)
                return
            model = getattr(model, "model", None)

    def complete(self, request):
        if self.validation_feedback and hasattr(request, "validation_feedback"):
            # Feedback is diagnostic data, never new evidence or authority. Keep the
            # original request and validators; only the existing retry receives it.
            feedback = self.validation_feedback.encode("utf-8")[:512].decode(
                "utf-8", errors="ignore")
            request = replace(request, validation_feedback=feedback)
        self.last_request = request
        self.status = "INVALID"
        attempt = {"request": to_json_value(request), "raw_response": None}
        self.attempts.append(attempt)
        before = getattr(self.model, "batch_attempts", 0)
        try:
            response = self.model.complete(request)
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            invalid = isinstance(error, (ValueError, TypeError)) or (
                isinstance(error, CodexCliSemanticError) and error.invalid_response
            )
            self.status = "INVALID" if invalid else "CHANNEL_FAILED"
            attempt["raw_response"] = to_json_value(getattr(error, "raw_response", None))
            attempt["model_error"] = str(error)
            raise _ModelFailure(error, self.status) from error
        finally:
            self.used_batch = getattr(self.model, "batch_attempts", 0) > before
        # Non-JSON objects are retained as their type/repr, never repaired or accepted.
        raw = to_json_value(response)
        if not isinstance(raw, (dict, list, str, int, float, bool, type(None))):
            raw = {"invalid_type": type(response).__name__, "repr": repr(response)}
        attempt["raw_response"] = raw
        return response


@dataclass
class SemanticAssembly:
    workflows: SemanticWorkflowResult | None = None
    tendencies: SemanticTendencyResult | None = None
    anti_patterns: SemanticAntiPatternResult | None = None
    structural: SemanticStructuralRuleResult | None = None
    failures: list[dict] = field(default_factory=list)
    workflow_retry_count: int = 0
    anti_pattern_retry_count: int = 0


def _failure(channel, candidate_id, reason, observed, retries):
    if isinstance(reason, SemanticDecisionValidationError):
        observed.note_validation_failure(reason)
    return {
        "channel": channel,
        "candidate_id": candidate_id,
        "status": observed.status,
        "reason": str(reason),
        "retry_count": retries,
        "attempts": observed.attempts,
    }


def _evaluate_candidate(evaluate, observed, *, retry, preliminary=None):
    """A failed batch candidate enters the full unchanged single retry path."""
    if preliminary is not None:
        result, error = preliminary
        if error is None:
            return result, 0
        if retry and isinstance(error, SemanticDecisionValidationError):
            observed.note_validation_failure(error)
            return evaluate(), 1
        result, retries = _retry_invalid_decision(evaluate, observed) if retry else (evaluate(), 0)
        return result, retries + 1
    if isinstance(observed.model, PrefetchedSemanticModel):
        try:
            return evaluate(), 0
        except (
            SemanticDecisionValidationError,
            SemanticResponseValidationError,
            _ModelFailure,
        ) as error:
            if retry and isinstance(error, SemanticDecisionValidationError):
                # A typed batch decision replaces the first original attempt. Preserve
                # exactly its one existing deterministic-decision retry, via single transport.
                observed.note_validation_failure(error)
                return evaluate(), 1
            if not observed.used_batch:
                raise
            result, retries = (
                _retry_invalid_decision(evaluate, observed) if retry else (evaluate(), 0)
            )
            return result, retries + 1
    return _retry_invalid_decision(evaluate, observed) if retry else (evaluate(), 0)


def _checkpoint_validated_candidate(observed, candidate_id, result, fields):
    """Commit only after this candidate has a terminal accepted/rejected result."""
    if candidate_id in getattr(observed, "checkpointed_ids", ()):
        return
    terminal_ids = set()
    for name in fields:
        for item in getattr(result, name, ()):
            terminal_id = getattr(item, "candidate_id", None)
            if terminal_id is None:
                terminal_id = getattr(getattr(item, "response", None), "candidate_id", None)
            if terminal_id is not None:
                terminal_ids.add(terminal_id)
    if candidate_id not in terminal_ids:
        return
    model = observed.model
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        checkpoint = getattr(model, "checkpoint_candidate", None)
        if callable(checkpoint):
            checkpoint(candidate_id)
            observed.checkpointed_ids = {*getattr(observed, "checkpointed_ids", ()), candidate_id}
            return
        model = getattr(model, "model", None)


def _prefetch_candidate_repairs(preliminary, requests, models):
    """Share existing fallback transports; keep each candidate's retry allowance.

    Only already observed batch failures qualify. Missing envelopes and opaque
    transports retain their original paths. Valid siblings were checkpointed by
    the first sweep and are never sent to a repair transport.
    """
    groups = {}
    for channel, items in requests.items():
        model = models[channel]
        if not isinstance(model, PrefetchedSemanticModel):
            continue
        repair = []
        for request in items:
            candidate_id = getattr(request, "context", request).candidate_id
            prior = preliminary.get((channel, candidate_id))
            if prior is None:
                continue
            observed, result, error = prior
            if error is None and not getattr(result, "failures", ()):
                continue
            if observed.validation_feedback and hasattr(request, "validation_feedback"):
                request = replace(request, validation_feedback=
                    observed.validation_feedback.encode("utf-8")[:512].decode(
                        "utf-8", errors="ignore"))
            repair.append(request)
        if repair:
            groups[channel] = (model.model, tuple(repair))
    if sum(len(items) for _, items in groups.values()) < 2:
        return
    budgets = [getattr(getattr(model, "model", model), "effect_budget", None)
               for model, _ in groups.values()]
    if any(getattr(budget, "cold_shape", None) is not None for budget in budgets):
        # Exact-shape campaign plans describe the complete cold cohort. Do not
        # reinterpret their declared population as a partial repair cohort.
        return
    from skillforge.repository.semantic_cohort import prefetch_semantic_cohort

    repaired = prefetch_semantic_cohort(groups)
    if repaired is None:
        repaired = {
            channel: prefetch_semantic_batch(model, items)
            for channel, (model, items) in groups.items()
        }
    for channel, (_, items) in groups.items():
        for request in items:
            candidate_id = getattr(request, "context", request).candidate_id
            # _evaluate_candidate still performs the original full validation,
            # counts the same retry, and retains both raw attempts on failure.
            preliminary[channel, candidate_id][0].model = repaired[channel]


def evaluate_semantic_knowledge(
    repository_id,
    facts,
    mined,
    structural_model,
    *,
    history=None,
    candidates=None,
    selected_workflows=(),
    negative=None,
    selected_anti_patterns=(),
    workflow_model=None,
    tendency_model=None,
    anti_pattern_model=None,
) -> SemanticAssembly:
    """Only response/model failures are local; input and aggregate integrity are fatal."""
    inputs = [item for item in (facts, mined, history, candidates, negative) if item is not None]
    if any(item.repository_id != repository_id for item in inputs):
        raise ValueError("Semantic assembly repository identity mismatch")
    # A corrupted Evidence namespace is fatal even when no candidate is selected.
    _merge_evidence(inputs)
    for item in inputs:
        if any(e.repository_id != repository_id for e in item.evidence):
            raise ValueError("Semantic assembly Evidence repository identity mismatch")
    # Build the same closed requests used below, without changing candidate selection.
    # Any input corruption is still fatal, before a remote batch can be dispatched.
    workflow_requests, tendency_requests, anti_requests = [], [], []
    if history is not None:
        for candidate in sorted(selected_workflows, key=lambda item: item.candidate_id):
            tendency = candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
            builder = (
                build_semantic_tendency_request if tendency else build_semantic_workflow_request
            )
            (tendency_requests if tendency else workflow_requests).append(
                builder(history, facts, candidates, candidate),
            )
        anti_requests = [
            build_semantic_antipattern_request(history, negative, candidate)
            for candidate in sorted(selected_anti_patterns, key=lambda item: item.candidate_id)
        ]
    structural_requests = [build_semantic_structural_request(mined, c) for c in mined.candidates]
    # Opt-in campaign governance operates before ANY prefetch transport. Normal
    # production candidate selection and semantic acceptance are unchanged.
    counts = {
        "workflow": len(workflow_requests),
        "tendency": len(tendency_requests),
        "anti-pattern": len(anti_requests),
        "structural": len(structural_requests),
    }
    budgets = []
    for model in (workflow_model, tendency_model, anti_pattern_model, structural_model):
        original = getattr(model, "model", model)
        budget = getattr(original, "effect_budget", None)
        if budget is not None and budget not in budgets:
            budgets.append(budget)
    for budget in budgets:
        budget.validate_shape(counts)
    from skillforge.repository.semantic_cohort import prefetch_semantic_cohort

    cohort = prefetch_semantic_cohort({
        "workflow": (workflow_model, workflow_requests),
        "tendency": (tendency_model, tendency_requests),
        "anti-pattern": (anti_pattern_model, anti_requests),
        "structural": (structural_model, structural_requests),
    })
    if cohort is not None:
        workflow_model = cohort.get("workflow", workflow_model)
        tendency_model = cohort.get("tendency", tendency_model)
        anti_pattern_model = cohort.get("anti-pattern", anti_pattern_model)
        structural_model = cohort.get("structural", structural_model)
    assembly = SemanticAssembly()
    # Validate/checkpoint each returned batch before the next channel transport,
    # and finish this sweep before ANY repair/single fallback. Valid siblings
    # survive an abort while fetching a later channel, not only during repairs.
    # Reuse evaluator results/observations below instead of validating them twice.
    preliminary = {}

    def inspect(channel, candidate, request, model, evaluate, fields):
        if (
            not isinstance(model, PrefetchedSemanticModel)
            or request_id(request) not in model.responses
        ):
            return
        observed = _ObservedModel(model)
        try:
            result = evaluate(observed)
            error = None
        except (
            SemanticDecisionValidationError,
            SemanticResponseValidationError,
            _ModelFailure,
        ) as exc:
            result, error = None, exc
            if isinstance(exc, SemanticDecisionValidationError):
                observed.note_validation_failure(exc)
        if result is not None:
            _checkpoint_validated_candidate(observed, candidate.candidate_id, result, fields)
        preliminary[channel, candidate.candidate_id] = (observed, result, error)

    if history is not None:
        by_workflow = {r.candidate_id: r for r in workflow_requests}
        by_tendency = {r.context.candidate_id: r for r in tendency_requests}
        for tendency in (False, True):
            if tendency:
                tendency_model = prefetch_semantic_batch(tendency_model, tendency_requests)
            else:
                workflow_model = prefetch_semantic_batch(workflow_model, workflow_requests)
            for candidate in sorted(selected_workflows, key=lambda item: item.candidate_id):
                if (candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE) != tendency:
                    continue
                inspect(
                    "tendency" if tendency else "workflow",
                    candidate,
                    (by_tendency if tendency else by_workflow)[candidate.candidate_id],
                    tendency_model if tendency else workflow_model,
                    lambda observed, candidate=candidate, tendency=tendency: (
                        SemanticTendencyEvaluator().evaluate(
                            history, facts, candidates, observed, selected_candidates=(candidate,)
                        )
                        if tendency
                        else SemanticWorkflowAbstracter().abstract(
                            history, facts, candidates, observed, selected_candidates=(candidate,)
                        )
                    ),
                    ("workflows", "accepted", "rejections"),
                )
        anti_pattern_model = prefetch_semantic_batch(anti_pattern_model, anti_requests)
        by_anti = {r.candidate_id: r for r in anti_requests}
        for candidate in sorted(selected_anti_patterns, key=lambda item: item.candidate_id):
            inspect(
                "anti-pattern",
                candidate,
                by_anti[candidate.candidate_id],
                anti_pattern_model,
                lambda observed, candidate=candidate: SemanticAntiPatternAbstracter().abstract(
                    history, negative, observed, selected_candidates=(candidate,)
                ),
                ("anti_patterns", "rejections"),
            )
    structural_model = prefetch_semantic_batch(structural_model, structural_requests)
    for candidate, request in zip(mined.candidates, structural_requests, strict=True):
        scoped = replace(
            mined,
            candidates=(candidate,),
            evidence=tuple(
                item for item in mined.evidence if item.evidence_id in candidate.rule.evidence_ids
            ),
        )
        inspect(
            "structural",
            candidate,
            request,
            structural_model,
            lambda observed, scoped=scoped: SemanticStructuralRuleEvaluator().evaluate(
                scoped, observed
            ),
            ("accepted", "rejections"),
        )

    _prefetch_candidate_repairs(
        preliminary,
        {"workflow": workflow_requests, "tendency": tendency_requests,
         "anti-pattern": anti_requests, "structural": structural_requests},
        {"workflow": workflow_model, "tendency": tendency_model,
         "anti-pattern": anti_pattern_model, "structural": structural_model},
    )

    def observation(channel, candidate_id, model):
        prior = preliminary.get((channel, candidate_id))
        return prior[0] if prior else _ObservedModel(model)

    def prior_result(channel, candidate_id):
        prior = preliminary.get((channel, candidate_id))
        return prior[1:] if prior else None

    if history is not None:
        workflow_results, tendency_results, anti_results = [], [], []
        for candidate in sorted(selected_workflows, key=lambda item: item.candidate_id):
            tendency = candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
            channel = "tendency" if tendency else "workflow"
            observed = observation(
                channel, candidate.candidate_id, tendency_model if tendency else workflow_model
            )

            def evaluate(candidate=candidate, tendency=tendency, observed=observed):
                if tendency:
                    return SemanticTendencyEvaluator().evaluate(
                        history,
                        facts,
                        candidates,
                        observed,
                        selected_candidates=(candidate,),
                    )
                return SemanticWorkflowAbstracter().abstract(
                    history,
                    facts,
                    candidates,
                    observed,
                    selected_candidates=(candidate,),
                )

            try:
                result, retries = _evaluate_candidate(
                    evaluate,
                    observed,
                    retry=not tendency,
                    preliminary=prior_result(channel, candidate.candidate_id),
                )
            except (
                SemanticDecisionValidationError,
                SemanticResponseValidationError,
                _ModelFailure,
            ) as error:
                retries = max(0, len(observed.attempts) - 1)
                assembly.workflow_retry_count += retries
                assembly.failures.append(
                    _failure(
                        channel,
                        candidate.candidate_id,
                        error,
                        observed,
                        retries,
                    )
                )
                continue
            assembly.workflow_retry_count += retries
            _checkpoint_validated_candidate(
                observed,
                candidate.candidate_id,
                result,
                ("workflows", "accepted", "rejections"),
            )
            (tendency_results if tendency else workflow_results).append(result)
        assembly.workflows = SemanticWorkflowResult(
            repository_id,
            tuple(d for r in workflow_results for d in r.workflows),
            _merge_evidence(workflow_results),
            tuple(item for r in workflow_results for item in r.rejections),
        )
        assembly.tendencies = SemanticTendencyResult(
            repository_id,
            tuple(item for r in tendency_results for item in r.accepted),
            tuple(item for r in tendency_results for item in r.rejections),
            _merge_evidence(tendency_results),
        )
        for candidate in sorted(selected_anti_patterns, key=lambda item: item.candidate_id):
            observed = observation("anti-pattern", candidate.candidate_id, anti_pattern_model)
            try:
                result, retries = _evaluate_candidate(
                    lambda candidate=candidate, observed=observed: (
                        SemanticAntiPatternAbstracter().abstract(
                            history,
                            negative,
                            observed,
                            selected_candidates=(candidate,),
                        )
                    ),
                    observed,
                    retry=True,
                    preliminary=prior_result("anti-pattern", candidate.candidate_id),
                )
            except (
                SemanticDecisionValidationError,
                SemanticResponseValidationError,
                _ModelFailure,
            ) as error:
                retries = max(0, len(observed.attempts) - 1)
                assembly.anti_pattern_retry_count += retries
                assembly.failures.append(
                    _failure(
                        "anti-pattern",
                        candidate.candidate_id,
                        error,
                        observed,
                        retries,
                    )
                )
                continue
            assembly.anti_pattern_retry_count += retries
            _checkpoint_validated_candidate(
                observed,
                candidate.candidate_id,
                result,
                ("anti_patterns", "rejections"),
            )
            anti_results.append(result)
        assembly.anti_patterns = SemanticAntiPatternResult(
            repository_id,
            tuple(d for r in anti_results for d in r.anti_patterns),
            tuple(item for r in anti_results for item in r.rejections),
            _merge_evidence(anti_results),
        )
    # One candidate at a time preserves raw responses for structural failures, too.
    structural_results = []
    for candidate in mined.candidates:
        observed = observation("structural", candidate.candidate_id, structural_model)
        scoped = replace(
            mined,
            candidates=(candidate,),
            evidence=tuple(
                item for item in mined.evidence if item.evidence_id in candidate.rule.evidence_ids
            ),
        )
        prior = prior_result("structural", candidate.candidate_id)
        result = prior[0] if prior else SemanticStructuralRuleEvaluator().evaluate(scoped, observed)
        if result.failures and observed.used_batch:
            result = SemanticStructuralRuleEvaluator().evaluate(scoped, observed)
        _checkpoint_validated_candidate(
            observed,
            candidate.candidate_id,
            result,
            ("accepted", "rejections"),
        )
        for failure in result.failures:
            assembly.failures.append(
                _failure(
                    "structural",
                    failure.candidate_id,
                    failure.reason,
                    observed,
                    0,
                )
            )
        structural_results.append(result)
    assembly.structural = SemanticStructuralRuleResult(
        repository_id,
        tuple(a for r in structural_results for a in r.accepted),
        tuple(a for r in structural_results for a in r.rejections),
        tuple(a for r in structural_results for a in r.failures),
        _merge_evidence(structural_results),
    )
    # Results remain subject to the same repository-wide Evidence conflict gate.
    _merge_evidence(
        item
        for item in (
            assembly.workflows,
            assembly.tendencies,
            assembly.anti_patterns,
            assembly.structural,
        )
        if item is not None
    )
    return assembly
