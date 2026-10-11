"""Canonical, data-only source and plan artifacts for deterministic evolution."""

from __future__ import annotations

import hashlib
import importlib
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from skillforge.domain.evolution import (
    EvolutionHistoryState,
    EvolutionKnowledge,
    EvolutionPlan,
    GitHistoryDelta,
    IncrementalEvolutionResult,
    fingerprint,
)
from skillforge.domain.feedback import HumanFeedbackLedger
from skillforge.domain.project_dna import ProjectDNA
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import SkillCompilationResult
from skillforge.domain.verification import VerifyReport
from skillforge.feedback.artifacts import _entry, restore_feedback_entry
from skillforge.repository.identity import repository_identity
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.verify.artifacts import (
    _canonical_json,
    _read_repo_json,
    _write_json,
    load_verify_artifact,
    restore_verify_artifact,
)

BASELINES = Path(".skillforge/productization/evolution/baselines")
PLANS = Path(".skillforge/productization/evolution/plans")
RESULTS = Path(".skillforge/productization/evolution/results")
BASELINE_SCHEMA = "skillforge-evolution-baseline-v1"
PLAN_SCHEMA = "skillforge-evolution-plan-v1"
RESULT_SCHEMA = "skillforge-evolution-result-v1"
_MAX_EVOLUTION_BYTES = 64 * 1_048_576


def _names() -> dict[str, Any]:
    names: dict[str, Any] = {}
    for module in (
        "change",
        "evidence",
        "evolution",
        "negative_patterns",
        "project_dna",
        "semantic_antipattern",
        "semantic_structural_rule",
        "semantic_workflow",
        "semantic_tendency",
        "static_analysis",
        "skills",
        "structural_rules",
        "workflow",
    ):
        names.update(vars(importlib.import_module(f"skillforge.domain.{module}")))
    return names


def _restore(kind: Any, value: Any, *, _schemas=None) -> Any:
    """Use public constructors and reject every unknown or noncanonical field."""
    # Share immutable domain schema resolution only inside this decode. Never
    # cache artifact values or keep type hints across independent loads.
    if _schemas is None:
        _schemas = {}
    if kind is Any:
        raise ValueError("untyped evolution artifact field")
    origin = get_origin(kind)
    if origin in (types.UnionType, Union):
        if value is None and type(None) in get_args(kind):
            return None
        matches = []
        for choice in get_args(kind):
            if choice is type(None):
                continue
            try:
                matches.append(_restore(choice, value, _schemas=_schemas))
            except (TypeError, ValueError, KeyError):
                pass
        if len(matches) != 1:
            raise ValueError("invalid evolution artifact union value")
        return matches[0]
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError("evolution tuple must be an array")
        args = get_args(kind)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_restore(args[0], item, _schemas=_schemas) for item in value)
        if len(args) != len(value):
            raise ValueError("invalid evolution tuple length")
        return tuple(
            _restore(arg, item, _schemas=_schemas) for arg, item in zip(args, value, strict=True)
        )
    if kind in (str, int, float, bool):
        if type(value) is not kind:
            raise ValueError("invalid evolution scalar type")
        return value
    if isinstance(kind, type) and issubclass(kind, Enum):
        return kind(value)
    if isinstance(kind, type) and is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError("evolution object must be a mapping")
        if kind not in _schemas:
            _schemas[kind] = fields(kind), get_type_hints(kind, globalns=_names())
        definitions, hints = _schemas[kind]
        expected = {item.name for item in definitions}
        actual = set(value)
        omittable_defaults = {
            item.name
            for item in definitions
            if item.metadata.get("omit_empty") and item.default is not MISSING
        }
        if actual - expected or (expected - actual) - omittable_defaults:
            raise ValueError("invalid evolution object fields")
        args = {
            item.name: _restore(hints[item.name], value[item.name], _schemas=_schemas)
            for item in definitions
            if item.init and item.name in value
        }
        result = kind(**args)
        if to_json_value(result) != value:
            raise ValueError("evolution object identity mismatch")
        return result
    raise ValueError(f"unsupported evolution artifact type: {kind}")


@dataclass(frozen=True, slots=True)
class EvolutionBaseline:
    repository_id: str
    head_commit: str
    history_limit: int
    source_dna: ProjectDNA
    source_compilation: SkillCompilationResult
    source_state: EvolutionHistoryState
    semantic_structural: SemanticStructuralRuleResult
    baseline_id: str
    production_knowledge: EvolutionKnowledge | None = None


@dataclass(frozen=True, slots=True)
class LoadedEvolutionPlan:
    plan: EvolutionPlan
    delta: GitHistoryDelta
    target_state: EvolutionHistoryState
    feedback: HumanFeedbackLedger | None


def _baseline_value(baseline: EvolutionBaseline) -> dict[str, object]:
    value = to_json_value(baseline)
    if baseline.production_knowledge is None:
        value.pop("production_knowledge")
    return {"schema": BASELINE_SCHEMA, **value}


def _short_id(parts: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def _validate_candidate_ids(state: EvolutionHistoryState, facts=None) -> None:
    from skillforge.domain.change import ChangePatternKind
    from skillforge.repository.static_delegation import delegation_candidates

    static_candidates = (
        {item.candidate_id: item for item in delegation_candidates(facts)}
        if facts is not None
        else {}
    )
    for item in state.workflow_candidates.candidates:
        if item.kind == ChangePatternKind.STATIC_DELEGATION:
            if static_candidates.get(item.candidate_id) != item:
                raise ValueError("static delegation candidate identity/facts mismatch")
            continue
        expected = _short_id(
            (state.repository_id, item.kind.value, *(member.value for member in item.members))
        )
        if item.candidate_id != expected:
            raise ValueError("workflow candidate identity mismatch")
    for signal in state.antipattern_candidates.signals:
        expected = _short_id(
            (
                "negative-signal-v1",
                state.repository_id,
                signal.kind.value,
                signal.method,
                *signal.commit_ids,
                "affected_paths",
                *signal.affected_paths,
                "matched_terms",
                *signal.matched_terms,
                "referenced_commit_id",
                signal.referenced_commit_id or "",
            )
        )
        if signal.signal_id != expected:
            raise ValueError("negative signal identity mismatch")
    for item in state.antipattern_candidates.candidates:
        expected = _short_id(
            (
                "negative-pattern-candidate-v1",
                state.repository_id,
                "affected_paths",
                *item.affected_paths,
                "signals",
                *item.signal_ids,
            )
        )
        if item.candidate_id != expected:
            raise ValueError("negative candidate identity mismatch")


def make_baseline(
    root: Path,
    head: str,
    limit: int,
    dna: ProjectDNA,
    compilation: SkillCompilationResult,
    state: EvolutionHistoryState,
    semantic: SemanticStructuralRuleResult,
    production_knowledge: EvolutionKnowledge | None = None,
) -> EvolutionBaseline:
    repo_id = repository_identity(root)
    if (
        dna.repository_id != repo_id
        or compilation.repository_id != repo_id
        or compilation.source_snapshot_id != dna.snapshot_id
        or state.repository_id != repo_id
        or state.head_commit != head
        or state.max_commits != limit
        or semantic.repository_id != repo_id
    ):
        raise ValueError("evolution baseline source identity mismatch")
    if production_knowledge is not None and production_knowledge.structural != semantic:
        raise ValueError("baseline structural decisions differ from production knowledge")
    if production_knowledge is not None:
        from skillforge.repository.incremental_evolution import (
            production_dna,
            production_inputs,
            validate_production_decisions,
        )

        validate_production_decisions(state, production_knowledge)
        if production_knowledge.inputs != production_inputs(
            state, production_knowledge.static_facts
        ) or dna != production_dna(dna, production_knowledge, production_knowledge):
            raise ValueError("baseline DNA or request dependencies differ from audited knowledge")
    if compilation != SkillCompiler().compile(
        dna,
        semantic,
        production_knowledge.tendencies if production_knowledge else None,
    ):
        raise ValueError("evolution baseline compilation differs from prepared DNA")
    _validate_candidate_ids(
        state, production_knowledge.static_facts if production_knowledge else None
    )
    content = (
        BASELINE_SCHEMA,
        repo_id,
        dna.snapshot_id,
        compilation,
        state.state_id,
        head,
        limit,
        semantic,
    )
    identity = fingerprint(
        content if production_knowledge is None else (*content, production_knowledge)
    )
    return EvolutionBaseline(
        repo_id, head, limit, dna, compilation, state, semantic, identity, production_knowledge
    )


def save_baseline(root: Path, baseline: EvolutionBaseline) -> str:
    return _write_json(
        root,
        BASELINES / f"{baseline.baseline_id}.json",
        _baseline_value(baseline),
        reject_different_existing=True,
        max_bytes=_MAX_EVOLUTION_BYTES,
    )


def load_baseline(root: Path, supplied: Path | str) -> EvolutionBaseline:
    path, data, value = _read_repo_json(root, supplied, "evolution baseline", _MAX_EVOLUTION_BYTES)
    keys = {
        "schema",
        "repository_id",
        "head_commit",
        "history_limit",
        "source_dna",
        "source_compilation",
        "source_state",
        "semantic_structural",
        "baseline_id",
    }
    if (
        not isinstance(value, dict)
        or set(value) not in (keys, keys | {"production_knowledge"})
        or value["schema"] != BASELINE_SCHEMA
    ):
        raise ValueError("invalid evolution baseline schema")
    schemas = {}
    dna = _restore(ProjectDNA, value["source_dna"], _schemas=schemas)
    compiled = _restore(SkillCompilationResult, value["source_compilation"], _schemas=schemas)
    state = _restore(EvolutionHistoryState, value["source_state"], _schemas=schemas)
    semantic = _restore(
        SemanticStructuralRuleResult, value["semantic_structural"], _schemas=schemas
    )
    production = (
        _restore(EvolutionKnowledge, value["production_knowledge"], _schemas=schemas)
        if "production_knowledge" in value
        else None
    )
    expected = make_baseline(
        root,
        value["head_commit"],
        value["history_limit"],
        dna,
        compiled,
        state,
        semantic,
        production,
    )
    if (
        expected.baseline_id != value["baseline_id"]
        or path.relative_to(root.resolve()) != BASELINES / f"{expected.baseline_id}.json"
        or data != _canonical_json(_baseline_value(expected))
    ):
        raise ValueError("evolution baseline identity or canonical JSON mismatch")
    return expected


def _verify_source_path(source: str, report: VerifyReport) -> None:
    verify_dir = Path(".skillforge/productization/verify")
    # Accept the historical immutable key and the content-addressed v2 key.
    # The caller has already loaded and canonically validated the artifact.
    keys = {
        fingerprint(("verify-report-artifact-v1", report.plan_id, report.patch.snapshot_id)),
        fingerprint(
            (
                "verify-report-artifact-v2",
                report.plan_id,
                report.patch.snapshot_id,
                report.to_dict(),
            )
        ),
    }
    expected = {(verify_dir / f"{key}.json").as_posix() for key in keys}
    if source not in expected:
        raise ValueError("frozen Verify source path is not canonical")


def save_plan(
    root: Path,
    baseline: EvolutionBaseline,
    target: EvolutionHistoryState,
    delta: GitHistoryDelta,
    plan: EvolutionPlan,
    feedback: HumanFeedbackLedger | None,
    feedback_sources: dict[str, str],
) -> str:
    if (
        baseline.repository_id != repository_identity(root)
        or plan.repository_id != baseline.repository_id
        or target.repository_id != baseline.repository_id
        or delta.repository_id != baseline.repository_id
        or plan.source_snapshot_id != baseline.source_dna.snapshot_id
        or plan.source_head_commit != baseline.head_commit
        or plan.target_head_commit != target.head_commit
        or plan.history_state_id != target.state_id
        or plan.delta_id != delta.delta_id
        or plan.feedback_ledger_id != (feedback.ledger_id if feedback else None)
    ):
        raise ValueError("evolution plan artifact source mismatch")
    if feedback is not None and not {record.feedback_id for record in feedback.records} <= set(
        feedback_sources
    ):
        raise ValueError("evolution plan feedback source mapping is incomplete")
    entries = (
        []
        if feedback is None
        else [_entry(record, feedback_sources[record.feedback_id]) for record in feedback.records]
    )
    sources = {}
    for source in sorted({entry["source_verify_artifact"] for entry in entries}):
        verify_plan, report = load_verify_artifact(root, source)
        _verify_source_path(source, report)
        sources[source] = {
            "schema": "skillforge-verify-artifact-v2",
            "plan": verify_plan.to_dict(),
            "report": report.to_dict(),
        }
    value = {
        "schema": PLAN_SCHEMA,
        "baseline_id": baseline.baseline_id,
        "source_snapshot_id": baseline.source_dna.snapshot_id,
        "source_head_commit": baseline.head_commit,
        "target_head_commit": target.head_commit,
        "delta": to_json_value(delta),
        "target_state": to_json_value(target),
        "applicable_feedback": entries,
        "feedback_sources": sources,
        "feedback_ledger_id": feedback.ledger_id if feedback else None,
        "plan": plan.to_dict(),
        "plan_id": plan.plan_id,
    }
    return _write_json(
        root,
        PLANS / f"{plan.plan_id}.json",
        value,
        reject_different_existing=True,
        max_bytes=_MAX_EVOLUTION_BYTES,
    )


def load_plan(root: Path, supplied: Path | str, baseline: EvolutionBaseline) -> EvolutionPlan:
    return load_plan_context(root, supplied, baseline).plan


def load_plan_context(
    root: Path, supplied: Path | str, baseline: EvolutionBaseline
) -> LoadedEvolutionPlan:
    path, data, value = _read_repo_json(root, supplied, "evolution plan", _MAX_EVOLUTION_BYTES)
    expected_keys = {
        "schema",
        "baseline_id",
        "source_snapshot_id",
        "source_head_commit",
        "target_head_commit",
        "delta",
        "target_state",
        "applicable_feedback",
        "feedback_sources",
        "feedback_ledger_id",
        "plan",
        "plan_id",
    }
    if (
        not isinstance(value, dict)
        or value.keys() != expected_keys
        or value["schema"] != PLAN_SCHEMA
        or value["baseline_id"] != baseline.baseline_id
    ):
        raise ValueError("invalid evolution plan schema or baseline")
    delta = _restore(GitHistoryDelta, value["delta"])
    target = _restore(EvolutionHistoryState, value["target_state"])
    if not isinstance(value["applicable_feedback"], list):
        raise ValueError("invalid applicable feedback")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("source_verify_artifact"), str)
        for item in value["applicable_feedback"]
    ):
        raise ValueError("invalid applicable feedback entry")
    sources = value["feedback_sources"]
    if not isinstance(sources, dict) or sources.keys() != {
        item["source_verify_artifact"] for item in value["applicable_feedback"]
    }:
        raise ValueError("invalid evolution feedback sources")
    reports = {}
    for source, artifact in sources.items():
        verify_plan, report = restore_verify_artifact(root, artifact)
        _verify_source_path(source, report)
        reports[source] = report
        if artifact != {
            "schema": "skillforge-verify-artifact-v2",
            "plan": verify_plan.to_dict(),
            "report": report.to_dict(),
        }:
            raise ValueError("invalid frozen Verify source")
    records = tuple(
        restore_feedback_entry(item, reports[item["source_verify_artifact"]])
        for item in value["applicable_feedback"]
    )
    feedback = HumanFeedbackLedger(baseline.repository_id, records) if records else None
    plan_value = value["plan"]
    plan_fields = fields(EvolutionPlan)
    if isinstance(plan_value, dict) and "production" not in plan_value:
        plan_value = {**plan_value, "production": None}
    if not isinstance(plan_value, dict) or plan_value.keys() != {item.name for item in plan_fields}:
        raise ValueError("invalid EvolutionPlan fields")
    hints = get_type_hints(EvolutionPlan, globalns=_names())
    plan_args = {
        item.name: _restore(hints[item.name], plan_value[item.name])
        for item in plan_fields
        if item.init
    }
    plan = EvolutionPlan(
        **plan_args,
        source_dna=baseline.source_dna,
        source_state=baseline.source_state,
        target_state=target,
        delta=delta,
        source_compilation=baseline.source_compilation,
        feedback=feedback,
        source_semantic_structural=baseline.semantic_structural,
        source_knowledge=baseline.production_knowledge,
    )
    _validate_candidate_ids(target, plan.production.static_facts if plan.production else None)
    expected = {
        "schema": PLAN_SCHEMA,
        "baseline_id": baseline.baseline_id,
        "source_snapshot_id": baseline.source_dna.snapshot_id,
        "source_head_commit": baseline.head_commit,
        "target_head_commit": target.head_commit,
        "delta": to_json_value(delta),
        "target_state": to_json_value(target),
        "applicable_feedback": value["applicable_feedback"],
        "feedback_sources": sources,
        "feedback_ledger_id": feedback.ledger_id if feedback else None,
        "plan": plan.to_dict(),
        "plan_id": plan.plan_id,
    }
    if (
        value != expected
        or path.relative_to(root.resolve()) != PLANS / f"{plan.plan_id}.json"
        or data != _canonical_json(expected)
    ):
        raise ValueError("evolution plan identity or canonical JSON mismatch")
    return LoadedEvolutionPlan(plan, delta, target, feedback)


def _result_value(
    baseline: EvolutionBaseline,
    plan: EvolutionPlan,
    semantic_w: SemanticWorkflowResult,
    semantic_a: SemanticAntiPatternResult,
    result: IncrementalEvolutionResult,
) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "baseline_id": baseline.baseline_id,
        "plan_id": plan.plan_id,
        "semantic_workflows": to_json_value(semantic_w),
        "semantic_antipatterns": to_json_value(semantic_a),
        "result": result.to_dict(),
        "result_id": result.result_id,
    }


def save_result(
    root: Path,
    baseline: EvolutionBaseline,
    plan: EvolutionPlan,
    semantic_w: SemanticWorkflowResult,
    semantic_a: SemanticAntiPatternResult,
    result: IncrementalEvolutionResult,
) -> str:
    if (
        baseline.repository_id != repository_identity(root)
        or result.repository_id != baseline.repository_id
        or result.source_snapshot_id != baseline.source_dna.snapshot_id
        or result.plan_id != plan.plan_id
    ):
        raise ValueError("evolution result source mismatch")
    return _write_json(
        root,
        RESULTS / f"{result.result_id}.json",
        _result_value(baseline, plan, semantic_w, semantic_a, result),
        reject_different_existing=True,
        max_bytes=_MAX_EVOLUTION_BYTES,
    )


def load_result(
    root: Path, supplied: Path | str, baseline: EvolutionBaseline, loaded_plan: LoadedEvolutionPlan
) -> IncrementalEvolutionResult:
    from skillforge.repository.incremental_evolution import IncrementalEvolution

    path, data, value = _read_repo_json(root, supplied, "evolution result", _MAX_EVOLUTION_BYTES)
    if (
        not isinstance(value, dict)
        or value.keys()
        != {
            "schema",
            "baseline_id",
            "plan_id",
            "semantic_workflows",
            "semantic_antipatterns",
            "result",
            "result_id",
        }
        or value["schema"] != RESULT_SCHEMA
        or value["baseline_id"] != baseline.baseline_id
        or value["plan_id"] != loaded_plan.plan.plan_id
    ):
        raise ValueError("invalid evolution result schema or source")
    semantic_w = _restore(SemanticWorkflowResult, value["semantic_workflows"])
    semantic_a = _restore(SemanticAntiPatternResult, value["semantic_antipatterns"])
    if baseline.production_knowledge is not None:
        knowledge = _restore(EvolutionKnowledge, value["result"]["production_knowledge"])
        if semantic_w != knowledge.workflows or semantic_a != knowledge.anti_patterns:
            raise ValueError("production result semantic decision artifacts disagree")
        result = IncrementalEvolution().apply_production_semantic(
            baseline.source_dna,
            baseline.source_compilation,
            loaded_plan.target_state,
            loaded_plan.plan,
            baseline.production_knowledge,
            knowledge,
        )
    else:
        result = IncrementalEvolution().apply_semantic(
            baseline.source_dna,
            baseline.source_compilation,
            loaded_plan.target_state,
            loaded_plan.plan,
            semantic_w,
            semantic_a,
            baseline.semantic_structural,
        )
    expected = _result_value(baseline, loaded_plan.plan, semantic_w, semantic_a, result)
    if (
        value != expected
        or data != _canonical_json(expected)
        or path.relative_to(root.resolve()) != RESULTS / f"{result.result_id}.json"
    ):
        raise ValueError("evolution result identity or canonical JSON mismatch")
    return result
