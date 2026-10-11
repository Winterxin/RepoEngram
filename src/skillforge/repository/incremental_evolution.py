"""Bounded fact refresh and selective, governance-preserving knowledge evolution."""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

from skillforge.domain.change import GitHistoryResult
from skillforge.domain.evolution import (
    EvolutionHistoryState,
    EvolutionKnowledge,
    EvolutionPlan,
    EvolutionProductionPlan,
    EvolutionRefresh,
    EvolutionSemanticInput,
    GitHistoryDelta,
    IncrementalEvolutionResult,
    expected_reviews,
    fingerprint,
)
from skillforge.domain.feedback import HumanFeedbackLedger
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectWorkflow,
)
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.domain.skills import SkillCompilationResult
from skillforge.domain.static_analysis import StaticAnalysisResult
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_antipatterns import (
    CONFIDENCE_METHOD as ANTIPATTERN_METHOD,
)
from skillforge.repository.semantic_antipatterns import (
    AntiPatternSemanticModel,
    SemanticAntiPatternAbstracter,
)
from skillforge.repository.semantic_workflows import (
    CONFIDENCE_METHOD as WORKFLOW_METHOD,
)
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    WorkflowSemanticModel,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def _semantic_core(value: object) -> object:
    """Exclude statistical/provenance fields while retaining actual meaning."""
    excluded = {"evidence_ids", "name_evidence_ids", "intent_evidence_ids",
                "description_evidence_ids", "avoid_behavior_evidence_ids",
                "negative_evidence_ids", "decision_evidence_ids", "support",
                "confidence", "draft_id", "candidate_id", "repository_id",
                "decision_reason"}
    if is_dataclass(value):
        return tuple((item.name, _semantic_core(getattr(value, item.name)))
                     for item in fields(value) if item.name not in excluded)
    if isinstance(value, (tuple, list)):
        return tuple(_semantic_core(item) for item in value)
    return value


def _carry_status(old: ProjectKnowledgeStatus, same_core: bool) -> ProjectKnowledgeStatus:
    if old == ProjectKnowledgeStatus.DEPRECATED:
        return old
    return old if same_core else ProjectKnowledgeStatus.CANDIDATE


def _history_from_delta(state: EvolutionHistoryState,
                        delta: GitHistoryDelta,
                        commit_order: tuple[str, ...] | None = None) -> GitHistoryResult:
    if state.repository_id != delta.repository_id or state.head_commit != delta.base_commit:
        raise ValueError("delta does not advance this history state")
    # Commits are newest first. Old facts are reused, never re-read through Git.
    by_commit = {}
    for change in (*delta.changesets, *state.history.changesets):
        previous = by_commit.setdefault(change.commit_id, change)
        if previous != change:
            raise ValueError("conflicting cached Git changeset")
    if commit_order is None:
        if delta.changesets and any(change.is_merge for change in
                                    (*delta.changesets, *state.history.changesets)):
            raise ValueError("merge history requires bounded Git commit order")
        selected = tuple(dict.fromkeys(
            change.commit_id for change in (*delta.changesets, *state.history.changesets)
        ))[:state.max_commits]
    else:
        if (not commit_order or commit_order[0] != delta.head_commit
                or len(commit_order) > state.max_commits
                or len(commit_order) != len(set(commit_order))):
            raise ValueError("invalid bounded Git commit order")
        if not set(commit_order) <= by_commit.keys():
            raise ValueError("bounded Git order needs uncached old Git facts")
        selected = commit_order
    ordered = tuple(by_commit[key] for key in selected)
    source = {}
    for item in (*state.history.evidence, *delta.evidence):
        previous = source.setdefault(item.evidence_id, item)
        if previous != item:
            raise ValueError("conflicting cached Git Evidence")
    ids = {key for change in ordered for key in change.evidence_ids}
    ids.update(key for change in ordered for file in change.files
               for key in file.evidence_ids)
    return GitHistoryResult(state.repository_id, ordered,
                            tuple(source[key] for key in sorted(ids)))


def _candidate_diff(old: dict[str, object], new: dict[str, object]
                    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (tuple(sorted(key for key in old.keys() & new.keys() if old[key] != new[key])),
            tuple(sorted(new.keys() - old.keys())),
            tuple(sorted(old.keys() - new.keys())))


class IncrementalEvolution:
    """Uses existing deterministic miners, semantic gates, and SkillCompiler."""

    def __init__(self, *, change_miner: ChangePatternMiner | None = None,
                 workflow_miner: WorkflowCandidateMiner | None = None,
                 negative_miner: NegativePatternMiner | None = None) -> None:
        self.change_miner = change_miner or ChangePatternMiner()
        self.workflow_miner = workflow_miner or WorkflowCandidateMiner()
        self.negative_miner = negative_miner or NegativePatternMiner()

    def make_state(self, history: GitHistoryResult, head_commit: str,
                   max_commits: int, static_analysis: StaticAnalysisResult, *,
                   repo_path: Path | str | None = None,
                   ) -> EvolutionHistoryState:
        if len(history.changesets) > max_commits:
            raise ValueError("initial history exceeds max_commits")
        if repo_path is not None:
            from skillforge.repository.history_patch_context import strip_git_patch_evidence

            history = strip_git_patch_evidence(history)
        patterns = self.change_miner.mine(history, static_analysis)
        workflows = self.workflow_miner.mine(history, static_analysis, patterns)
        negative = self.negative_miner.mine(history)
        if repo_path is not None:
            from skillforge.repository.history_patch_context import enrich_selected_history

            enriched = enrich_selected_history(
                repo_path, history, static_analysis, workflows, negative,
            )
            if enriched.history != history:
                history = enriched.history
                patterns = self.change_miner.mine(history, static_analysis)
                workflows = self.workflow_miner.mine(history, static_analysis, patterns)
                negative = self.negative_miner.mine(history)
        return EvolutionHistoryState(history.repository_id, head_commit, max_commits,
                                     history, workflows, negative)

    def advance(self, state: EvolutionHistoryState, delta: GitHistoryDelta,
                static_analysis: StaticAnalysisResult, *,
                repo_path: Path | str | None = None) -> EvolutionHistoryState:
        order = None
        if repo_path is not None:
            order = GitHistoryMiner().commit_window_order(
                repo_path, state.repository_id, delta.head_commit, state.max_commits
            )
        history = _history_from_delta(state, delta, order)
        return self.make_state(history, delta.head_commit, state.max_commits,
                               static_analysis, repo_path=repo_path)

    def plan(
        self,
        source: ProjectDNA,
        source_compilation: SkillCompilationResult,
        state: EvolutionHistoryState,
        target: EvolutionHistoryState,
        delta: GitHistoryDelta,
        feedback: HumanFeedbackLedger | None = None,
        source_semantic_structural: SemanticStructuralRuleResult | None = None,
        *,
        source_knowledge: EvolutionKnowledge | None = None,
        static_analysis: StaticAnalysisResult | None = None,
    ) -> EvolutionPlan:
        if (source.repository_id != state.repository_id
                or source_compilation.repository_id != source.repository_id
                or source_compilation.source_snapshot_id != source.snapshot_id):
            raise ValueError("source DNA/compilation/state mismatch")
        if feedback is not None and feedback.repository_id != source.repository_id:
            raise ValueError("feedback repository_id mismatch")
        old_w = {item.candidate_id: item for item in state.workflow_candidates.candidates}
        new_w = {item.candidate_id: item for item in target.workflow_candidates.candidates}
        old_a = {item.candidate_id: item for item in state.antipattern_candidates.candidates}
        new_a = {item.candidate_id: item for item in target.antipattern_candidates.candidates}
        for item in source.workflows:
            old_w.setdefault(item.draft.candidate_id, None)
        for item in source.anti_patterns:
            old_a.setdefault(item.draft.candidate_id, None)
        workflow_diff = _candidate_diff(old_w, new_w)
        anti_diff = _candidate_diff(old_a, new_a)

        reviews = expected_reviews(source, source_compilation, target, delta, feedback)
        production = None
        if source_knowledge is not None:
            if static_analysis is None:
                raise ValueError("production evolution requires current static facts")
            inputs = production_inputs(target, static_analysis)
            production = EvolutionProductionPlan(
                inputs,
                fingerprint(static_analysis),
                source_knowledge.production_version,
                production_refresh(source_knowledge, inputs, reviews),
                static_analysis,
            )
        return EvolutionPlan(
            source.repository_id,
            source.snapshot_id,
            state.head_commit,
            target.head_commit,
            target.state_id,
            delta.delta_id,
            feedback.ledger_id if feedback else None,
            *workflow_diff,
            *anti_diff,
            *reviews,
            source,
            state,
            target,
            delta,
            source_compilation,
            feedback,
            source_semantic_structural,
            production,
            source_knowledge,
        )

    def apply(self, source: ProjectDNA, source_compilation: SkillCompilationResult,
              state: EvolutionHistoryState, plan: EvolutionPlan,
              static_analysis: StaticAnalysisResult,
              workflow_model: WorkflowSemanticModel,
              antipattern_model: AntiPatternSemanticModel,
              source_semantic_structural: SemanticStructuralRuleResult | None = None
              ) -> IncrementalEvolutionResult:
        if (plan.source_snapshot_id != source.snapshot_id
                or plan.history_state_id != state.state_id
                or source_compilation.source_snapshot_id != source.snapshot_id):
            raise ValueError("application source artifacts do not match plan")
        selected_w = tuple(item for item in state.workflow_candidates.candidates
                           if item.candidate_id in plan.workflow_refresh_ids)
        selected_a = tuple(item for item in state.antipattern_candidates.candidates
                           if item.candidate_id in plan.antipattern_refresh_ids)
        semantic_w = SemanticWorkflowAbstracter().abstract(
            state.history, static_analysis, state.workflow_candidates,
            workflow_model, selected_candidates=selected_w)
        semantic_a = SemanticAntiPatternAbstracter().abstract(
            state.history, state.antipattern_candidates, antipattern_model,
            selected_candidates=selected_a)
        return self.apply_semantic(source, source_compilation, state, plan,
                                   semantic_w, semantic_a, source_semantic_structural)

    def apply_semantic(self, source: ProjectDNA,
                       source_compilation: SkillCompilationResult,
                       state: EvolutionHistoryState, plan: EvolutionPlan,
                       semantic_w: SemanticWorkflowResult,
                       semantic_a: SemanticAntiPatternResult,
                       source_semantic_structural: SemanticStructuralRuleResult | None = None
                       ) -> IncrementalEvolutionResult:
        if (plan.source_snapshot_id != source.snapshot_id
                or plan.history_state_id != state.state_id
                or source_compilation.source_snapshot_id != source.snapshot_id):
            raise ValueError("application source artifacts do not match plan")
        if (semantic_w.repository_id != source.repository_id
                or semantic_a.repository_id != source.repository_id):
            raise ValueError("semantic result repository_id mismatch")
        decision_w = {item.candidate_id for item in (*semantic_w.workflows,
                                                     *semantic_w.rejections)}
        decision_a = {item.candidate_id for item in (*semantic_a.anti_patterns,
                                                     *semantic_a.rejections)}
        if (decision_w != set(plan.workflow_refresh_ids)
                or decision_a != set(plan.antipattern_refresh_ids)):
            raise ValueError("semantic decisions must exactly match planned refresh")
        for semantic, candidates, label in (
            (semantic_w.evidence, state.workflow_candidates.evidence, "workflow"),
            (semantic_a.evidence, state.antipattern_candidates.evidence, "Anti-pattern"),
        ):
            canonical = {item.evidence_id: item for item in candidates}
            if any(canonical.get(item.evidence_id) != item for item in semantic):
                raise ValueError(f"semantic {label} Evidence differs from target candidate")
        workflow_candidates = {item.candidate_id: item
                               for item in state.workflow_candidates.candidates}
        anti_candidates = {item.candidate_id: item
                           for item in state.antipattern_candidates.candidates}
        for item in semantic_w.workflows:
            candidate = workflow_candidates[item.candidate_id]
            expected_id = hashlib.sha256(
                f"{WORKFLOW_METHOD}\0{source.repository_id}\0{item.candidate_id}".encode()
            ).hexdigest()[:24]
            if (item.draft_id != expected_id or item.support != candidate.support
                    or item.confidence.method != WORKFLOW_METHOD
                    or item.confidence.support != candidate.support
                    or not set(item.evidence_ids) <= set(candidate.evidence_ids)):
                raise ValueError("semantic workflow is not closed over its candidate")
        for item in semantic_w.rejections:
            candidate = workflow_candidates[item.candidate_id]
            if (item.support != candidate.support
                    or not set(item.evidence_ids) <= set(candidate.evidence_ids)):
                raise ValueError("workflow rejection is not closed over its candidate")
        for item in semantic_a.anti_patterns:
            candidate = anti_candidates[item.candidate_id]
            expected_id = hashlib.sha256(
                f"{ANTIPATTERN_METHOD}\0{source.repository_id}\0{item.candidate_id}".encode()
            ).hexdigest()[:24]
            if (item.draft_id != expected_id
                    or not set(item.evidence_ids) <= set(candidate.evidence_ids)):
                raise ValueError("semantic Anti-pattern is not closed over its candidate")
        for item in semantic_a.rejections:
            candidate = anti_candidates[item.candidate_id]
            if not set(item.evidence_ids) <= set(candidate.evidence_ids):
                raise ValueError("Anti-pattern rejection is not closed over its candidate")

        workflows = {item.workflow_id: item for item in source.workflows}
        anti_patterns = {item.anti_pattern_id: item for item in source.anti_patterns}
        refreshed_w = set()
        added_w = set()
        unresolved_w = {item.workflow_id for item in source.workflows
                        if item.draft.candidate_id in plan.missing_workflow_candidate_ids}
        refreshed_a = set()
        added_a = set()
        unresolved_a = {item.anti_pattern_id for item in source.anti_patterns
                        if item.draft.candidate_id in plan.missing_antipattern_candidate_ids}
        for draft in semantic_w.workflows:
            old = workflows.get(draft.draft_id)
            if old is None:
                workflows[draft.draft_id] = ProjectWorkflow(
                    draft.draft_id, source.repository_id,
                    ProjectKnowledgeStatus.CANDIDATE, draft)
                added_w.add(draft.draft_id)
            else:
                status = _carry_status(
                    old.status, _semantic_core(old.draft) == _semantic_core(draft)
                )
                workflows[draft.draft_id] = ProjectWorkflow(
                    draft.draft_id, source.repository_id, status, draft)
                refreshed_w.add(draft.draft_id)
        for rejection in semantic_w.rejections:
            unresolved_w.update(item.workflow_id for item in source.workflows
                                if item.draft.candidate_id == rejection.candidate_id)
        for draft in semantic_a.anti_patterns:
            old = anti_patterns.get(draft.draft_id)
            if old is None:
                anti_patterns[draft.draft_id] = ProjectAntiPattern(
                    draft.draft_id, source.repository_id,
                    ProjectKnowledgeStatus.CANDIDATE, draft)
                added_a.add(draft.draft_id)
            else:
                status = _carry_status(
                    old.status, _semantic_core(old.draft) == _semantic_core(draft)
                )
                anti_patterns[draft.draft_id] = ProjectAntiPattern(
                    draft.draft_id, source.repository_id, status, draft)
                refreshed_a.add(draft.draft_id)
        for rejection in semantic_a.rejections:
            unresolved_a.update(item.anti_pattern_id for item in source.anti_patterns
                                if item.draft.candidate_id == rejection.candidate_id)

        sources = {}
        for item in (*source.evidence, *semantic_w.evidence, *semantic_a.evidence):
            previous = sources.setdefault(item.evidence_id, item)
            if previous != item:
                raise ValueError("conflicting Project DNA Evidence")
        ids = {key for item in source.rules for key in item.evidence_ids}
        ids.update(key for item in workflows.values() for key in item.draft.evidence_ids)
        ids.update(key for item in anti_patterns.values() for key in item.draft.evidence_ids)
        dna = ProjectDNA(source.repository_id, source.rules,
                         tuple(sorted(workflows.values(), key=lambda item: item.workflow_id)),
                         tuple(sorted(anti_patterns.values(),
                                      key=lambda item: item.anti_pattern_id)),
                         tuple(sources[key] for key in sorted(ids)))
        compilation = SkillCompiler().compile(dna, source_semantic_structural)
        old_skills = {item.skill_id: item for item in source_compilation.skills}
        new_skills = {item.skill_id: item for item in compilation.skills}
        added, removed, changed, unchanged = (
            tuple(sorted(new_skills.keys() - old_skills.keys())),
            tuple(sorted(old_skills.keys() - new_skills.keys())),
            tuple(sorted(key for key in old_skills.keys() & new_skills.keys()
                         if old_skills[key].version != new_skills[key].version)),
            tuple(sorted(key for key in old_skills.keys() & new_skills.keys()
                         if old_skills[key].version == new_skills[key].version)),
        )
        return IncrementalEvolutionResult(
            source.repository_id, source.snapshot_id, dna.snapshot_id,
            plan.source_head_commit, plan.target_head_commit, plan.plan_id,
            fingerprint(semantic_w), fingerprint(semantic_a),
            tuple(sorted(refreshed_w)), tuple(sorted(added_w)), tuple(sorted(unresolved_w)),
            tuple(sorted(refreshed_a)), tuple(sorted(added_a)), tuple(sorted(unresolved_a)),
            plan.rule_review_ids, added, removed, changed, unchanged,
            dna, compilation, state, source, source_compilation, plan,
            semantic_w, semantic_a, source_semantic_structural,
        )

    def apply_production_semantic(
        self, source, source_compilation, state, plan, source_knowledge, knowledge
    ):
        """Assemble current audited decisions, pruning removed or rejected sources."""
        dna = production_dna(source, source_knowledge, knowledge)
        compilation = SkillCompiler().compile(dna, knowledge.structural, knowledge.tendencies)
        old = {item.skill_id: item for item in source_compilation.skills}
        new = {item.skill_id: item for item in compilation.skills}
        diff = (
            tuple(sorted(new.keys() - old.keys())),
            tuple(sorted(old.keys() - new.keys())),
            tuple(
                sorted(
                    key for key in old.keys() & new.keys() if old[key].version != new[key].version
                )
            ),
            tuple(
                sorted(
                    key for key in old.keys() & new.keys() if old[key].version == new[key].version
                )
            ),
        )
        changes = []
        for old_items, new_items, key, channel in (
            (source.workflows, dna.workflows, "workflow_id", "workflow"),
            (source.anti_patterns, dna.anti_patterns, "anti_pattern_id", "anti-pattern"),
        ):
            prior = {getattr(item, key): item for item in old_items}
            current = {getattr(item, key): item for item in new_items}
            refresh = set(plan.production.refresh_ids(channel))
            changes.extend(
                (
                    tuple(
                        sorted(
                            identifier
                            for identifier in prior.keys() & current.keys()
                            if current[identifier].draft.candidate_id in refresh
                        )
                    ),
                    tuple(sorted(current.keys() - prior.keys())),
                    tuple(sorted(prior.keys() - current.keys())),
                )
            )
        return IncrementalEvolutionResult(
            source.repository_id,
            source.snapshot_id,
            dna.snapshot_id,
            plan.source_head_commit,
            plan.target_head_commit,
            plan.plan_id,
            fingerprint(knowledge.workflows),
            fingerprint(knowledge.anti_patterns),
            *changes,
            plan.rule_review_ids,
            *diff,
            dna,
            compilation,
            state,
            source,
            source_compilation,
            plan,
            knowledge.workflows,
            knowledge.anti_patterns,
            source_knowledge.structural,
            knowledge,
            source_knowledge,
        )


def production_inputs(state, facts):
    """Hash original requests in full, including every context and Evidence field."""
    from skillforge.domain.change import ChangePatternKind
    from skillforge.productization.prepare import (
        ANTIPATTERN_SEMANTIC_LIMIT,
        WORKFLOW_SEMANTIC_LIMIT,
    )
    from skillforge.productization.semantic_assembly import _merge_evidence
    from skillforge.repository.semantic_antipattern_evaluation import (
        select_semantic_antipattern_candidates,
    )
    from skillforge.repository.semantic_antipatterns import build_semantic_antipattern_request
    from skillforge.repository.semantic_evaluation import select_semantic_candidates
    from skillforge.repository.semantic_structural_rules import build_semantic_structural_request
    from skillforge.repository.semantic_tendencies import build_semantic_tendency_request
    from skillforge.repository.semantic_workflows import build_semantic_workflow_request
    from skillforge.repository.structural_rules import StructuralRuleMiner

    result = []
    mined = StructuralRuleMiner().mine(facts)
    evidence = {
        item.evidence_id: item
        for item in _merge_evidence(
            (
                state.history,
                facts,
                state.workflow_candidates,
                state.antipattern_candidates,
                mined,
            )
        )
    }

    def add(channel, candidate, request):
        result.append(
            EvolutionSemanticInput(
                channel,
                candidate.candidate_id,
                fingerprint(request),
                fingerprint(candidate),
                fingerprint(tuple(evidence[item.evidence_id] for item in request.evidence)),
            )
        )

    for candidate in select_semantic_candidates(
        state.workflow_candidates,
        limit=WORKFLOW_SEMANTIC_LIMIT,
    ):
        tendency = candidate.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
        builder = build_semantic_tendency_request if tendency else build_semantic_workflow_request
        add(
            "tendency" if tendency else "workflow",
            candidate,
            builder(state.history, facts, state.workflow_candidates, candidate),
        )
    for candidate in select_semantic_antipattern_candidates(
        state.antipattern_candidates,
        limit=ANTIPATTERN_SEMANTIC_LIMIT,
    ):
        add(
            "anti-pattern",
            candidate,
            build_semantic_antipattern_request(
                state.history,
                state.antipattern_candidates,
                candidate,
            ),
        )
    for candidate in mined.candidates:
        add("structural", candidate, build_semantic_structural_request(mined, candidate))
    return tuple(sorted(result, key=lambda item: (item.channel, item.candidate_id)))


def _decision_id(item):
    return item.response.candidate_id if hasattr(item, "response") else item.candidate_id


def _decision_ids(result):
    return {
        _decision_id(item)
        for name in ("workflows", "anti_patterns", "accepted", "rejections")
        for item in getattr(result, name, ())
    }


def production_refresh(source, inputs, reviews=()):
    old = {(item.channel, item.candidate_id): item for item in source.inputs}
    new = {(item.channel, item.candidate_id): item for item in inputs}
    complete = {
        channel: _decision_ids(result)
        for channel, result in (
            ("workflow", source.workflows),
            ("tendency", source.tendencies),
            ("anti-pattern", source.anti_patterns),
            ("structural", source.structural),
        )
    }
    result = []
    for channel, identifier in sorted(old.keys() | new.keys()):
        key = (channel, identifier)
        if key not in new:
            reason = "REMOVED_SOURCE"
        elif key not in old:
            reason = "NEW_CANDIDATE"
        elif identifier not in complete[channel]:
            reason = "MISSING_DECISION"
        elif reviews and identifier in reviews[0 if channel in {"workflow", "tendency"} else 1]:
            reason = "DIAGNOSTIC_REVIEW"
        elif old[key].evidence_hash != new[key].evidence_hash:
            reason = "SOURCE_EVIDENCE_CHANGED"
        elif old[key].candidate_hash != new[key].candidate_hash:
            reason = "STATIC_FACT_CHANGED" if channel == "structural" else "CANDIDATE_CHANGED"
        elif old[key].request_hash != new[key].request_hash:
            reason = "STATIC_FACT_CHANGED" if channel == "structural" else "SOURCE_CONTEXT_CHANGED"
        else:
            reason = "UNCHANGED_REUSED"
        result.append(EvolutionRefresh(channel, identifier, reason))
    return tuple(result)


def select_decisions(result, ids):
    """Retain the exact audited objects and their exact Evidence closure."""
    selected = {}
    refs = set()
    for name in ("workflows", "anti_patterns", "accepted", "rejections", "failures"):
        if not hasattr(result, name):
            continue
        items = tuple(item for item in getattr(result, name) if _decision_id(item) in ids)
        selected[name] = items
        for item in items:
            if hasattr(item, "rule"):
                refs.update(item.rule.evidence_ids)
            elif hasattr(item, "candidate"):
                refs.update(item.candidate.rule.evidence_ids)
            else:
                refs.update(getattr(item, "evidence_ids", ()))
    selected["evidence"] = tuple(item for item in result.evidence if item.evidence_id in refs)
    return replace(result, **selected)


def merge_production_knowledge(source, plan, assembly):
    from skillforge.productization.semantic_assembly import _merge_evidence

    results = []
    for channel, name in (
        ("workflow", "workflows"),
        ("tendency", "tendencies"),
        ("anti-pattern", "anti_patterns"),
        ("structural", "structural"),
    ):
        reused = {
            item.candidate_id
            for item in plan.decisions
            if item.channel == channel and item.reason == "UNCHANGED_REUSED"
        }
        prior = select_decisions(getattr(source, name), reused)
        fresh = getattr(assembly, name)
        values = {}
        for field in ("workflows", "anti_patterns", "accepted", "rejections", "failures"):
            if hasattr(prior, field):
                values[field] = tuple(
                    sorted(
                        (*getattr(prior, field), *getattr(fresh, field)),
                        key=_decision_id,
                    )
                )
        values["evidence"] = _merge_evidence((prior, fresh))
        results.append(replace(prior, **values))
    return EvolutionKnowledge(
        plan.inputs, plan.static_hash, plan.production_version, *results, plan.static_facts
    )


def production_dna(source, old, new):
    from skillforge.repository.project_dna import ProjectDNABuilder

    generated = {item.rule.rule_id for item in old.tendencies.accepted}
    generated.update(item.candidate.rule.rule_id for item in old.structural.accepted)
    rules = tuple(item for item in source.rules if item.rule_id not in generated)
    rule_refs = {key for item in rules for key in item.evidence_ids}
    dna = ProjectDNABuilder().build(
        source.repository_id,
        rules=rules,
        rule_evidence=tuple(item for item in source.evidence if item.evidence_id in rule_refs),
        workflows=new.workflows,
        anti_patterns=new.anti_patterns,
        semantic_structural_rules=new.structural,
        semantic_tendencies=new.tendencies,
    )
    # Human governance survives reuse; changed meaning returns to candidate status.
    updates = {}
    for name, key in (("workflows", "workflow_id"), ("anti_patterns", "anti_pattern_id")):
        prior = {getattr(item, key): item for item in getattr(source, name)}
        updates[name] = tuple(
            replace(
                item,
                status=_carry_status(
                    prior[getattr(item, key)].status,
                    _semantic_core(prior[getattr(item, key)].draft) == _semantic_core(item.draft),
                ),
            )
            if getattr(item, key) in prior
            else item
            for item in getattr(dna, name)
        )
    return replace(dna, **updates)


def validate_production_result(result, source, compiled, plan, old):
    new = result.production_knowledge
    if old is None or plan.production is None:
        raise ValueError("production result requires its source and plan")
    if (
        new.inputs != plan.production.inputs
        or new.static_hash != plan.production.static_hash
        or new.production_version != plan.production.production_version
        or result.resulting_history_state.state_id != plan.history_state_id
        or result.plan_id != plan.plan_id
        or result.source_snapshot_id != source.snapshot_id
        or result.result_snapshot_id != result.resulting_project_dna.snapshot_id
        or result.source_head_commit != plan.source_head_commit
        or result.target_head_commit != plan.target_head_commit
        or new.inputs != production_inputs(result.resulting_history_state, new.static_facts)
        or result.repository_id != source.repository_id
        or result.semantic_workflow_result_id != fingerprint(new.workflows)
        or result.semantic_antipattern_result_id != fingerprint(new.anti_patterns)
        or result.rule_review_ids != plan.rule_review_ids
    ):
        raise ValueError("production result source mismatch")
    validate_production_decisions(result.resulting_history_state, new)
    for channel, name in (
        ("workflow", "workflows"),
        ("tendency", "tendencies"),
        ("anti-pattern", "anti_patterns"),
        ("structural", "structural"),
    ):
        reused = {
            item.candidate_id
            for item in plan.production.decisions
            if item.channel == channel and item.reason == "UNCHANGED_REUSED"
        }
        current = getattr(new, name)
        if select_decisions(current, reused) != select_decisions(getattr(old, name), reused):
            raise ValueError("unchanged validated knowledge was modified")
        allowed = {item.candidate_id for item in new.inputs if item.channel == channel}
        observed = _decision_ids(current) | {
            item.candidate_id for item in getattr(current, "failures", ())
        }
        if not observed <= allowed:
            raise ValueError("production decisions include removed or unselected candidates")
    expected_dna = production_dna(source, old, new)
    if (
        result.resulting_project_dna != expected_dna
        or compiled != SkillCompiler().compile(source, old.structural, old.tendencies)
        or result.resulting_compilation
        != SkillCompiler().compile(expected_dna, new.structural, new.tendencies)
    ):
        raise ValueError("production DNA/compilation differs from audited decisions")
    before = {item.skill_id: item for item in compiled.skills}
    after = {item.skill_id: item for item in result.resulting_compilation.skills}
    expected = (
        tuple(sorted(after.keys() - before.keys())),
        tuple(sorted(before.keys() - after.keys())),
        tuple(
            sorted(
                key
                for key in before.keys() & after.keys()
                if before[key].version != after[key].version
            )
        ),
        tuple(
            sorted(
                key
                for key in before.keys() & after.keys()
                if before[key].version == after[key].version
            )
        ),
    )
    if expected != (
        result.skill_added_ids,
        result.skill_removed_ids,
        result.skill_version_changed_ids,
        result.skill_unchanged_ids,
    ):
        raise ValueError("production Skill diff mismatch")
    for prior_items, current_items, key, channel, refreshed, added, unresolved in (
        (
            source.workflows,
            result.resulting_project_dna.workflows,
            "workflow_id",
            "workflow",
            result.refreshed_workflow_ids,
            result.added_workflow_ids,
            result.unresolved_workflow_ids,
        ),
        (
            source.anti_patterns,
            result.resulting_project_dna.anti_patterns,
            "anti_pattern_id",
            "anti-pattern",
            result.refreshed_antipattern_ids,
            result.added_antipattern_ids,
            result.unresolved_antipattern_ids,
        ),
    ):
        prior = {getattr(item, key): item for item in prior_items}
        current = {getattr(item, key): item for item in current_items}
        refresh = set(plan.production.refresh_ids(channel))
        expected = (
            tuple(
                sorted(
                    identifier
                    for identifier in prior.keys() & current.keys()
                    if current[identifier].draft.candidate_id in refresh
                )
            ),
            tuple(sorted(current.keys() - prior.keys())),
            tuple(sorted(prior.keys() - current.keys())),
        )
        if expected != (refreshed, added, unresolved):
            raise ValueError("production knowledge diff mismatch")


def validate_production_decisions(state, knowledge):
    """Keep v1's candidate closure checks on both fresh and persisted v2 decisions."""
    from skillforge.productization.semantic_assembly import _merge_evidence
    from skillforge.repository.semantic_tendencies import (
        build_semantic_tendency_request,
        validate_semantic_tendency_response,
    )
    from skillforge.repository.structural_rules import StructuralRuleMiner

    sources = {
        item.evidence_id: item
        for item in _merge_evidence(
            (
                state.history,
                state.workflow_candidates,
                state.antipattern_candidates,
                knowledge.static_facts,
            )
        )
    }
    for result in (
        knowledge.workflows,
        knowledge.tendencies,
        knowledge.anti_patterns,
        knowledge.structural,
    ):
        if any(sources.get(item.evidence_id) != item for item in result.evidence):
            raise ValueError("production semantic Evidence differs from target candidate sources")
    workflows = {item.candidate_id: item for item in state.workflow_candidates.candidates}
    anti = {item.candidate_id: item for item in state.antipattern_candidates.candidates}
    for draft in knowledge.workflows.workflows:
        candidate = workflows[draft.candidate_id]
        expected = hashlib.sha256(
            f"{WORKFLOW_METHOD}\0{state.repository_id}\0{draft.candidate_id}".encode()
        ).hexdigest()[:24]
        if (
            draft.draft_id != expected
            or draft.support != candidate.support
            or draft.confidence.method != WORKFLOW_METHOD
            or draft.confidence.support != candidate.support
            or not set(draft.evidence_ids) <= set(candidate.evidence_ids)
        ):
            raise ValueError("production workflow is not closed over its candidate")
    for rejection in knowledge.workflows.rejections:
        candidate = workflows[rejection.candidate_id]
        if rejection.support != candidate.support or not set(rejection.evidence_ids) <= set(
            candidate.evidence_ids
        ):
            raise ValueError("production workflow rejection differs from target candidate")
    for draft in knowledge.anti_patterns.anti_patterns:
        candidate = anti[draft.candidate_id]
        expected = hashlib.sha256(
            f"{ANTIPATTERN_METHOD}\0{state.repository_id}\0{draft.candidate_id}".encode()
        ).hexdigest()[:24]
        if draft.draft_id != expected or not set(draft.evidence_ids) <= set(candidate.evidence_ids):
            raise ValueError("production Anti-pattern is not closed over its candidate")
    for rejection in knowledge.anti_patterns.rejections:
        if not set(rejection.evidence_ids) <= set(anti[rejection.candidate_id].evidence_ids):
            raise ValueError("production Anti-pattern rejection differs from target candidate")
    for decision in (*knowledge.tendencies.accepted, *knowledge.tendencies.rejections):
        response = getattr(decision, "response", decision)
        request = build_semantic_tendency_request(
            state.history,
            knowledge.static_facts,
            state.workflow_candidates,
            workflows[response.candidate_id],
        )
        if (
            hasattr(decision, "request")
            and decision.request != request
            or validate_semantic_tendency_response(request, response) != decision
        ):
            raise ValueError("production tendency differs from target candidate gate")
    mined = StructuralRuleMiner().mine(knowledge.static_facts)
    structural = {item.candidate_id: item for item in mined.candidates}
    for accepted in knowledge.structural.accepted:
        if structural.get(accepted.candidate_id) != accepted.candidate:
            raise ValueError("production structural acceptance differs from target candidate")
    for rejection in knowledge.structural.rejections:
        if not set(rejection.evidence_ids) <= set(
            structural[rejection.candidate_id].rule.evidence_ids
        ):
            raise ValueError("production structural rejection differs from target candidate")
