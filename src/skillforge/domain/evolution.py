"""Immutable, evidence-closed inputs and outputs for bounded evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass, field

from skillforge.domain.change import ChangeSet, GitHistoryResult
from skillforge.domain.evidence import Evidence
from skillforge.domain.feedback import (
    FeedbackKind,
    HumanFeedbackLedger,
    KnowledgeFeedbackTarget,
    SkillFeedbackTarget,
)
from skillforge.domain.negative_patterns import NegativePatternCandidateResult
from skillforge.domain.project_dna import ProjectDNA
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_tendency import SemanticTendencyResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import SkillCompilationResult
from skillforge.domain.static_analysis import StaticAnalysisResult
from skillforge.domain.workflow import WorkflowCandidateResult


def fingerprint(value: object) -> str:
    encoded = json.dumps(to_json_value(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvolutionSemanticInput:
    channel: str
    candidate_id: str
    request_hash: str
    candidate_hash: str
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class EvolutionKnowledge:
    """Production decisions and their complete deterministic request dependencies."""

    inputs: tuple[EvolutionSemanticInput, ...]
    static_hash: str
    production_version: str
    workflows: SemanticWorkflowResult
    tendencies: SemanticTendencyResult
    anti_patterns: SemanticAntiPatternResult
    structural: SemanticStructuralRuleResult
    static_facts: StaticAnalysisResult

    def __post_init__(self) -> None:
        keys = tuple((item.channel, item.candidate_id) for item in self.inputs)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("production inputs must be sorted and unique")
        repository = self.workflows.repository_id
        if (
            self.static_facts.repository_id != repository
            or fingerprint(self.static_facts) != self.static_hash
        ):
            raise ValueError("production static facts fingerprint mismatch")
        if any(
            item.repository_id != repository
            for item in (
                self.tendencies,
                self.anti_patterns,
                self.structural,
            )
        ):
            raise ValueError("production knowledge repository mismatch")


@dataclass(frozen=True, slots=True)
class EvolutionRefresh:
    channel: str
    candidate_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvolutionProductionPlan:
    inputs: tuple[EvolutionSemanticInput, ...]
    static_hash: str
    production_version: str
    decisions: tuple[EvolutionRefresh, ...]
    static_facts: StaticAnalysisResult

    def __post_init__(self) -> None:
        if fingerprint(self.static_facts) != self.static_hash:
            raise ValueError("production plan static facts fingerprint mismatch")

    def refresh_ids(self, channel: str) -> tuple[str, ...]:
        return tuple(
            item.candidate_id
            for item in self.decisions
            if item.channel == channel
            and item.reason
            not in {
                "UNCHANGED_REUSED",
                "REMOVED_SOURCE",
            }
        )


def canonical_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


def validate_history(repository_id: str, changesets: tuple[ChangeSet, ...],
                     evidence: tuple[Evidence, ...]) -> None:
    if any(item.repository_id != repository_id for item in (*changesets, *evidence)):
        raise ValueError("history repository_id mismatch")
    commits = tuple(item.commit_id for item in changesets)
    if len(commits) != len(set(commits)):
        raise ValueError("duplicate history commit")
    ids = tuple(item.evidence_id for item in evidence)
    canonical_ids(ids, "history Evidence")
    referenced = {key for change in changesets for key in change.evidence_ids}
    referenced.update(key for change in changesets for file in change.files
                      for key in file.evidence_ids)
    if referenced != set(ids):
        raise ValueError("history Evidence must exactly match changeset references")
    by_id = {item.evidence_id: item for item in evidence}
    for change in changesets:
        for key in change.evidence_ids:
            if by_id[key].location.commit != change.commit_id:
                raise ValueError("history Evidence commit mismatch")
        if not any(by_id[key].provenance == "git_log" for key in change.evidence_ids):
            raise ValueError("changeset requires Git log Evidence")
        for file in change.files:
            if not set(file.evidence_ids) <= set(change.evidence_ids):
                raise ValueError("file Evidence must belong to its changeset")
            if any(by_id[key].location.path != file.path
                   or by_id[key].location.commit != change.commit_id
                   for key in file.evidence_ids):
                raise ValueError("file Evidence path or commit mismatch")


def expected_reviews(
    source: ProjectDNA, compilation: SkillCompilationResult,
    target: EvolutionHistoryState, delta: GitHistoryDelta,
    feedback: HumanFeedbackLedger | None,
) -> tuple[tuple[str, ...], ...]:
    impacted = {path for change in delta.changesets for file in change.files
                for path in (file.path, file.previous_path) if path is not None}
    evidence = {item.evidence_id: item for item in source.evidence}
    rules = {item.rule_id for item in source.rules
             if any(evidence[key].location.path in impacted for key in item.evidence_ids)}
    diagnostic_w: set[str] = set()
    diagnostic_a: set[str] = set()
    routing: set[str] = set()
    verify: set[str] = set()
    unresolved: set[str] = set()
    new_w = {item.candidate_id for item in target.workflow_candidates.candidates}
    new_a = {item.candidate_id for item in target.antipattern_candidates.candidates}
    workflows = {item.workflow_id: item.draft.candidate_id for item in source.workflows}
    anti = {item.anti_pattern_id: item.draft.candidate_id
            for item in source.anti_patterns}
    skills = {item.skill_id: item for item in compilation.skills}
    if feedback is not None:
        for record in feedback.active_records:
            if record.kind in {FeedbackKind.CONFIRM, FeedbackKind.DEPRECATE,
                               FeedbackKind.MARK_CONFLICTED,
                               FeedbackKind.MANUAL_OVERRIDE,
                               FeedbackKind.ADD_RULE_EXCEPTION,
                               FeedbackKind.ACCURATE}:
                continue
            target_ref = record.target
            if isinstance(target_ref, KnowledgeFeedbackTarget):
                key = target_ref.knowledge_id
                if key in {item.rule_id for item in source.rules}:
                    rules.add(key)
                elif key in workflows and workflows[key] in new_w:
                    diagnostic_w.add(workflows[key])
                elif key in anti and anti[key] in new_a:
                    diagnostic_a.add(anti[key])
                else:
                    unresolved.add(record.feedback_id)
            elif isinstance(target_ref, SkillFeedbackTarget):
                skill = skills.get(target_ref.skill_id)
                candidate = workflows.get(skill.workflow_id) if skill is not None else None
                if (skill is not None and skill.version == target_ref.version
                        and candidate in new_w):
                    diagnostic_w.add(candidate)
                else:
                    unresolved.add(record.feedback_id)
            elif record.kind in {FeedbackKind.WRONG_ROUTE, FeedbackKind.MISSING_SKILL}:
                routing.add(record.feedback_id)
            elif record.kind in {FeedbackKind.FALSE_POSITIVE,
                                 FeedbackKind.FALSE_NEGATIVE,
                                 FeedbackKind.PERMANENT_WHITELIST,
                                 FeedbackKind.MODULE_EXCEPTION}:
                verify.add(record.feedback_id)
            else:
                unresolved.add(record.feedback_id)
    return tuple(tuple(sorted(items)) for items in (
        diagnostic_w, diagnostic_a, rules, routing, verify, unresolved
    ))


@dataclass(frozen=True, slots=True)
class GitHistoryDelta:
    repository_id: str
    base_commit: str
    head_commit: str
    changesets: tuple[ChangeSet, ...]
    evidence: tuple[Evidence, ...]
    delta_id: str = field(init=False)

    def __post_init__(self) -> None:
        validate_history(self.repository_id, self.changesets, self.evidence)
        if not self.base_commit or not self.head_commit:
            raise ValueError("delta endpoints are required")
        if (self.base_commit == self.head_commit) != (not self.changesets):
            raise ValueError("empty delta must have identical endpoints")
        if self.changesets and self.changesets[0].commit_id != self.head_commit:
            raise ValueError("delta must start at head")
        object.__setattr__(self, "delta_id", fingerprint(("git-delta-v1", self.repository_id,
            self.base_commit, self.head_commit, self.changesets, self.evidence)))

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class EvolutionHistoryState:
    repository_id: str
    head_commit: str
    max_commits: int
    history: GitHistoryResult
    workflow_candidates: WorkflowCandidateResult
    antipattern_candidates: NegativePatternCandidateResult
    state_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (isinstance(self.max_commits, bool) or not isinstance(self.max_commits, int)
                or self.max_commits < 1):
            raise ValueError("max_commits must be positive")
        if not self.head_commit or self.history.repository_id != self.repository_id:
            raise ValueError("history state identity mismatch")
        if any(item.repository_id != self.repository_id for item in
               (self.workflow_candidates, self.antipattern_candidates)):
            raise ValueError("candidate repository_id mismatch")
        if len(self.history.changesets) > self.max_commits:
            raise ValueError("history exceeds bounded window")
        if self.history.changesets and self.history.changesets[0].commit_id != self.head_commit:
            raise ValueError("history head mismatch")
        validate_history(self.repository_id, self.history.changesets, self.history.evidence)
        if self.antipattern_candidates.examined_commits != len(self.history.changesets):
            raise ValueError("negative candidates have a different history window")
        object.__setattr__(self, "state_id", fingerprint(("evolution-history-v1",
            self.repository_id, self.head_commit, self.max_commits, self.history,
            self.workflow_candidates, self.antipattern_candidates)))

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class EvolutionPlan:
    repository_id: str
    source_snapshot_id: str
    source_head_commit: str
    target_head_commit: str
    history_state_id: str
    delta_id: str
    feedback_ledger_id: str | None
    changed_workflow_candidate_ids: tuple[str, ...]
    new_workflow_candidate_ids: tuple[str, ...]
    missing_workflow_candidate_ids: tuple[str, ...]
    changed_antipattern_candidate_ids: tuple[str, ...]
    new_antipattern_candidate_ids: tuple[str, ...]
    missing_antipattern_candidate_ids: tuple[str, ...]
    diagnostic_workflow_candidate_ids: tuple[str, ...]
    diagnostic_antipattern_candidate_ids: tuple[str, ...]
    rule_review_ids: tuple[str, ...]
    routing_review_feedback_ids: tuple[str, ...]
    verify_review_feedback_ids: tuple[str, ...]
    unresolved_feedback_ids: tuple[str, ...]
    source_dna: InitVar[ProjectDNA]
    source_state: InitVar[EvolutionHistoryState]
    target_state: InitVar[EvolutionHistoryState]
    delta: InitVar[GitHistoryDelta]
    source_compilation: InitVar[SkillCompilationResult]
    feedback: InitVar[HumanFeedbackLedger | None]
    source_semantic_structural: InitVar[SemanticStructuralRuleResult | None] = None
    production: EvolutionProductionPlan | None = None
    source_knowledge: InitVar[EvolutionKnowledge | None] = None
    plan_id: str = field(init=False)

    def __post_init__(
        self,
        source_dna: ProjectDNA,
        source_state: EvolutionHistoryState,
        target_state: EvolutionHistoryState,
        delta: GitHistoryDelta,
        source_compilation: SkillCompilationResult,
        feedback: HumanFeedbackLedger | None,
        source_semantic_structural: SemanticStructuralRuleResult | None,
        source_knowledge: EvolutionKnowledge | None,
    ) -> None:
        if not all(isinstance(item, kind) for item, kind in (
            (source_dna, ProjectDNA), (source_state, EvolutionHistoryState),
            (target_state, EvolutionHistoryState), (delta, GitHistoryDelta),
            (source_compilation, SkillCompilationResult))):
            raise ValueError("plan requires exact source artifacts")
        if feedback is not None and not isinstance(feedback, HumanFeedbackLedger):
            raise ValueError("plan feedback ledger must be HumanFeedbackLedger")
        if (any(item.repository_id != self.repository_id for item in (
            source_dna, source_state, target_state, delta, source_compilation
        )) or (feedback is not None and feedback.repository_id != self.repository_id)):
            raise ValueError("plan repository_id mismatch")
        if source_compilation.source_snapshot_id != source_dna.snapshot_id:
            raise ValueError("plan source compilation mismatch")
        from skillforge.repository.skill_compiler import SkillCompiler

        if source_compilation != SkillCompiler().compile(
            source_dna,
            source_semantic_structural,
            source_knowledge.tendencies if source_knowledge else None,
        ):
            raise ValueError("plan source compilation is not the compiled DNA")
        if self.feedback_ledger_id != (feedback.ledger_id if feedback else None):
            raise ValueError("plan feedback ledger identity mismatch")
        if (self.source_snapshot_id != source_dna.snapshot_id
                or self.source_head_commit != source_state.head_commit
                or self.target_head_commit != target_state.head_commit
                or self.history_state_id != target_state.state_id
                or self.delta_id != delta.delta_id
                or delta.base_commit != source_state.head_commit
                or delta.head_commit != target_state.head_commit):
            raise ValueError("plan source artifact mismatch")
        old_w = {item.candidate_id: item for item in source_state.workflow_candidates.candidates}
        new_w = {item.candidate_id: item for item in target_state.workflow_candidates.candidates}
        old_a = {item.candidate_id: item for item in source_state.antipattern_candidates.candidates}
        new_a = {item.candidate_id: item for item in target_state.antipattern_candidates.candidates}
        old_w.update({item.draft.candidate_id: old_w.get(item.draft.candidate_id)
                      for item in source_dna.workflows if item.draft.candidate_id not in old_w})
        old_a.update({item.draft.candidate_id: old_a.get(item.draft.candidate_id)
                      for item in source_dna.anti_patterns if item.draft.candidate_id not in old_a})
        expected = (
            tuple(sorted(key for key in old_w.keys() & new_w.keys() if old_w[key] != new_w[key])),
            tuple(sorted(new_w.keys() - old_w.keys())),
            tuple(sorted(old_w.keys() - new_w.keys())),
            tuple(sorted(key for key in old_a.keys() & new_a.keys() if old_a[key] != new_a[key])),
            tuple(sorted(new_a.keys() - old_a.keys())),
            tuple(sorted(old_a.keys() - new_a.keys())),
        )
        actual = (self.changed_workflow_candidate_ids, self.new_workflow_candidate_ids,
                  self.missing_workflow_candidate_ids,
                  self.changed_antipattern_candidate_ids, self.new_antipattern_candidate_ids,
                  self.missing_antipattern_candidate_ids)
        if actual != expected:
            raise ValueError("plan candidate diff does not match source artifacts")
        for name in ("diagnostic_workflow_candidate_ids",
                     "diagnostic_antipattern_candidate_ids", "rule_review_ids",
                     "routing_review_feedback_ids", "verify_review_feedback_ids",
                     "unresolved_feedback_ids"):
            canonical_ids(getattr(self, name), name)
        if not set(self.diagnostic_workflow_candidate_ids) <= new_w.keys():
            raise ValueError("diagnostic workflow candidate is unavailable")
        if not set(self.diagnostic_antipattern_candidate_ids) <= new_a.keys():
            raise ValueError("diagnostic Anti-pattern candidate is unavailable")
        if not set(self.rule_review_ids) <= {item.rule_id for item in source_dna.rules}:
            raise ValueError("unknown rule review ID")
        if (self.diagnostic_workflow_candidate_ids,
                self.diagnostic_antipattern_candidate_ids,
                self.rule_review_ids,
                self.routing_review_feedback_ids,
                self.verify_review_feedback_ids,
                self.unresolved_feedback_ids) != expected_reviews(
                    source_dna, source_compilation, target_state, delta, feedback):
            raise ValueError("plan reviews do not match source artifacts")
        if self.production is not None:
            from skillforge.repository.incremental_evolution import (
                IncrementalEvolution,
                production_inputs,
                production_refresh,
            )

            if (
                source_knowledge is None
                or self.production.production_version != source_knowledge.production_version
                or self.production.decisions
                != production_refresh(
                    source_knowledge,
                    self.production.inputs,
                    expected_reviews(source_dna, source_compilation, target_state, delta, feedback),
                )
                or self.production.inputs
                != production_inputs(
                    target_state,
                    self.production.static_facts,
                )
                or target_state
                != IncrementalEvolution().make_state(
                    target_state.history,
                    target_state.head_commit,
                    target_state.max_commits,
                    self.production.static_facts,
                )
            ):
                raise ValueError("production refresh decisions differ from source dependencies")
        content = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "source_dna",
                "source_state",
                "target_state",
                "delta",
                "source_compilation",
                "feedback",
                "source_semantic_structural",
                "source_knowledge",
                "plan_id",
            }
            and not (name == "production" and self.production is None)
        )
        object.__setattr__(self, "plan_id", fingerprint(("evolution-plan-v1", content)))

    @property
    def workflow_refresh_ids(self) -> tuple[str, ...]:
        if self.production is not None:
            return self.production.refresh_ids("workflow")
        return tuple(sorted(set(self.changed_workflow_candidate_ids)
                            | set(self.new_workflow_candidate_ids)
                            | set(self.diagnostic_workflow_candidate_ids)))

    @property
    def antipattern_refresh_ids(self) -> tuple[str, ...]:
        if self.production is not None:
            return self.production.refresh_ids("anti-pattern")
        return tuple(sorted(set(self.changed_antipattern_candidate_ids)
                            | set(self.new_antipattern_candidate_ids)
                            | set(self.diagnostic_antipattern_candidate_ids)))

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        if self.production is None:
            value.pop("production")
        return value


@dataclass(frozen=True, slots=True)
class IncrementalEvolutionResult:
    repository_id: str
    source_snapshot_id: str
    result_snapshot_id: str
    source_head_commit: str
    target_head_commit: str
    plan_id: str
    semantic_workflow_result_id: str
    semantic_antipattern_result_id: str
    refreshed_workflow_ids: tuple[str, ...]
    added_workflow_ids: tuple[str, ...]
    unresolved_workflow_ids: tuple[str, ...]
    refreshed_antipattern_ids: tuple[str, ...]
    added_antipattern_ids: tuple[str, ...]
    unresolved_antipattern_ids: tuple[str, ...]
    rule_review_ids: tuple[str, ...]
    skill_added_ids: tuple[str, ...]
    skill_removed_ids: tuple[str, ...]
    skill_version_changed_ids: tuple[str, ...]
    skill_unchanged_ids: tuple[str, ...]
    resulting_project_dna: ProjectDNA
    resulting_compilation: SkillCompilationResult
    resulting_history_state: EvolutionHistoryState
    source_dna: InitVar[ProjectDNA]
    source_compilation: InitVar[SkillCompilationResult]
    plan: InitVar[EvolutionPlan]
    semantic_workflows: InitVar[SemanticWorkflowResult]
    semantic_antipatterns: InitVar[SemanticAntiPatternResult]
    source_semantic_structural: InitVar[SemanticStructuralRuleResult | None] = None
    production_knowledge: EvolutionKnowledge | None = None
    source_knowledge: InitVar[EvolutionKnowledge | None] = None
    result_id: str = field(init=False)

    def __post_init__(
        self,
        source_dna: ProjectDNA,
        source_compilation: SkillCompilationResult,
        plan: EvolutionPlan,
        semantic_workflows: SemanticWorkflowResult,
        semantic_antipatterns: SemanticAntiPatternResult,
        source_semantic_structural: SemanticStructuralRuleResult | None,
        source_knowledge: EvolutionKnowledge | None,
    ) -> None:
        if self.production_knowledge is not None:
            from skillforge.repository.incremental_evolution import validate_production_result

            validate_production_result(self, source_dna, source_compilation, plan, source_knowledge)
            object.__setattr__(
                self,
                "result_id",
                fingerprint(
                    (
                        "incremental-evolution-v2",
                        self.plan_id,
                        self.resulting_project_dna,
                        self.resulting_compilation,
                        self.production_knowledge,
                    )
                ),
            )
            return
        if not isinstance(plan, EvolutionPlan) or not isinstance(source_dna, ProjectDNA):
            raise ValueError("result requires source artifacts")
        if not isinstance(source_compilation, SkillCompilationResult):
            raise ValueError("result requires source compilation")
        if (not isinstance(semantic_workflows, SemanticWorkflowResult)
                or not isinstance(semantic_antipatterns, SemanticAntiPatternResult)):
            raise ValueError("result requires semantic decision artifacts")
        from skillforge.repository.skill_compiler import SkillCompiler

        compiler = SkillCompiler()
        if (source_compilation != compiler.compile(source_dna, source_semantic_structural)
                or self.resulting_compilation != compiler.compile(
                    self.resulting_project_dna, source_semantic_structural)):
            raise ValueError("result compilations must come from the existing SkillCompiler")
        if (self.repository_id != source_dna.repository_id
                or self.source_snapshot_id != source_dna.snapshot_id
                or self.result_snapshot_id != self.resulting_project_dna.snapshot_id
                or self.source_head_commit != plan.source_head_commit
                or self.target_head_commit != plan.target_head_commit
                or self.plan_id != plan.plan_id
                or self.semantic_workflow_result_id != fingerprint(semantic_workflows)
                or self.semantic_antipattern_result_id != fingerprint(semantic_antipatterns)
                or semantic_workflows.repository_id != self.repository_id
                or semantic_antipatterns.repository_id != self.repository_id
                or self.resulting_history_state.state_id != plan.history_state_id
                or self.resulting_compilation.source_snapshot_id != self.result_snapshot_id
                or source_compilation.source_snapshot_id != self.source_snapshot_id):
            raise ValueError("result source artifact mismatch")
        if ({item.candidate_id for item in (*semantic_workflows.workflows,
                                             *semantic_workflows.rejections)}
                != set(plan.workflow_refresh_ids)
                or {item.candidate_id for item in (*semantic_antipatterns.anti_patterns,
                                                   *semantic_antipatterns.rejections)}
                != set(plan.antipattern_refresh_ids)):
            raise ValueError("result semantic decisions do not match plan")
        old = {item.skill_id: item for item in source_compilation.skills}
        new = {item.skill_id: item for item in self.resulting_compilation.skills}
        expected = (tuple(sorted(new.keys() - old.keys())),
                    tuple(sorted(old.keys() - new.keys())),
                    tuple(sorted(key for key in old.keys() & new.keys()
                                 if old[key].version != new[key].version)),
                    tuple(sorted(key for key in old.keys() & new.keys()
                                 if old[key].version == new[key].version)))
        if expected != (self.skill_added_ids, self.skill_removed_ids,
                        self.skill_version_changed_ids, self.skill_unchanged_ids):
            raise ValueError("result Skill diff does not match compilations")
        if (self.rule_review_ids != plan.rule_review_ids
                or self.resulting_project_dna.rules != source_dna.rules):
            raise ValueError("Evolution cannot alter Rules")
        for name in ("refreshed_workflow_ids", "added_workflow_ids",
                     "unresolved_workflow_ids", "refreshed_antipattern_ids",
                     "added_antipattern_ids", "unresolved_antipattern_ids"):
            canonical_ids(getattr(self, name), name)
        for (old_items, new_items, accepted, rejected, missing,
             refreshed, added, unresolved, key) in (
            (source_dna.workflows, self.resulting_project_dna.workflows,
             semantic_workflows.workflows, semantic_workflows.rejections,
             plan.missing_workflow_candidate_ids, self.refreshed_workflow_ids,
             self.added_workflow_ids, self.unresolved_workflow_ids, "workflow_id"),
            (source_dna.anti_patterns, self.resulting_project_dna.anti_patterns,
             semantic_antipatterns.anti_patterns, semantic_antipatterns.rejections,
             plan.missing_antipattern_candidate_ids, self.refreshed_antipattern_ids,
             self.added_antipattern_ids, self.unresolved_antipattern_ids,
             "anti_pattern_id"),
        ):
            old = {getattr(item, key): item for item in old_items}
            new = {getattr(item, key): item for item in new_items}
            accepted_ids = {item.draft_id for item in accepted}
            if tuple(sorted(accepted_ids & old.keys())) != refreshed:
                raise ValueError("result refreshed knowledge does not match accepted drafts")
            if tuple(sorted(accepted_ids - old.keys())) != added:
                raise ValueError("result added knowledge does not match accepted drafts")
            if any(item.draft_id in old and
                   old[item.draft_id].draft.candidate_id != item.candidate_id
                   for item in accepted):
                raise ValueError("accepted draft changed an existing candidate identity")
            if any(identifier not in new or new[identifier].draft != item
                   for item in accepted for identifier in (item.draft_id,)):
                raise ValueError("result DNA does not contain accepted semantic draft")
            unresolved_candidates = set(missing) | {
                item.candidate_id for item in rejected
            }
            expected_unresolved = tuple(sorted(
                getattr(item, key) for item in old_items
                if item.draft.candidate_id in unresolved_candidates
            ))
            if unresolved != expected_unresolved:
                raise ValueError("result unresolved knowledge does not match decisions")
        for old_items, new_items, key, refreshed, added, allowed in (
            (source_dna.workflows, self.resulting_project_dna.workflows, "workflow_id",
             self.refreshed_workflow_ids, self.added_workflow_ids,
             set(plan.workflow_refresh_ids)),
            (source_dna.anti_patterns, self.resulting_project_dna.anti_patterns,
             "anti_pattern_id", self.refreshed_antipattern_ids,
             self.added_antipattern_ids, set(plan.antipattern_refresh_ids)),
        ):
            old = {getattr(item, key): item for item in old_items}
            new = {getattr(item, key): item for item in new_items}
            if set(added) != new.keys() - old.keys() or old.keys() - new.keys():
                raise ValueError("result knowledge additions/removals mismatch")
            if not set(refreshed) <= old.keys() & new.keys():
                raise ValueError("result refreshed knowledge IDs mismatch")
            if any(old[item] != new[item] for item in old.keys() - set(refreshed)):
                raise ValueError("unaffected knowledge was modified")
            if any(item.draft.candidate_id not in allowed for item in (*(
                new[identifier] for identifier in refreshed),
                *(new[identifier] for identifier in added))):
                raise ValueError("result knowledge was not selected by plan")
        object.__setattr__(self, "result_id", fingerprint(("incremental-evolution-v1",
            self.repository_id, self.source_snapshot_id, self.result_snapshot_id,
            self.plan_id, self.semantic_workflow_result_id,
            self.semantic_antipattern_result_id, self.resulting_project_dna,
            self.resulting_compilation)))

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        if self.production_knowledge is None:
            value.pop("production_knowledge")
        return value
