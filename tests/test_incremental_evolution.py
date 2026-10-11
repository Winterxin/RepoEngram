"""Real Git facts and bounded evolution contracts; no network or live model."""

import hashlib
import os
import subprocess
from dataclasses import replace

import pytest

from skillforge.domain.change import GitHistoryResult
from skillforge.domain.evidence import Confidence
from skillforge.domain.evolution import EvolutionHistoryState, GitHistoryDelta
from skillforge.domain.feedback import FeedbackKind, FeedbackTargetKind, HumanFeedbackLedger
from skillforge.domain.negative_patterns import NegativePatternCandidateResult
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectRuleException,
    ProjectScope,
    ProjectScopeKind,
    ProjectWorkflow,
    RuleType,
)
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDraft,
    SemanticAntiPatternRejection,
    SemanticAntiPatternResult,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticStepClassification,
    SemanticWorkflowDraft,
    SemanticWorkflowRejection,
    SemanticWorkflowResult,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.workflow import WorkflowCandidateResult
from skillforge.errors import GitHistoryError
from skillforge.feedback.service import HumanFeedbackService
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.incremental_evolution import IncrementalEvolution
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout.decode().strip()


def dated_git(repo, date, *args):
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, env=env).stdout.decode().strip()


def commit(repo, group, number):
    for name in ("a", "b", "c"):
        (repo / f"{group}_{name}.py").write_text(
            f"number = {number}\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", f"{group} {number}")
    return git(repo, "rev-parse", "HEAD")


def setup(repo):
    git(repo, "init", "-qb", "main")
    commit(repo, "base", 0)
    commit(repo, "alpha", 1)
    commit(repo, "alpha", 2)
    commit(repo, "beta", 1)
    return commit(repo, "beta", 2)


def draft(candidate, *, name="Workflow", evidence_id=None):
    key = evidence_id or candidate.evidence_ids[0]
    refs = (key,)
    method = "llm_semantic_workflow_v1"
    draft_id = hashlib.sha256(
        f"{method}\0{candidate.repository_id}\0{candidate.candidate_id}".encode()
    ).hexdigest()[:24]
    return SemanticWorkflowDraft(
        draft_id, candidate.repository_id,
        candidate.candidate_id, name, refs, "Coordinate changes", refs, (),
        (SemanticWorkflowStep("step", "Review changes", refs),),
        (SemanticStepClassification("step", "required", refs),), (), (),
        SemanticWorkflowScope(SemanticWorkflowScopeKind.REPOSITORY, (),
                              "repository", "observed", refs), (), refs,
        candidate.support, Confidence(0.8, method, candidate.support),
    )


def baseline_dna(state, status=ProjectKnowledgeStatus.CONFIRMED):
    available = {item.evidence_id: item for item in state.workflow_candidates.evidence}
    workflows = tuple(sorted((ProjectWorkflow(
        item.draft_id, state.repository_id, status, item)
        for item in (draft(candidate) for candidate in state.workflow_candidates.candidates)),
        key=lambda item: item.workflow_id))
    refs = sorted({key for item in workflows for key in item.draft.evidence_ids})
    return ProjectDNA(state.repository_id, (), workflows, (),
                      tuple(available[key] for key in refs))


def evolution_fixture(repo):
    base = setup(repo)
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution()
    source_state = evolution.make_state(miner.mine(repo, 4), base, 4,
                                        analyzer.analyze(repo))
    source = baseline_dna(source_state)
    compilation = SkillCompiler().compile(source)
    commit(repo, "alpha", 3)
    delta = miner.mine_delta(repo, base)
    static = analyzer.analyze(repo)
    target = evolution.advance(source_state, delta, static)
    plan = evolution.plan(source, compilation, source_state, target, delta)
    return evolution, source_state, source, compilation, delta, static, target, plan


def rejected_anti(target, plan):
    candidates = {item.candidate_id: item for item in target.antipattern_candidates.candidates}
    rejected = tuple(SemanticAntiPatternRejection(
        target.repository_id, key,
        SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE, "insufficient",
        candidates[key].evidence_ids,
        Confidence(0.2, "llm_semantic_antipattern_v1"))
        for key in plan.antipattern_refresh_ids)
    by_id = {item.evidence_id: item for item in target.antipattern_candidates.evidence}
    refs = sorted({key for item in rejected for key in item.evidence_ids})
    return SemanticAntiPatternResult(target.repository_id, (), rejected,
                                     tuple(by_id[key] for key in refs))


def rejected_workflows(target, plan):
    candidates = {item.candidate_id: item for item in target.workflow_candidates.candidates}
    rejected = tuple(SemanticWorkflowRejection(
        target.repository_id, key, SemanticDecisionKind.INSUFFICIENT_EVIDENCE,
        "insufficient", (candidates[key].evidence_ids[0],),
        Confidence(0.2, "llm_semantic_workflow_v1", candidates[key].support),
        candidates[key].support) for key in plan.workflow_refresh_ids)
    by_id = {item.evidence_id: item for item in target.workflow_candidates.evidence}
    refs = sorted({key for item in rejected for key in item.evidence_ids})
    return SemanticWorkflowResult(target.repository_id, (),
                                  tuple(by_id[key] for key in refs), rejected)


def reject_result_edit(result, source, compilation, plan, semantic_w, semantic_a,
                       **changes):
    with pytest.raises(ValueError, match="unresolved knowledge"):
        replace(result, source_dna=source, source_compilation=compilation, plan=plan,
                semantic_workflows=semantic_w, semantic_antipatterns=semantic_a,
                **changes)


def test_exact_delta_empty_divergent_and_rename(tmp_path):
    base = setup(tmp_path)
    miner = GitHistoryMiner()
    empty = miner.mine_delta(tmp_path, base)
    assert empty.changesets == empty.evidence == ()
    assert empty.delta_id == miner.mine_delta(tmp_path, base).delta_id
    git(tmp_path, "mv", "alpha_a.py", "renamed.py")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "rename")
    head = git(tmp_path, "rev-parse", "HEAD")
    delta = miner.mine_delta(tmp_path, base)
    assert [item.commit_id for item in delta.changesets] == [head]
    assert delta.changesets[0].files[0].previous_path == "alpha_a.py"
    assert delta.changesets[0].files[0].path == "renamed.py"
    assert miner.mine_delta(tmp_path, base).delta_id == delta.delta_id
    with pytest.raises(GitHistoryError, match="ancestor"):
        miner.mine_delta(tmp_path, head, base)
    with pytest.raises(GitHistoryError, match="does not exist"):
        miner.mine_delta(tmp_path, "0" * 40)


def test_bounded_window_equivalence_and_candidate_selectivity(tmp_path):
    base = setup(tmp_path)
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution()
    state = evolution.make_state(miner.mine(tmp_path, 4), base, 4,
                                 analyzer.analyze(tmp_path))
    assert len(state.workflow_candidates.candidates) >= 2
    commit(tmp_path, "alpha", 3)
    delta = miner.mine_delta(tmp_path, base)
    static = analyzer.analyze(tmp_path)
    target = evolution.advance(state, delta, static)
    full = evolution.make_state(miner.mine(tmp_path, 4), delta.head_commit, 4, static)
    assert target.history == full.history
    assert target.workflow_candidates == full.workflow_candidates
    assert target.antipattern_candidates == full.antipattern_candidates
    assert target.state_id == full.state_id
    evicted_commit = state.history.changesets[-1].commit_id
    assert all(item.location.commit != evicted_commit for item in target.history.evidence)
    old = {item.candidate_id: item for item in state.workflow_candidates.candidates}
    changed = [item for item in target.workflow_candidates.candidates
               if old[item.candidate_id] != item]
    unchanged = [item for item in target.workflow_candidates.candidates
                 if old[item.candidate_id] == item]
    assert changed and unchanged
    assert all("alpha" in item.members[0].value for item in changed)
    assert all("beta" in item.members[0].value for item in unchanged)


def test_merge_window_uses_current_head_git_order_without_old_diffs(tmp_path, monkeypatch):
    git(tmp_path, "init", "-qb", "main")
    (tmp_path / "root.py").write_text("root = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    dated_git(tmp_path, "2026-01-01T00:00:00+00:00", "-c", "user.name=Test",
              "-c", "user.email=test@example.invalid", "commit", "-qm", "A")
    root = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "switch", "-qc", "side")
    for name in ("a", "b", "c"):
        (tmp_path / f"side_{name}.py").write_text("side = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    dated_git(tmp_path, "2026-01-02T00:00:00+00:00", "-c", "user.name=Test",
              "-c", "user.email=test@example.invalid", "commit", "-qm", "S")
    side = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "switch", "-q", "main")
    for name in ("a", "b", "c"):
        (tmp_path / f"alpha_{name}.py").write_text("alpha = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    dated_git(tmp_path, "2026-01-03T00:00:00+00:00", "-c", "user.name=Test",
              "-c", "user.email=test@example.invalid", "commit", "-qm", "fix B")
    base = git(tmp_path, "rev-parse", "HEAD")
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution(change_miner=ChangePatternMiner(min_support=1))
    state = evolution.make_state(miner.mine(tmp_path, 2), base, 2,
                                 analyzer.analyze(tmp_path))
    dated_git(tmp_path, "2026-01-04T00:00:00+00:00", "-c", "user.name=Test",
              "-c", "user.email=test@example.invalid", "merge", "--no-ff",
              "-qm", "M", "side")
    head = git(tmp_path, "rev-parse", "HEAD")
    assert git(tmp_path, "log", "-2", "--format=%H", "HEAD").splitlines() == [head, base]
    observed = []
    original = miner._read_file_changes

    def read_new_only(repo, parsed):
        observed.append(parsed.commit_id)
        return original(repo, parsed)

    miner._read_file_changes = read_new_only
    delta = miner.mine_delta(tmp_path, base)
    assert [item.commit_id for item in delta.changesets] == [head, side]
    assert observed == [head, side]
    assert [item.commit_id for item in (*delta.changesets, *state.history.changesets)[:2]] == [
        head, side]
    with pytest.raises(ValueError, match="requires bounded Git commit order"):
        evolution.advance(state, delta, analyzer.analyze(tmp_path))
    static = analyzer.analyze(tmp_path)
    def forbidden_diff(*_args, **_kwargs):
        raise AssertionError("advance must not read any Git diff")

    with monkeypatch.context() as patch:
        patch.setattr(GitHistoryMiner, "_read_file_changes", forbidden_diff)
        target = evolution.advance(state, delta, static, repo_path=tmp_path)
    full = evolution.make_state(miner.mine(tmp_path, 2), head, 2, static)
    assert tuple(item.commit_id for item in target.history.changesets) == (head, base)
    assert target.history == full.history
    assert evolution.change_miner.mine(target.history, static) == (
        evolution.change_miner.mine(full.history, static))
    assert target.workflow_candidates == full.workflow_candidates
    assert target.antipattern_candidates == full.antipattern_candidates
    assert target.state_id == full.state_id
    assert root not in {item.commit_id for item in target.history.changesets}


def test_support_growth_and_decay_from_mined_facts(tmp_path):
    base = setup(tmp_path)
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution()
    state = evolution.make_state(miner.mine(tmp_path, 5), base, 5,
                                 analyzer.analyze(tmp_path))
    old = {item.pattern_id: item for item in ChangePatternMiner().mine(
        state.history, analyzer.analyze(tmp_path)).patterns}
    commit(tmp_path, "alpha", 3)
    grown = evolution.advance(state, miner.mine_delta(tmp_path, base),
                              analyzer.analyze(tmp_path))
    new = {item.pattern_id: item for item in ChangePatternMiner().mine(
        grown.history, analyzer.analyze(tmp_path)).patterns}
    ids = [key for key, item in old.items() if "alpha" in item.items[0].value]
    assert any(new[key].support.supporting > old[key].support.supporting for key in ids)
    for index in range(3):
        before = grown.head_commit
        commit(tmp_path, "other", index)
        grown = evolution.advance(grown, miner.mine_delta(tmp_path, before),
                                  analyzer.analyze(tmp_path))
    decayed = {item.pattern_id: item for item in ChangePatternMiner().mine(
        grown.history, analyzer.analyze(tmp_path)).patterns}
    assert any(key not in decayed or decayed[key].support.supporting < new[key].support.supporting
               for key in ids)


def test_plan_status_and_skill_version_propagation(tmp_path):
    evolution, state, source, compilation, delta, static, target, plan = evolution_fixture(
        tmp_path)
    assert plan.changed_workflow_candidate_ids
    assert not plan.new_workflow_candidate_ids
    assert not plan.missing_workflow_candidate_ids
    assert any("alpha" in item.members[0].value for item in
               target.workflow_candidates.candidates
               if item.candidate_id in plan.workflow_refresh_ids)
    assert all("beta" not in item.members[0].value for item in
               target.workflow_candidates.candidates
               if item.candidate_id in plan.workflow_refresh_ids)
    with pytest.raises(ValueError, match="candidate diff"):
        replace(plan, changed_workflow_candidate_ids=(), source_dna=source,
                source_state=state, target_state=target, delta=delta,
                source_compilation=compilation, feedback=None)
    selected = [item for item in target.workflow_candidates.candidates
                if item.candidate_id in plan.workflow_refresh_ids]
    by_id = {item.evidence_id: item for item in target.workflow_candidates.evidence}
    drafts = tuple(draft(item, evidence_id=next(
        key for key in item.evidence_ids if by_id[key].location.commit == delta.head_commit
    )) for item in selected)
    refs = sorted({key for item in drafts for key in item.evidence_ids})
    semantic = SemanticWorkflowResult(source.repository_id, drafts,
                                      tuple(by_id[key] for key in refs))
    semantic_anti = rejected_anti(target, plan)
    result = evolution.apply_semantic(source, compilation, target, plan, semantic,
                                      semantic_anti)
    assert result.refreshed_workflow_ids
    assert result.skill_version_changed_ids
    assert result.skill_unchanged_ids
    assert all(item.status == ProjectKnowledgeStatus.CONFIRMED
               for item in result.resulting_project_dna.workflows)
    assert result.resulting_project_dna.rules == source.rules
    assert result.resulting_history_state.state_id == target.state_id
    with pytest.raises(ValueError, match="Skill diff"):
        replace(result, skill_unchanged_ids=(), source_dna=source,
                source_compilation=compilation, plan=plan,
                semantic_workflows=semantic, semantic_antipatterns=semantic_anti)
    reject_result_edit(result, source, compilation, plan, semantic, semantic_anti,
                       unresolved_workflow_ids=result.refreshed_workflow_ids[:1])


def test_plan_rejects_foreign_source_state_at_service_and_domain_boundaries():
    head = "a" * 40

    def empty_state(repository_id):
        return EvolutionHistoryState(
            repository_id, head, 2,
            GitHistoryResult(repository_id, (), ()),
            WorkflowCandidateResult(repository_id, (), (), ()),
            NegativePatternCandidateResult(repository_id, (), (), (), 0),
        )

    source = ProjectDNA("local", (), (), (), ())
    compilation = SkillCompiler().compile(source)
    local_state = empty_state("local")
    foreign_state = empty_state("foreign")
    delta = GitHistoryDelta("local", head, head, (), ())
    evolution = IncrementalEvolution()
    plan = evolution.plan(source, compilation, local_state, local_state, delta)
    assert foreign_state.head_commit == local_state.head_commit
    assert foreign_state.workflow_candidates.candidates == (
        local_state.workflow_candidates.candidates
    )
    assert foreign_state.antipattern_candidates.candidates == (
        local_state.antipattern_candidates.candidates)
    with pytest.raises(ValueError, match="source DNA/compilation/state mismatch"):
        evolution.plan(source, compilation, foreign_state, local_state, delta)
    with pytest.raises(ValueError, match="plan repository_id mismatch"):
        replace(plan, source_dna=source, source_state=foreign_state,
                target_state=local_state, delta=delta,
                source_compilation=compilation, feedback=None)


def test_changed_semantic_body_resets_status_and_deprecated_sticks(tmp_path):
    evolution, state, source, compilation, delta, _, target, plan = evolution_fixture(
        tmp_path)
    selected = [item for item in target.workflow_candidates.candidates
                if item.candidate_id in plan.workflow_refresh_ids]
    drafts = tuple(draft(item, name="Revised workflow") for item in selected)
    by_id = {item.evidence_id: item for item in target.workflow_candidates.evidence}
    refs = sorted({key for item in drafts for key in item.evidence_ids})
    semantic = SemanticWorkflowResult(source.repository_id, drafts,
                                      tuple(by_id[key] for key in refs))
    result = evolution.apply_semantic(source, compilation, target, plan, semantic,
                                      rejected_anti(target, plan))
    changed = set(result.refreshed_workflow_ids)
    assert all(item.status == ProjectKnowledgeStatus.CANDIDATE
               for item in result.resulting_project_dna.workflows
               if item.workflow_id in changed)
    deprecated = ProjectDNA(source.repository_id, source.rules,
                            tuple(replace(item, status=ProjectKnowledgeStatus.DEPRECATED)
                                  if item.workflow_id in changed else item
                                  for item in source.workflows),
                            source.anti_patterns, source.evidence)
    deprecated_compilation = SkillCompiler().compile(deprecated)
    # The refreshed source snapshot carries explicit human governance forward.
    deprecated_plan = evolution.plan(deprecated, deprecated_compilation, state, target,
                                     delta)
    result = evolution.apply_semantic(deprecated, deprecated_compilation, target,
                                      deprecated_plan, semantic,
                                      rejected_anti(target, deprecated_plan))
    assert all(item.status == ProjectKnowledgeStatus.DEPRECATED
               for item in result.resulting_project_dna.workflows
               if item.workflow_id in changed)


class RejectModel:
    def __init__(self):
        self.calls = []

    def complete(self, request):
        self.calls.append(request.candidate_id)
        return SemanticDecisionResponse(
            request.repository_id, request.candidate_id,
            SemanticDecisionKind.INSUFFICIENT_EVIDENCE, "insufficient",
            (request.evidence[0].evidence_id,), 0.2, request.support, None,
        )


def test_selective_calls_and_rejection_preserves_old_knowledge(tmp_path):
    evolution, _, source, compilation, _, static, target, plan = evolution_fixture(tmp_path)
    model = RejectModel()

    class AntiModel:
        def complete(self, request):
            candidate = next(item for item in target.antipattern_candidates.candidates
                             if item.candidate_id == request.candidate_id)
            from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionResponse
            return SemanticAntiPatternDecisionResponse(
                request.repository_id, request.candidate_id,
                SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE, "insufficient",
                candidate.evidence_ids, 0.2, None)

    result = evolution.apply(source, compilation, target, plan, static, model, AntiModel())
    assert tuple(model.calls) == plan.workflow_refresh_ids
    assert result.resulting_project_dna == source
    assert result.unresolved_workflow_ids
    assert result.skill_unchanged_ids


def test_semantic_workflow_evidence_object_must_match_target(tmp_path):
    evolution, _, source, compilation, _, _, target, plan = evolution_fixture(tmp_path)
    semantic = rejected_workflows(target, plan)
    forged = SemanticWorkflowResult(
        semantic.repository_id, semantic.workflows,
        (replace(semantic.evidence[0], description="forged"), *semantic.evidence[1:]),
        semantic.rejections)
    with pytest.raises(ValueError, match="Evidence differs from target candidate"):
        evolution.apply_semantic(source, compilation, target, plan, forged,
                                  rejected_anti(target, plan))


def test_semantic_antipattern_evidence_object_must_match_target(tmp_path):
    evolution, _, source, compilation, _, _, target, plan = evolution_fixture(tmp_path)
    semantic = rejected_anti(target, plan)
    assert semantic.evidence
    forged = SemanticAntiPatternResult(
        semantic.repository_id, semantic.anti_patterns, semantic.rejections,
        (replace(semantic.evidence[0], provenance="forged"), *semantic.evidence[1:]))
    with pytest.raises(ValueError, match="Evidence differs from target candidate"):
        evolution.apply_semantic(source, compilation, target, plan,
                                  rejected_workflows(target, plan), forged)


def test_unresolved_result_is_exactly_missing_plus_rejected_existing(tmp_path):
    evolution, _, source, compilation, _, _, target, plan = evolution_fixture(tmp_path)
    semantic_w = rejected_workflows(target, plan)
    semantic_a = rejected_anti(target, plan)
    result = evolution.apply_semantic(source, compilation, target, plan,
                                      semantic_w, semantic_a)
    expected = tuple(sorted(item.workflow_id for item in source.workflows
                            if item.draft.candidate_id in plan.workflow_refresh_ids))
    assert result.unresolved_workflow_ids == expected
    assert result.unresolved_antipattern_ids == ()
    reject_result_edit(result, source, compilation, plan, semantic_w, semantic_a,
                       unresolved_workflow_ids=("fake",))
    reject_result_edit(result, source, compilation, plan, semantic_w, semantic_a,
                       unresolved_antipattern_ids=("fake",))
    reject_result_edit(result, source, compilation, plan, semantic_w, semantic_a,
                       unresolved_workflow_ids=())


def test_human_feedback_diagnostics_and_manual_rule_preserved(tmp_path):
    evolution, state, source, _, delta, _, target, _ = evolution_fixture(tmp_path)
    rule = ProjectRule(
        "manual-rule", source.repository_id, RuleType.TESTING,
        "Use a reviewed exception", ProjectScope(ProjectScopeKind.REPOSITORY),
        ProjectKnowledgeStatus.MANUAL_OVERRIDE, (),
        support=None, confidence=None,
        exceptions=(ProjectRuleException("Legacy exception", ()),),
    )
    source = ProjectDNA(source.repository_id, (rule,), source.workflows,
                        source.anti_patterns, source.evidence)
    compilation = SkillCompiler().compile(source)
    service = HumanFeedbackService()
    alpha_candidates = {item.candidate_id for item in state.workflow_candidates.candidates
                        if "alpha" in item.members[0].value}
    alpha_workflows = {item.workflow_id for item in source.workflows
                       if item.draft.candidate_id in alpha_candidates}
    alpha_skill = next(item for item in compilation.skills
                       if item.workflow_id in alpha_workflows)
    skill_feedback = service.record_skill(alpha_skill, FeedbackKind.MISSING_STEP,
                                          "Missing review step")
    registry = SkillRegistryBuilder().build(compilation)
    route = SkillRouter().route(registry, SkillRoutingRequest(
        source.repository_id, "Coordinate changes"))
    route_feedback = service.record_route(route, FeedbackKind.WRONG_ROUTE,
                                          "Wrong choice")
    stale = service.record_knowledge(source, FeedbackTargetKind.PROJECT_WORKFLOW,
                                     source.workflows[0].workflow_id,
                                     FeedbackKind.INACCURATE, "Old claim")
    ledger = HumanFeedbackLedger(source.repository_id,
                                 (skill_feedback, route_feedback, stale))
    plan = evolution.plan(source, compilation, state, target, delta, ledger)
    assert source.workflows[0].draft.candidate_id in plan.diagnostic_workflow_candidate_ids
    assert any(item.draft.candidate_id in plan.diagnostic_workflow_candidate_ids
               for item in source.workflows if item.workflow_id == alpha_skill.workflow_id)
    assert plan.routing_review_feedback_ids == (route_feedback.feedback_id,)
    assert source.rules == (rule,)
    result = evolution.apply_semantic(source, compilation, target, plan,
                                      rejected_workflows(target, plan),
                                      rejected_anti(target, plan))
    assert result.resulting_project_dna.rules == source.rules
    assert result.resulting_project_dna.evidence == source.evidence


def test_new_and_missing_candidates_have_no_guessed_lineage(tmp_path):
    base = setup(tmp_path)
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution()
    state = evolution.make_state(miner.mine(tmp_path, 4), base, 4,
                                 analyzer.analyze(tmp_path))
    source = baseline_dna(state)
    commit(tmp_path, "gamma", 1)
    commit(tmp_path, "gamma", 2)
    delta = miner.mine_delta(tmp_path, base)
    target = evolution.advance(state, delta, analyzer.analyze(tmp_path))
    compilation = SkillCompiler().compile(source)
    plan = evolution.plan(source, compilation, state, target, delta)
    assert plan.new_workflow_candidate_ids
    assert plan.missing_workflow_candidate_ids
    assert any("gamma" in item.members[0].value for item in
               target.workflow_candidates.candidates
               if item.candidate_id in plan.new_workflow_candidate_ids)
    selected = [item for item in target.workflow_candidates.candidates
                if item.candidate_id in plan.workflow_refresh_ids]
    accepted = tuple(draft(item) for item in selected)
    by_id = {item.evidence_id: item for item in target.workflow_candidates.evidence}
    refs = sorted({key for item in accepted for key in item.evidence_ids})
    semantic = SemanticWorkflowResult(source.repository_id, accepted,
                                      tuple(by_id[key] for key in refs))
    semantic_anti = rejected_anti(target, plan)
    result = evolution.apply_semantic(source, compilation, target, plan, semantic,
                                      semantic_anti)
    assert result.added_workflow_ids
    assert result.unresolved_workflow_ids
    assert all(item.status == ProjectKnowledgeStatus.CANDIDATE
               for item in result.resulting_project_dna.workflows
               if item.workflow_id in result.added_workflow_ids)
    assert all(item in {workflow.workflow_id for workflow in result.resulting_project_dna.workflows}
               for item in result.unresolved_workflow_ids)
    reject_result_edit(result, source, compilation, plan, semantic, semantic_anti,
                       unresolved_workflow_ids=())
    rejected_new = rejected_workflows(target, plan)
    rejected_result = evolution.apply_semantic(source, compilation, target, plan,
                                               rejected_new, semantic_anti)
    assert rejected_result.added_workflow_ids == ()
    assert rejected_result.unresolved_workflow_ids == result.unresolved_workflow_ids
    new_candidate = next(item for item in target.workflow_candidates.candidates
                         if item.candidate_id in plan.new_workflow_candidate_ids)
    invented_old_id = draft(new_candidate).draft_id
    reject_result_edit(rejected_result, source, compilation, plan, rejected_new,
                       semantic_anti, unresolved_workflow_ids=tuple(sorted((
                           *rejected_result.unresolved_workflow_ids, invented_old_id))))


def test_antipattern_identity_change_is_unresolved(tmp_path):
    setup(tmp_path)
    base = commit(tmp_path, "alpha", 3)
    miner = GitHistoryMiner()
    analyzer = PythonStaticAnalyzer()
    evolution = IncrementalEvolution()
    state = evolution.make_state(miner.mine(tmp_path, 4), base, 4,
                                 analyzer.analyze(tmp_path))
    candidate = next(item for item in state.antipattern_candidates.candidates
                     if "alpha" in item.affected_paths[0])
    key = candidate.evidence_ids[0]
    method = "llm_semantic_antipattern_v1"
    draft_id = hashlib.sha256(
        f"{method}\0{state.repository_id}\0{candidate.candidate_id}".encode()
    ).hexdigest()[:24]
    draft_anti = SemanticAntiPatternDraft(
        draft_id, state.repository_id, candidate.candidate_id, "Repeated edits", (key,),
        "Recurring correction", (key,), "Avoid repeat", (key,), "History", (key,),
        SemanticAntiPatternScope(SemanticAntiPatternScopeKind.UNKNOWN, (),
                                 "unknown", "unknown", (key,)), (),
        "accepted", (key,), (key,), Confidence(0.5, method),
    )
    baseline = baseline_dna(state)
    by_id = {item.evidence_id: item for item in state.antipattern_candidates.evidence}
    refs = {item.evidence_id: item for item in baseline.evidence}
    refs[key] = by_id[key]
    source = ProjectDNA(state.repository_id, baseline.rules, baseline.workflows,
                        (ProjectAntiPattern(draft_id, state.repository_id,
                                            ProjectKnowledgeStatus.CONFIRMED,
                                            draft_anti),),
                        tuple(refs[item] for item in sorted(refs)))
    compilation = SkillCompiler().compile(source)
    empty_delta = miner.mine_delta(tmp_path, base)
    review = HumanFeedbackService().record_knowledge(
        source, FeedbackTargetKind.PROJECT_ANTI_PATTERN, draft_id,
        FeedbackKind.INACCURATE, "Review this risk")
    diagnostic_plan = evolution.plan(
        source, compilation, state, state, empty_delta,
        HumanFeedbackLedger(source.repository_id, (review,)))
    assert diagnostic_plan.antipattern_refresh_ids == (candidate.candidate_id,)
    unchanged_semantic = SemanticAntiPatternResult(
        source.repository_id, (replace(draft_anti, confidence=Confidence(0.8, method)),),
        (), (by_id[key],))
    empty_workflows = SemanticWorkflowResult(source.repository_id, (), ())
    preserved = evolution.apply_semantic(
        source, compilation, state, diagnostic_plan,
        empty_workflows, unchanged_semantic)
    assert preserved.resulting_project_dna.anti_patterns[0].status == (
        ProjectKnowledgeStatus.CONFIRMED)
    reject_result_edit(preserved, source, compilation, diagnostic_plan,
                       empty_workflows, unchanged_semantic,
                       unresolved_antipattern_ids=(draft_id,))
    rejected_existing = rejected_anti(state, diagnostic_plan)
    rejected_result = evolution.apply_semantic(
        source, compilation, state, diagnostic_plan,
        empty_workflows, rejected_existing)
    assert rejected_result.unresolved_antipattern_ids == (draft_id,)
    reject_result_edit(rejected_result, source, compilation, diagnostic_plan,
                       empty_workflows, rejected_existing,
                       unresolved_antipattern_ids=())
    changed_semantic = SemanticAntiPatternResult(
        source.repository_id, (replace(draft_anti, description="Revised risk"),),
        (), (by_id[key],))
    reset = evolution.apply_semantic(
        source, compilation, state, diagnostic_plan,
        SemanticWorkflowResult(source.repository_id, (), ()), changed_semantic)
    assert reset.resulting_project_dna.anti_patterns[0].status == (
        ProjectKnowledgeStatus.CANDIDATE)
    commit(tmp_path, "alpha", 4)
    delta = miner.mine_delta(tmp_path, base)
    target = evolution.advance(state, delta, analyzer.analyze(tmp_path))
    plan = evolution.plan(source, compilation, state, target, delta)
    assert candidate.candidate_id in plan.missing_antipattern_candidate_ids
    assert plan.new_antipattern_candidate_ids
    semantic_w = rejected_workflows(target, plan)
    semantic_a = rejected_anti(target, plan)
    result = evolution.apply_semantic(source, compilation, target, plan,
                                      semantic_w, semantic_a)
    assert draft_id in result.unresolved_antipattern_ids
    assert result.resulting_project_dna.anti_patterns == source.anti_patterns
    reject_result_edit(result, source, compilation, plan, semantic_w, semantic_a,
                       unresolved_antipattern_ids=())


def test_stale_feedback_target_is_unresolved_without_guess(tmp_path):
    evolution, state, source, _, delta, _, target, _ = evolution_fixture(tmp_path)
    record = HumanFeedbackService().record_knowledge(
        source, FeedbackTargetKind.PROJECT_WORKFLOW, source.workflows[0].workflow_id,
        FeedbackKind.INACCURATE, "stale review")
    kept = source.workflows[1:]
    refs = {key for item in kept for key in item.draft.evidence_ids}
    pruned = ProjectDNA(source.repository_id, (), kept, (),
                        tuple(item for item in source.evidence if item.evidence_id in refs))
    compilation = SkillCompiler().compile(pruned)
    plan = evolution.plan(pruned, compilation, state, target, delta,
                          HumanFeedbackLedger(source.repository_id, (record,)))
    assert plan.unresolved_feedback_ids == (record.feedback_id,)
    assert plan.diagnostic_workflow_candidate_ids == ()


def test_verify_feedback_requests_review_without_rewriting_report(tmp_path):
    from test_human_feedback import make_verify_report
    from test_skill_compiler import dna as fixture_dna

    source, report = make_verify_report(tmp_path, fixture_dna())
    report_before = report.to_dict()
    root = tmp_path / "feedback-verify-repo"
    head = git(root, "rev-parse", "HEAD")
    miner = GitHistoryMiner()
    evolution = IncrementalEvolution()
    state = evolution.make_state(miner.mine(root, 4), head, 4,
                                 PythonStaticAnalyzer().analyze(root))
    delta = miner.mine_delta(root, head)
    target = evolution.advance(state, delta, PythonStaticAnalyzer().analyze(root))
    compilation = SkillCompiler().compile(source)
    feedback = HumanFeedbackService().record_verify(
        report, FeedbackKind.FALSE_POSITIVE, "False alarm", check_id="probe")
    plan = evolution.plan(source, compilation, state, target, delta,
                          HumanFeedbackLedger(source.repository_id, (feedback,)))
    assert plan.verify_review_feedback_ids == (feedback.feedback_id,)
    assert report.to_dict() == report_before
