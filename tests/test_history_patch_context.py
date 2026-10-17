"""Cold and incremental inputs share real bounded patch references and original gates."""

from dataclasses import replace

import pytest
from test_git_history import _commit, _init_repo, _write

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
)
from skillforge.domain.semantic_workflow import SemanticDecisionKind, SemanticDecisionResponse
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.history_patch_context import (
    _distinct_commit_examples_first,
    enrich_selected_history,
    strip_git_patch_evidence,
)
from skillforge.repository.incremental_evolution import IncrementalEvolution
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_antipattern_prompt import build_semantic_antipattern_prompt
from skillforge.repository.semantic_antipatterns import (
    SemanticAntiPatternAbstracter,
    build_semantic_antipattern_request,
)
from skillforge.repository.semantic_evaluation import select_semantic_candidates
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    build_semantic_workflow_request,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner


def _inputs(root):
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    workflows = WorkflowCandidateMiner().mine(history, facts, patterns)
    negative = NegativePatternMiner().mine(history)
    return history, facts, workflows, negative


def _repository(root):
    _init_repo(root)
    for index in range(4):
        _write(root, "service.py", "def normalize(value):\n    return value.strip()\n"
               + f"REVISION = {index}\n")
        _write(root, "test_service.py", "from service import normalize\n"
               "def test_normalize():\n    assert normalize('  a ') == 'a'\n"
               + f"REVISION = {index}\n")
        _commit(root, "fix normalized output boundary")


def test_two_examples_cover_distinct_revisions_before_more_files():
    pairs = (("new", "service.py"), ("new", "tests/test_service.py"),
             ("old", "service.py"), ("old", "tests/test_service.py"))
    selected = _distinct_commit_examples_first(pairs)
    assert selected[:2] == (("new", "service.py"), ("old", "service.py"))
    assert selected[2:] == (("new", "tests/test_service.py"),
                            ("old", "tests/test_service.py"))
    assert set(selected) == set(pairs)


def test_single_revision_keeps_second_file_as_fallback():
    pairs = (("only", "service.py"), ("only", "tests/test_service.py"))
    assert _distinct_commit_examples_first(pairs) == pairs
    assert _distinct_commit_examples_first(()) == ()


def test_leading_repeated_candidate_keeps_comparison_under_fixed_read_budget(tmp_path):
    _init_repo(tmp_path)
    for family in range(8):
        _write(tmp_path, f"component_{family}.py", "def normalize(value):\n    return value\n")
        _write(tmp_path, f"test_component_{family}.py", "VALUE = 0\n")
    _commit(tmp_path, "initial components")
    for family in range(8):
        for revision in range(2):
            _write(tmp_path, f"component_{family}.py",
                   f"def normalize(value):\n    return value + {revision + 1}\n")
            _write(tmp_path, f"test_component_{family}.py", f"VALUE = {revision + 1}\n")
            _commit(tmp_path, "fix component behavior with regression coverage")
    history, facts, workflows, negative = _inputs(tmp_path)
    leading = next(item for item in select_semantic_candidates(workflows, limit=5)
                   if item.kind in {ChangePatternKind.FILE_CO_CHANGE,
                                    ChangePatternKind.MODULE_CO_CHANGE})
    paths = {member.value for member in leading.members}
    paths.update(module.path for module in facts.modules
                 if module.module in leading.current_modules)
    enriched = enrich_selected_history(tmp_path, history, facts, workflows, negative)
    examples = {
        item.location.commit for item in enriched.evidence
        if item.location.commit in leading.supporting_commits
        and item.location.path in paths
    }
    assert len(examples) == 2
    assert enriched.git_reads <= 16
    assert enriched.added_reference_bytes <= 16384
    assert strip_git_patch_evidence(enriched.history) == history


def test_shared_cold_and_incremental_history_preserves_ids_and_actual_reference_closure(tmp_path):
    _repository(tmp_path)
    history, facts, workflows, negative = _inputs(tmp_path)
    enriched = enrich_selected_history(tmp_path, history, facts, workflows, negative)
    patches = [item for item in enriched.history.evidence if item.provenance == "git_patch"]
    assert patches and enriched.git_reads <= 16 and enriched.added_reference_bytes <= 16384
    assert strip_git_patch_evidence(enriched.history) == history
    state = IncrementalEvolution().make_state(
        history, history.changesets[0].commit_id, 50, facts, repo_path=tmp_path,
    )
    assert state.history == enriched.history
    assert IncrementalEvolution().make_state(
        state.history, state.head_commit, 50, facts, repo_path=tmp_path,
    ) == state
    old = {item.candidate_id: item for item in workflows.candidates}
    assert old.keys() == {item.candidate_id for item in state.workflow_candidates.candidates}
    for candidate in state.workflow_candidates.candidates:
        before = old[candidate.candidate_id]
        assert replace(candidate, evidence_ids=before.evidence_ids,
                       noise_signals=before.noise_signals) == before
        request = build_semantic_workflow_request(
            state.history, facts, state.workflow_candidates, candidate,
        )
        assert any(item.provenance == "git_patch" for item in request.evidence)
    previous = {item.candidate_id: item for item in negative.candidates}
    for candidate in state.antipattern_candidates.candidates:
        assert replace(candidate, evidence_ids=previous[candidate.candidate_id].evidence_ids) == (
            previous[candidate.candidate_id]
        )
        request = build_semantic_antipattern_request(
            state.history, state.antipattern_candidates, candidate,
        )
        assert any(item.provenance == "git_patch" for item in request.evidence)
        assert "complete small historical hunk" in build_semantic_antipattern_prompt(request)


def test_saved_patch_cannot_be_relocated_or_silently_stripped(tmp_path):
    _repository(tmp_path)
    history, facts, workflows, negative = _inputs(tmp_path)
    enriched = enrich_selected_history(tmp_path, history, facts, workflows, negative).history
    patch = next(item for item in enriched.evidence if item.provenance == "git_patch")
    broken = replace(patch, location=replace(patch.location, line=999))
    changed = replace(enriched, evidence=tuple(broken if item == patch else item
                                              for item in enriched.evidence))
    with pytest.raises(ValueError, match="patch"):
        strip_git_patch_evidence(changed)
    with pytest.raises(ValueError, match="patch"):
        NegativePatternMiner().mine(changed)


def test_patch_examples_keep_legal_rejection_and_original_decision_citation_gates(tmp_path):
    _repository(tmp_path)
    history, facts, workflows, negative = _inputs(tmp_path)
    enriched = enrich_selected_history(tmp_path, history, facts, workflows, negative).history
    updated = WorkflowCandidateMiner().mine(
        enriched, facts, ChangePatternMiner().mine(enriched, facts),
    )

    class RejectingModel:
        def complete(self, request):
            return SemanticDecisionResponse(
                request.repository_id, request.candidate_id,
                SemanticDecisionKind.INSUFFICIENT_EVIDENCE,
                "A hunk does not establish an ordered workflow", (request.evidence[0].evidence_id,),
                0.8, request.support, None,
            )

    result = SemanticWorkflowAbstracter().abstract(enriched, facts, updated, RejectingModel())
    assert not result.workflows and len(result.rejections) == len(updated.candidates)
    negative = NegativePatternMiner().mine(enriched)

    class RejectingAntiModel:
        def complete(self, request):
            return SemanticAntiPatternDecisionResponse(
                request.repository_id, request.candidate_id,
                SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
                "A revision alone does not establish avoidable behavior",
                tuple(item.evidence_id for item in request.evidence), 0.8, None,
            )

    result = SemanticAntiPatternAbstracter().abstract(enriched, negative, RejectingAntiModel())
    assert not result.anti_patterns and len(result.rejections) == len(negative.candidates)
