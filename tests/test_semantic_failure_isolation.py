"""Production prepare regressions for independent semantic failure boundaries."""

from dataclasses import replace

import pytest
from test_prepare import (
    AntiPatternModel,
    FakeModel,
    WorkflowModel,
)
from test_prepare import (
    historical_repository as historical_repository,
)
from test_prepare import (
    repository as repository,
)

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind
from skillforge.domain.structural_rules import StructuralRuleMiningResult
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.productization.semantic_assembly import evaluate_semantic_knowledge
from skillforge.repository.project_dna import ProjectDNABuilder


def test_invalid_workflow_channel_preserves_structural(historical_repository):
    class InvalidWorkflow(WorkflowModel):
        def complete(self, request):
            return replace(super().complete(request), decision_evidence_ids=("unknown",))

    model = InvalidWorkflow()
    result = SkillForgePreparer(
        FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE),
        model, AntiPatternModel(),
    ).prepare(PrepareRequest(
        historical_repository, "task", CodeAgentTarget.CODEX, "app.api.route0",
    ))
    assert result.workflow_accepted_count == 0
    assert result.semantic_accepted_count == 1
    assert result.compiled_skill_count >= 1
    assert result.semantic_failures
    assert all(item["status"] == "INVALID" for item in result.semantic_failures)
    assert len(model.requests) == 2 * result.workflow_semantic_selected_count
    assert result.to_dict()["semantic_channels"]["workflow"] == "failed"
    assert result.to_dict()["semantic_channels"]["structural"] == "ok"


def test_all_structural_invalid_produces_no_knowledge(repository):
    class Invalid(FakeModel):
        def complete(self, request):
            return replace(super().complete(request), decision_evidence_ids=("unknown",))

    result = SkillForgePreparer(
        Invalid(SemanticStructuralDecisionKind.ARCHITECTURE_RULE),
    ).prepare(PrepareRequest(repository, "task", CodeAgentTarget.CODEX))
    assert result.dna_rule_count == result.compiled_skill_count == 0
    assert result.installed_packages == ()
    assert result.semantic_failure_count == 1
    assert result.semantic_failures[0]["status"] == "INVALID"


def test_accepted_invalid_accepted_retains_both_and_original_responses():
    from test_semantic_workflows import FakeModel as WorkflowFake
    from test_semantic_workflows import _fixture

    history, facts, candidates = _fixture()
    seed, = candidates.candidates
    candidates = replace(candidates, candidates=tuple(
        replace(seed, candidate_id=key) for key in ("first", "middle", "third")
    ))

    class Model(WorkflowFake):
        def complete(self, request):
            response = super().complete(request)
            if request.candidate_id == "middle":
                return replace(response, decision_evidence_ids=("unknown",))
            return response

    model = Model()
    assembly = evaluate_semantic_knowledge(
        history.repository_id, facts, StructuralRuleMiningResult(history.repository_id, (), ()),
        FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE),
        history=history, candidates=candidates, selected_workflows=candidates.candidates,
        workflow_model=model,
    )
    assert {w.candidate_id for w in assembly.workflows.workflows} == {"first", "third"}
    failure, = assembly.failures
    assert failure["candidate_id"] == "middle" and failure["status"] == "INVALID"
    assert failure["retry_count"] == assembly.workflow_retry_count == 1
    assert len(failure["attempts"]) == 2
    assert failure["attempts"][0]["raw_response"]["decision_evidence_ids"] == ["unknown"]
    assert [r.candidate_id for r in model.requests] == ["first", "middle", "middle", "third"]
    dna = ProjectDNABuilder().build(history.repository_id, workflows=assembly.workflows)
    assert len(dna.workflows) == 2


def test_input_evidence_content_conflict_is_not_isolated():
    from test_semantic_workflows import _fixture

    history, facts, candidates = _fixture()
    conflicting = replace(candidates, evidence=tuple(
        replace(e, description=e.description + " changed") for e in candidates.evidence
    ))
    with pytest.raises(ValueError, match="conflicting semantic Evidence ID"):
        evaluate_semantic_knowledge(
            history.repository_id, facts, StructuralRuleMiningResult(history.repository_id, (), ()),
            FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE),
            history=history, candidates=conflicting,
        )


def test_all_history_candidates_invalid_do_not_create_dna_or_skill(historical_repository):
    from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionKind

    class InvalidWorkflow(WorkflowModel):
        def complete(self, request):
            return replace(super().complete(request), decision_evidence_ids=("unknown",))

    result = SkillForgePreparer(
        FakeModel(SemanticStructuralDecisionKind.INSUFFICIENT_EVIDENCE),
        InvalidWorkflow(), AntiPatternModel(SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE),
    ).prepare(PrepareRequest(historical_repository, "task", CodeAgentTarget.CODEX))
    assert result.dna_rule_count == result.dna_workflow_count == result.dna_anti_pattern_count == 0
    assert result.compiled_skill_count == 0 and result.selected_skill_ids == ()
    assert result.workflow_semantic_retry_count == result.workflow_semantic_selected_count


def test_tendency_invalid_does_not_block_structural(repository):
    from test_prepare import _git

    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Isolation test")
    _git(repository, "config", "user.email", "isolation@example.invalid")
    _git(repository, "add", "src")
    _git(repository, "commit", "-qm", "base")
    for index in range(3):
        (repository / f"src/app/service/extra{index}.py").write_text("VALUE = 1\n")
        _git(repository, "add", "src")
        _git(repository, "commit", "-qm", f"Add service module {index}")

    class InvalidTendency:
        calls = 0

        def complete(self, request):
            from test_semantic_tendencies import response_for

            self.calls += 1
            return replace(response_for(request), evidence_ids=("unknown",))

    tendency = InvalidTendency()
    result = SkillForgePreparer(
        FakeModel(SemanticStructuralDecisionKind.ARCHITECTURE_RULE),
        WorkflowModel(), AntiPatternModel(), tendency,
    ).prepare(PrepareRequest(repository, "task", CodeAgentTarget.CODEX, "app.api.route0"))
    assert tendency.calls >= 1
    assert result.semantic_accepted_count == 1 and result.selected_skill_ids
    assert result.tendency_accepted_count == 0
    failures = [f for f in result.semantic_failures if f["channel"] == "tendency"]
    assert len(failures) == tendency.calls
    assert all(f["retry_count"] == 0 and f["status"] == "INVALID" for f in failures)


@pytest.mark.parametrize("corruption", ["identity", "evidence", "request"])
def test_repository_integrity_failure_remains_fatal(repository, monkeypatch, corruption):
    import skillforge.productization.prepare as prepare
    from skillforge.repository.static_analysis import PythonStaticAnalyzer

    facts = PythonStaticAnalyzer().analyze(repository)
    if corruption == "identity":
        broken = replace(facts, repository_id="other-repository")
    elif corruption == "evidence":
        broken = replace(facts, evidence=tuple(
            replace(e, repository_id="other-repository") for e in facts.evidence
        ))
    else:
        # A missing authoritative source reference must fail before model invocation.
        broken = replace(facts, evidence=())
    monkeypatch.setattr(prepare.PythonStaticAnalyzer, "analyze", lambda self, root: broken)
    with pytest.raises((ValueError, KeyError, prepare.SkillForgeError)):
        SkillForgePreparer(FakeModel(
            SemanticStructuralDecisionKind.ARCHITECTURE_RULE,
        )).prepare(PrepareRequest(repository, "task", CodeAgentTarget.CODEX))
    assert not (repository / ".codex").exists()
