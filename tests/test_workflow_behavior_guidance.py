"""Synthetic behavior survives existing semantic gates, compilation and rendering."""

import json
from dataclasses import asdict, replace

import pytest
from test_semantic_workflows import FakeModel, _fixture, _valid_response

from skillforge.domain.project_dna import ProjectKnowledgeStatus
from skillforge.domain.semantic_workflow import (
    SemanticStepClassification,
    SemanticWorkflowException,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.skills import SkillCompilationResult
from skillforge.integration.agent_skills import _render_skill
from skillforge.productization.evolution_artifacts import _restore
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    _validate_text_claims,
)
from skillforge.repository.skill_compiler import SkillCompiler

ACTION = "For an empty batch, retain the recorded total and return an unavailable result"
CONDITION = "The batch has no rows"
EXCEPTION = "Legacy compatibility batches retain their original schema"
REF = "c1:log"


def behavior_fixture():
    history, facts, candidates = _fixture()

    def grounded(items):
        return tuple(replace(item, description=f"{ACTION}. {EXCEPTION}.")
                     if item.evidence_id == REF else item for item in items)

    return (replace(history, evidence=grounded(history.evidence)), facts,
            replace(candidates, evidence=grounded(candidates.evidence)))


def behavior_response(request):
    valid = _valid_response(request)
    return replace(
        valid,
        ordered_steps=(SemanticWorkflowStep("empty-batch", ACTION, (REF,)),),
        required_steps=(), optional_steps=(),
        conditional_steps=(SemanticStepClassification(
            "empty-batch", "Observed empty input boundary", (REF,), CONDITION,
        ),),
        scope=replace(valid.scope, kind=SemanticWorkflowScopeKind.MODULE,
                      selectors=request.current_modules),
        exceptions=(SemanticWorkflowException(EXCEPTION, (REF,)),),
    )


def test_grounded_boundary_survives_semantic_dna_skill_artifact_and_render():
    history, facts, candidates = behavior_fixture()
    accepted = SemanticWorkflowAbstracter().abstract(
        history, facts, candidates, FakeModel(behavior_response),
    )
    assert accepted.workflows
    source = ProjectDNABuilder().build(history.repository_id, workflows=accepted)
    compiled = SkillCompiler().compile(source)
    restored = _restore(SkillCompilationResult, json.loads(json.dumps(asdict(compiled))))
    assert restored == compiled
    assert len(compiled.skills) == len(accepted.workflows)
    for skill in restored.skills:
        assert skill.source_status == ProjectKnowledgeStatus.CANDIDATE
        assert skill.steps[0].instruction == ACTION
        assert skill.steps[0].condition == CONDITION
        assert skill.exceptions[0].description == EXCEPTION
        assert REF in skill.steps[0].evidence_ids
        assert skill.exceptions[0].evidence_ids == (REF,)
        assert [item.step_id for item in skill.procedure.verify] == [
            "verify-behavior", "verify-placement",
        ]
        text = _render_skill(skill, "synthetic repository")
        assert all(value in text for value in (ACTION, CONDITION, EXCEPTION, REF))


@pytest.mark.parametrize("field", ["conditional_steps", "exceptions"])
def test_specific_behavior_does_not_authorize_foreign_citations(field):
    history, facts, candidates = behavior_fixture()

    def foreign(request):
        valid = behavior_response(request)
        items = getattr(valid, field)
        return replace(valid, **{field: (replace(items[0], evidence_ids=("foreign",)),)})

    with pytest.raises(ValueError):
        SemanticWorkflowAbstracter().abstract(history, facts, candidates, FakeModel(foreign))


@pytest.mark.parametrize("suffix", ["", "."])
def test_known_python_module_is_not_misclassified_as_foreign_c_file(suffix):
    _validate_text_claims((f"Update pkg.c{suffix}",), set(), {"pkg.c"}, set())


def test_module_disambiguation_preserves_unknown_path_gate():
    with pytest.raises(ValueError, match="unknown path"):
        _validate_text_claims(("Update other.c",), set(), {"pkg.c"}, set())
