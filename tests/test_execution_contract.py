"""Execution obligations stay visible without changing source knowledge or authority."""

import pytest
from test_executable_skill_compiler import tendency  # noqa: F401
from test_skill_compiler import dna

from skillforge.integration.agent_skills import _render_guidance, _render_skill
from skillforge.integration.autonomous import CONTRACT
from skillforge.repository.skill_compiler import SkillCompiler


@pytest.fixture
def rendered():
    skill, = SkillCompiler().compile(dna()).skills
    return skill, _render_skill(skill, "synthetic")


def test_source_actions_order_conditions_and_authority_survive(rendered):
    skill, text = rendered
    offsets = []
    for step in skill.steps:
        assert text.count(step.instruction) == 1
        assert f"[{step.kind.value.upper()}] {step.instruction}" in text
        assert step.reason in text
        assert ", ".join(step.evidence_ids) in text
        offsets.append(text.index(f"Step ID: {step.step_id}"))
        if step.condition:
            assert f"Condition: {step.condition}" in text
    assert offsets == sorted(offsets)
    assert "Source status: candidate" in text
    assert "conditional observations" in text
    assert "never unconditional MUST rules" in text


def test_generic_transport_protocol_is_not_added_to_task_context(rendered):
    skill, text = rendered
    for phrase in ("Shared execution protocol", "COMPLETION GATE", "DONE WHEN",
                   "SKIP ONLY IF", "NOT_APPLICABLE_WITH_EVIDENCE"):
        assert phrase not in text
    # The typed source procedure remains intact for formal consumers.
    assert skill.procedure.verify
    for item in skill.procedure.verify:
        if not item.evidence_ids:
            assert item.instruction not in text


def test_source_exceptions_and_actual_task_trigger_remain(rendered):
    skill, text = rendered
    assert skill.trigger.task_description in text
    assert all(item.description in text for item in skill.exceptions)
    assert "contrary current evidence" in text
    assert "permission to expand the task" in text
    assert "references/evidence.md" in text


def test_rule_guidance_stays_conditional_without_workflow_gate(tendency):  # noqa: F811
    source, semantic = tendency
    skill, = SkillCompiler().compile(source, semantic_tendencies=semantic).skills
    text = _render_guidance(skill, "synthetic")
    assert "# Execution contract" not in text
    assert "# COMPLETION GATE" not in text
    assert "never unconditional MUST rules" in text
    assert "contrary current evidence" in text
    assert skill.procedure.steps[2].condition in text


def test_autonomous_discovery_declares_contract_without_changing_invocation():
    assert "call get_related_workflow" in CONTRACT
    assert "Call get_skill" in CONTRACT
    assert "call verify_patch" in CONTRACT
    assert "execution contracts" in CONTRACT
    assert "final completion gate" in CONTRACT
