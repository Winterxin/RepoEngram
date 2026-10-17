"""Executable workflows need action context; location and language alone are insufficient."""

from dataclasses import replace

import pytest
from test_skill_registry import compilation, skill

from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.routing import (
    SkillRoutingRequest,
    routing_trace,
    workflow_intent_tokens,
)
from skillforge.domain.skills import (
    SkillCompletionCondition,
    SkillConstraint,
    SkillConstraintEnforcement,
    SkillProcedure,
    SkillStep,
    SkillStepKind,
)
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.verify import VerifyPlanBuilder


def workflow(key, task, action):
    item = skill(key, task=task, scope=ProjectScope(ProjectScopeKind.MODULE, ("pkg.renderer",)))
    step = SkillStep("render", action, SkillStepKind.REQUIRED, None, "Observed action", (key,))
    verify = SkillStep("check", "Run relevant checks", SkillStepKind.REQUIRED, None, "Verify", ())
    return replace(
        item,
        steps=(step,),
        completion_conditions=(SkillCompletionCondition("render", None, (key,)),),
        procedure=SkillProcedure((step,), (), (verify,)),
    )


def routed(items, task):
    registry = SkillRegistryBuilder().build(compilation(tuple(items)))
    route = SkillRouter().route(
        registry, SkillRoutingRequest("repo", task, current_module="pkg.renderer")
    )
    return route, routing_trace(registry, route)


def test_same_module_language_and_test_changes_do_not_select_unrelated_action():
    item = workflow(
        "rationale",
        "Render concise English artifact text",
        "Normalize rationale whitespace in _format_rationale; update renderer tests.",
    )
    route, trace = routed(
        (item,), "Restore English output after empty section fallback; add tests."
    )
    assert route.matches == ()
    assert VerifyPlanBuilder().build(route).obligations == ()
    assert trace["candidates"][0]["non_selection_reason"] == "WORKFLOW_INTENT_MISMATCH"


def test_action_detail_discovers_workflow_even_when_trigger_uses_different_words():
    item = workflow(
        "rationale",
        "Render concise English artifact text",
        "Normalize rationale whitespace in _format_rationale.",
    )
    # Location still allows discovery when the action supplies the matching topic.
    route, _ = routed((item,), "Handle whitespace in rationale correctly.")
    assert tuple(m.skill.skill_id for m in route.matches) == ("rationale",)
    assert len(VerifyPlanBuilder().build(route).obligations) == 1


def test_two_workflows_in_one_module_select_only_the_matching_action():
    rationale = workflow(
        "rationale",
        "Render concise English artifact text",
        "Normalize rationale whitespace in _format_rationale.",
    )
    fallback = workflow(
        "fallback",
        "Restore deterministic section fallback",
        "Recover empty markdown from the verified deterministic plan.",
    )
    route, _ = routed((rationale, fallback), "Recover empty markdown with deterministic fallback.")
    assert tuple(m.skill.skill_id for m in route.matches) == ("fallback",)
    assert tuple(e.evidence_id for e in route.evidence) == ("fallback",)


def test_exact_trigger_and_advisory_scope_discovery_remain_available():
    item = workflow("exact", "Render concise English artifact text", "Normalize rationale.")
    rule = skill(
        "advisory",
        task="Unrelated observation",
        scope=ProjectScope(ProjectScopeKind.MODULE, ("pkg.renderer",)),
    )
    route, _ = routed((item, rule), item.trigger.task_description)
    assert {m.skill.skill_id for m in route.matches} == {"exact", "advisory"}
    route, _ = routed((item, rule), "Restore deterministic fallback")
    assert {m.skill.skill_id for m in route.matches} == {"advisory"}


def test_action_intent_never_rescues_an_explicit_scope_conflict():
    item = workflow("rationale", "Render concise English artifact text", "Normalize rationale.")
    registry = SkillRegistryBuilder().build(compilation((item,)))
    route = SkillRouter().route(
        registry, SkillRoutingRequest("repo", "Normalize rationale", current_module="pkg.unrelated")
    )
    assert route.matches == ()
    assert (
        routing_trace(registry, route)["candidates"][0]["non_selection_reason"]
        == "EXPLICIT_SCOPE_MISMATCH"
    )


def test_distinct_helpers_do_not_match_just_by_fragments_of_their_names():
    item = workflow("fallback", "Prepare the fallback", "Use _build_fallback_response.")
    route, _ = routed((item,), "Improve _fallback_response_metrics.")
    assert route.matches == ()
    route, _ = routed((item,), "Improve _build_fallback_response.")
    assert len(route.matches) == 1


def test_embedded_project_policy_is_not_lost_when_host_workflow_has_other_intent():
    item = workflow("rationale", "Render concise artifact text", "Normalize rationale.")
    policy = SkillConstraint(
        "privacy",
        RuleType.ARCHITECTURE,
        "Redact customer identifiers",
        item.trigger.scope,
        ProjectKnowledgeStatus.CONFIRMED,
        SkillConstraintEnforcement.REQUIRED,
        (),
        ("rationale",),
    )
    item = replace(
        item,
        constraints=(policy,),
        verify_rule_ids=("privacy",),
        procedure=replace(item.procedure, constraints=(policy,)),
    )
    route, _ = routed((item,), "Restore deterministic fallback")
    assert len(route.matches) == 1
    assert any(
        o.target_ref.ref_id == "privacy" for o in VerifyPlanBuilder().build(route).obligations
    )


def test_translated_task_retains_discovery_when_lexical_applicability_is_unknown():
    item = workflow("rationale", "Render concise artifact text", "Normalize rationale.")
    route, _ = routed((item,), "修复英文图表说明的截断问题")
    assert len(route.matches) == 1


@pytest.mark.parametrize("package", ["atlas", "cedar"])
@pytest.mark.parametrize("context", [
    {"current_module": "{package}.reader"},
    {"current_modules": ("{package}.reader", "{package}.writer")},
    {"current_module": "{package} reader"},
])
def test_repository_workflow_needs_action_beyond_explicit_module_location(package, context):
    item = workflow(
        "release", f"Synchronize {package} release metadata",
        f"Synchronize {package} version metadata across release files.",
    )
    item = replace(item, trigger=replace(
        item.trigger, scope=ProjectScope(ProjectScopeKind.REPOSITORY, ())
    ))
    context = {key: tuple(value.format(package=package) for value in values)
               if isinstance(values, tuple) else values.format(package=package)
               for key, values in context.items()}
    registry = SkillRegistryBuilder().build(compilation((item,)))
    route = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", f"Improve {package} escaping behavior", **context
    ))
    assert route.matches == ()
    assert routing_trace(registry, route)["candidates"][0]["non_selection_reason"] == (
        "WORKFLOW_INTENT_MISMATCH"
    )
    related = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", f"Synchronize {package} release version metadata", **context
    ))
    assert len(related.matches) == 1
    exact = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", item.trigger.task_description, **context
    ))
    assert len(exact.matches) == 1


def test_repository_location_filter_does_not_guess_missing_module_context():
    item = workflow("release", "Synchronize atlas release metadata", "Update atlas version.")
    item = replace(item, trigger=replace(
        item.trigger, scope=ProjectScope(ProjectScopeKind.REPOSITORY, ())
    ))
    registry = SkillRegistryBuilder().build(compilation((item,)))
    route = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", "Improve atlas escaping behavior"
    ))
    assert len(route.matches) == 1


@pytest.mark.parametrize("scope", [
    ProjectScope(ProjectScopeKind.MODULE, ("pkg.renderer",)),
    ProjectScope(ProjectScopeKind.FILE, ("src/renderer.py",)),
    ProjectScope(ProjectScopeKind.DIRECTORY, ("src/renderer",)),
])
def test_guard_words_and_plural_location_do_not_select_an_unrelated_workflow(scope):
    item = workflow(
        "tax", "Guard unsupported renderers when tax mapping is absent",
        "Guard fallback renderers against unsupported tax claims without a mapping.",
    )
    item = replace(item, trigger=replace(item.trigger, scope=scope))
    context = {
        ProjectScopeKind.MODULE: {"current_module": "pkg.renderer"},
        ProjectScopeKind.FILE: {"current_file": "src/renderer.py"},
        ProjectScopeKind.DIRECTORY: {"current_directory": "src/renderer"},
    }[scope.kind]
    registry = SkillRegistryBuilder().build(compilation((item,)))
    route = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", "Filter unsupported numeric figures in public renderers.", **context,
    ))
    assert route.matches == ()
    assert VerifyPlanBuilder().build(route).obligations == ()
    assert routing_trace(registry, route)["candidates"][0][
        "non_selection_reason"
    ] == "WORKFLOW_INTENT_MISMATCH"


def test_specific_guard_subject_still_matches_after_generic_words_are_removed():
    item = workflow(
        "tax", "Guard unsupported renderers when tax mapping is absent",
        "Guard fallback renderers against unsupported tax claims without a mapping.",
    )
    route, _ = routed((item,), "Handle missing tax mapping in the renderer.")
    assert len(route.matches) == 1


def conditional_workflow(key, task, action, condition):
    item = workflow(key, task, action)
    step = replace(item.steps[0], kind=SkillStepKind.CONDITIONAL, condition=condition)
    return replace(
        item, steps=(step,),
        completion_conditions=(SkillCompletionCondition(step.step_id, condition, (key,)),),
        procedure=replace(item.procedure, steps=(step,)),
    )


@pytest.mark.parametrize("condition,unrelated", [
    ("when transport is disconnected", "Improve _inspect_transport transport diagnostics"),
    ("when authentication is rejected", "Improve _inspect_auth authentication logging"),
])
def test_condition_only_overlap_does_not_rescue_an_explicitly_different_helper(
    condition, unrelated,
):
    item = conditional_workflow(
        "cache", "Invalidate stale cache entries", "Remove cached records via _clear_cache",
        condition,
    )
    route, trace = routed((item,), unrelated)
    assert route.matches == route.evidence == ()
    assert trace["candidates"][0]["non_selection_reason"] == "WORKFLOW_INTENT_MISMATCH"
    # Full conditions stay in canonical knowledge and audit visibility.
    assert item.procedure.steps[0].condition == condition
    assert set(workflow_intent_tokens(item, include_conditions=False)) < set(
        trace["candidates"][0]["workflow_intent_tokens"]
    )


def test_conditional_workflow_retains_actual_action_exact_and_translated_discovery():
    item = conditional_workflow(
        "cache", "Invalidate stale cache entries", "Remove cached records via _clear_cache",
        "when transport is disconnected",
    )
    for task in ("Remove cached records", "Fix _clear_cache", item.trigger.task_description,
                 "清理失效缓存", "Improve transport diagnostics"):
        route, _ = routed((item,), task)
        assert len(route.matches) == 1
        assert route.matches[0].skill == item


def test_conditional_workflow_with_policy_keeps_scope_based_policy_discovery():
    item = conditional_workflow(
        "cache", "Invalidate stale cache entries", "Remove cached records",
        "when transport is disconnected",
    )
    policy = SkillConstraint(
        "privacy", RuleType.TESTING, "Retain observed privacy boundaries",
        item.trigger.scope, ProjectKnowledgeStatus.CONFIRMED,
        SkillConstraintEnforcement.REQUIRED, (), ("cache",), None, None,
    )
    item = replace(
        item, constraints=(policy,), verify_rule_ids=("privacy",),
        procedure=replace(item.procedure, constraints=(policy,)),
    )
    route, _ = routed((item,), "Improve transport diagnostics")
    assert len(route.matches) == 1
    assert any(o.target_ref.ref_id == "privacy"
               for o in VerifyPlanBuilder().build(route).obligations)


def test_conditional_subject_is_available_for_workflow_discovery():
    item = workflow("currency", "Validate incoming input", "Validate the input.")
    condition = "When the currency code is absent"
    step = replace(item.steps[0], kind=SkillStepKind.CONDITIONAL, condition=condition)
    completion = replace(item.completion_conditions[0], condition=condition)
    item = replace(item, steps=(step,), completion_conditions=(completion,),
                   procedure=replace(item.procedure, steps=(step,)))
    route, _ = routed((item,), "Handle currency conversion correctly.")
    assert len(route.matches) == 1
    assert VerifyPlanBuilder().build(route).obligations[0].condition == condition


def test_plural_location_with_y_does_not_supply_action_intent():
    item = workflow("entry", "Guard entries", "Guard entries against tax claims.")
    scope = ProjectScope(ProjectScopeKind.MODULE, ("pkg.entry",))
    item = replace(item, trigger=replace(item.trigger, scope=scope))
    registry = SkillRegistryBuilder().build(compilation((item,)))
    route = SkillRouter().route(registry, SkillRoutingRequest(
        "repo", "Filter unsupported numeric entries.", current_module="pkg.entry",
    ))
    assert route.matches == ()
