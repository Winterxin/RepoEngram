"""Executable conditional obligations from accepted public testing linkage."""

from dataclasses import replace

from skillforge.domain.skills import SkillStep, SkillStepKind
from skillforge.domain.structural_rules import DELEGATION_METHOD
from skillforge.repository.delegation_guidance import delegation_decision_steps


def companion_steps(current, semantic, rules):
    """Compose accepted executable rules without changing their source semantics."""
    steps = ()
    for accepted in semantic.accepted:
        other = accepted.candidate
        if (other.rule.rule_id == current.rule.rule_id
                or rules.get(other.rule.rule_id) != other.rule
                or other.rule.scope != current.rule.scope
                or not current.operation_family
                or other.operation_family != current.operation_family):
            continue
        extra = (testing_decision_steps(other) if other.test_linkages else
                 delegation_decision_steps(other) if other.method == DELEGATION_METHOD else ())
        steps += tuple(replace(step, step_id=f"{other.rule.rule_id}-{step.step_id}")
                       for step in extra)
    return steps


def testing_decision_steps(candidate):
    condition = (
        "The task adds an operation in the observed public operation family, after checking "
        "current production and test evidence and explicit exceptions"
    )
    descriptions = (
        "Determine whether the task belongs to the observed public operation family "
        + str(candidate.operation_family) + "; compare "
        + ", ".join(candidate.operation_symbols)
        + ". Inspect contrary examples and explicit repository exceptions before editing.",
        "Inspect the linked formal project testing pattern: "
        + "; ".join(f"{r.test_symbol} in {r.path} invokes {r.operation_symbol} "
                    f"at line {r.call_line} and asserts behavior at line {r.assertion_line}"
                    for r in candidate.test_linkages),
        "CONDITIONAL REQUIRED — ACTION: Add or update the corresponding project-level "
        "behavior test using the observed public testing convention. Persist it in the "
        "repository's formal test suite, exercise the new public entry and assert externally "
        "visible behavior. Follow the observed test location and setup pattern. "
        "Preserve the observed behavior case types for the new operation: "
        + ", ".join(sorted({r.assertion_kind for r in candidate.test_linkages}))
        + " (result assertions cover successful visible behavior; raises assertions "
        "cover failure behavior where observed and applicable).",
        "CONDITIONAL REQUIRED — DONE WHEN: A persistent repository test exists, exercises "
        "the new public behavior, and the relevant test suite is actually run and passes. "
        "Record the test symbol, public call, behavior assertions, command and result. "
        "A temporary shell behavior probe does NOT satisfy this obligation. Existing regression "
        "tests alone do NOT satisfy the new feature behavior test obligation.",
        "SKIP ONLY IF: The task is proven not to belong to the family, current repository "
        "evidence shows an explicit exception, or the work is blocked with explicit evidence. "
        "Record that evidence; otherwise the testing action and DONE WHEN are required.",
    )
    return tuple(SkillStep(
        f"testing-{index}", description, SkillStepKind.CONDITIONAL, condition,
        "Conditional required behavior testing from accepted Structural Semantic Evidence",
        candidate.rule.evidence_ids,
    ) for index, description in enumerate(descriptions, 1))
