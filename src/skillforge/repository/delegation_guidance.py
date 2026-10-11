"""Conditional architecture decision procedure for repeated delegation facts."""

from skillforge.domain.skills import SkillStep, SkillStepKind
from skillforge.domain.structural_rules import StructuralRuleCandidate


def delegation_decision_steps(candidate: StructuralRuleCandidate) -> tuple[SkillStep, ...]:
    condition = ("The new operation belongs to the same observed family and no contrary "
                 "repository evidence, stronger rule or exception applies")
    descriptions = (
        "Identify analogous operations in the same family: "
        + ", ".join(candidate.operation_symbols),
        "Inspect their observed complete delegation topology: "
        + " -> ".join(candidate.delegation_topology),
        "Determine the actual owning implementation layer from current symbols, "
        "resolved calls and behavior; inspect contrary examples and exceptions.",
        "If the new operation belongs to this family and no contrary repository "
        "evidence exists, preserve the observed delegation structure; otherwise "
        "record the exception and follow the current evidence.",
        "Implement the requested behavior at the actual owning layer.",
        "Connect intermediate and public layers as required by the chosen structure.",
        "Verify the end-to-end public operation actually works and relevant "
        "regressions remain passing.",
    )
    return tuple(SkillStep(
        f"delegation-{index}", description, SkillStepKind.CONDITIONAL, condition,
        "Conditional architecture convention from current source Evidence",
        candidate.rule.evidence_ids,
    ) for index, description in enumerate(descriptions, 1))
