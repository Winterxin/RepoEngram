"""Task guidance rendering, separate from compilation and provenance serialization."""

from __future__ import annotations

import json

from skillforge.domain.project_dna import ProjectScope
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import ProjectSkill, RuleGuidanceSkill


def _scope(scope: ProjectScope) -> str:
    return f"{scope.kind.value}: {', '.join(scope.selectors) or '(whole repository)'}"


def _exceptions(items: tuple) -> list[str]:
    return [f"- Exception: {item.description} (Evidence: {', '.join(item.evidence_ids)})"
            for item in items]


def render_executable(
    skill: ProjectSkill | RuleGuidanceSkill, name: str, description: str,
) -> str:
    """Keep source actions and authority; omit transport-generated workflow bureaucracy."""
    procedure = skill.procedure
    if procedure is None:
        raise ValueError("executable rendering requires a compiled procedure")
    workflow = isinstance(skill, ProjectSkill)
    lines = ["---", f"name: {json.dumps(name, ensure_ascii=False)}",
             f"description: {json.dumps(description, ensure_ascii=False)}", "---", "",
             f"# {skill.description}", "", "## When this applies", "",
             skill.trigger.task_description, f"Scope: {_scope(skill.trigger.scope)}.",
             f"Source status: {skill.source_status.value}.",
             "Apply guidance within this task scope and its exceptions; historical candidates "
             "are conditional observations, never unconditional MUST rules or permission "
             "to expand the task. Stronger instructions and contrary current evidence "
             "take precedence; candidate knowledge does not become a confirmed project policy.",
             "", "## Decision procedure", ""]
    for index, step in enumerate(procedure.steps, 1):
        lines += [f"{index}. [{step.kind.value.upper()}] {step.instruction}",
                  f"   - Step ID: {step.step_id}"]
        if step.condition:
            lines.append(f"   - Condition: {step.condition}")
        lines += [f"   - Why: {step.reason}",
                  f"   - Evidence: {', '.join(step.evidence_ids) or 'generic reasoning'}"]
    if procedure.constraints:
        lines += ["", "## Project constraints", ""]
    for item in procedure.constraints:
        lines += [f"- [{item.enforcement.value.upper()}] {item.statement}",
                  f"  - Scope: {_scope(item.scope)}; source status: {item.source_status.value}",
                  f"  - Evidence: {', '.join(item.evidence_ids)}", *_exceptions(item.exceptions)]
    # Generic compiler checks remain in the typed procedure, not in every task's context.
    verification = tuple(item for item in procedure.verify if item.evidence_ids or not workflow)
    if verification:
        lines += ["", "## Verify", ""]
    for item in verification:
        lines.append(f"- [{item.kind.value.upper()}] {item.instruction}")
        if item.condition:
            lines.append(f"  - Condition: {item.condition}")
        lines.append(f"  - Evidence: {', '.join(item.evidence_ids) or 'generic reasoning'}")
    if workflow:
        boundaries = _exceptions(skill.exceptions)
        for risk in skill.risks:
            boundaries += [f"- Risk ({risk.source_status.value}): {risk.name}: "
                           f"{risk.description} Avoid: {risk.avoid_behavior} "
                           f"(scope: {_scope(risk.scope)}; Evidence: "
                           f"{', '.join(risk.evidence_ids)})", *_exceptions(risk.exceptions)]
    else:
        boundaries = [f"- Source authority: {skill.constraint.enforcement.value}.",
                      *_exceptions(skill.constraint.exceptions)]
    if boundaries:
        lines += ["", "## Exceptions and risks", "", *boundaries]
    lines += ["", "## References", "",
              "Read [evidence and source audit](references/evidence.md) when checking "
              "a claim's provenance or applicability.", ""]
    return "\n".join(lines)


def render_instruction_audit(skill: ProjectSkill | RuleGuidanceSkill) -> str:
    """Keep machine provenance available without repeating it in primary instructions."""
    audit = {"source_status": skill.source_status.value,
             "support": to_json_value(skill.source_support),
             "confidence": to_json_value(skill.source_confidence),
             "evidence_ids": list(skill.evidence_ids),
             "excluded_rule_ids": list(skill.procedure.excluded_rule_ids)}
    if isinstance(skill, RuleGuidanceSkill):
        audit.update(source_rule_id=skill.source_rule_id, semantic_method=skill.semantic_method,
                     semantic_reason=skill.semantic_reason)
    return "\n## Instruction source audit\n\n```json\n" + json.dumps(
        audit, ensure_ascii=False, sort_keys=True, indent=2,
    ) + "\n```\n"
