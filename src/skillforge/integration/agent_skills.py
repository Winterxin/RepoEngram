"""Faithful, deterministic rendering of the Router's selected Skills."""

from __future__ import annotations

import json

from skillforge.domain.agent_integration import (
    SCHEMA,
    AgentSkillExportPlan,
    AgentSkillFile,
    AgentSkillPackage,
    CodeAgentTarget,
    digest,
    plan_fingerprint,
    skill_name,
)
from skillforge.domain.evidence import Evidence
from skillforge.domain.routing import SkillRouteResult
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import ProjectSkill, RuleGuidanceSkill


def _scalar(value: str) -> str:
    # A JSON quoted scalar is valid YAML; escaping also prevents frontmatter injection.
    return json.dumps(value, ensure_ascii=False)


def _metric(value: object) -> str:
    if value is None:
        return "not provided"
    return json.dumps(to_json_value(value), ensure_ascii=False, sort_keys=True)


def _scope(scope: object) -> str:
    selectors = ", ".join(scope.selectors) if scope.selectors else "(whole repository)"
    return f"{scope.kind.value}: {selectors}"


def _exception(item: object, indent: str = "") -> str:
    evidence = ", ".join(item.evidence_ids) or "none"
    return f"{indent}- Exception: {item.description} (Evidence: {evidence})"


def _exceptions(items: tuple[object, ...]) -> list[str]:
    return [f"- {item.description} (Evidence: {', '.join(item.evidence_ids) or 'none'})"
            for item in items]


def _render_skill(skill: ProjectSkill, name: str) -> str:
    if skill.procedure is not None:
        return _render_executable(skill, name)
    scope = _scope(skill.trigger.scope)
    description = f"{skill.trigger.task_description} (scope: {scope})"
    if len(description) > 1024:
        raise ValueError(
            "portable Agent Skill description exceeds specification limit (1024 chars)"
        )
    if not description:
        raise ValueError("portable Agent Skill description must be nonempty")
    lines = ["---", f"name: {_scalar(name)}", f"description: {_scalar(description)}",
             "---", "", f"# {skill.description}", "", "## Applicability", "",
             f"- Task: {skill.trigger.task_description}", f"- Scope: {scope}",
             f"- Source status: {skill.source_status.value}",
             f"- Source Support: {_metric(skill.source_support)}",
             f"- Source Confidence: {_metric(skill.source_confidence)}", "",
             "## Workflow", ""]
    for index, step in enumerate(skill.steps, 1):
        lines += [f"{index}. [{step.kind.value.upper()}] {step.instruction}",
                  f"   - Step ID: {step.step_id}",
                  f"   - Reason: {step.reason}"]
        if step.condition is not None:
            lines.append(f"   - Condition: {step.condition}")
        lines.append(f"   - Evidence: {', '.join(step.evidence_ids) or 'none'}")
    for enforcement, heading in (("required", "Required constraints"),
                                 ("advisory", "Advisory constraints")):
        lines += ["", f"## {heading}", ""]
        for rule in skill.constraints:
            if rule.enforcement.value != enforcement:
                continue
            lines += [f"- [{rule.enforcement.value.upper()}] {rule.rule_id}: {rule.statement}",
                      f"  - Rule type: {rule.rule_type.value}",
                      f"  - Scope: {_scope(rule.scope)}",
                      f"  - Source status: {rule.source_status.value}",
                      f"  - Source Support: {_metric(rule.source_support)}",
                      f"  - Source Confidence: {_metric(rule.source_confidence)}",
                      f"  - Evidence: {', '.join(rule.evidence_ids) or 'none'}"]
            lines += [_exception(item, "  ") for item in rule.exceptions]
    lines += ["", "## Risks / anti-patterns", ""]
    for risk in skill.risks:
        lines += [f"- {risk.anti_pattern_id}: {risk.name}",
                  f"  - Description: {risk.description}",
                  f"  - Avoid behavior: {risk.avoid_behavior}",
                  f"  - Scope: {_scope(risk.scope)}",
                  f"  - Source status: {risk.source_status.value}",
                  f"  - Confidence: {_metric(risk.confidence)}",
                  f"  - Evidence: {', '.join(risk.evidence_ids) or 'none'}"]
        lines += [_exception(item, "  ") for item in risk.exceptions]
    lines += ["", "## Workflow exceptions", "", *_exceptions(skill.exceptions),
              "", "## Completion conditions", ""]
    for item in skill.completion_conditions:
        condition = item.condition if item.condition is not None else "none"
        evidence = ", ".join(item.evidence_ids) or "none"
        lines.append(f"- Step ID: {item.step_id}; condition: {condition}; Evidence: {evidence}")
    lines += ["", "## Verification references", "",
              "These are Project DNA Rule references, not executable verification commands."]
    for key in skill.verify_rule_ids:
        rule = next(item for item in skill.constraints if item.rule_id == key)
        lines.append(f"- {key}: {rule.statement}")
    lines += ["", "## Evidence", "", "See [references/evidence.md](references/evidence.md).",
              f"Referenced Evidence IDs: {', '.join(skill.evidence_ids) or 'none'}", ""]
    return "\n".join(lines)


def _render_guidance(skill: RuleGuidanceSkill, name: str) -> str:
    if skill.procedure is not None:
        return _render_executable(skill, name)
    scope = _scope(skill.source_scope)
    description = f"{skill.trigger.task_description} (scope: {scope})"
    if len(description) > 1024:
        raise ValueError(
            "portable Agent Skill description exceeds specification limit (1024 chars)"
        )
    lines = [
        "---", f"name: {_scalar(name)}", f"description: {_scalar(description)}",
        "---", "", f"# {skill.description}", "",
        "## When this applies", "", f"- Scope: {scope}",
        f"- Source status: {skill.source_status.value}", "",
        "## Project guidance", "",
        f"[ADVISORY] {skill.constraint.statement}", "",
        "This legacy observation has no compiled procedure; recompile before execution.",
    ]
    if skill.constraint.exceptions:
        lines += ["", "## Exceptions", "", *_exceptions(skill.constraint.exceptions)]
    lines += ["", "## Evidence", "", "See [references/evidence.md](references/evidence.md).",
              f"Referenced Evidence IDs: {', '.join(skill.evidence_ids)}", ""]
    return "\n".join(lines)


def _execution_contract() -> list[str]:
    return [
        "# Execution contract v1", "",
        "MATCH SKILL -> RESOLVE APPLICABILITY -> BUILD APPLICABLE OBLIGATIONS -> EXECUTE "
        "-> CHECK DONE CONDITIONS -> FINAL COMPLETION REVIEW -> ONLY THEN FINISH.", "",
        "First determine whether this workflow and each step apply to the current user task. "
        "Record the applicability decisions using current-repository evidence before execution. "
        "Routing alone does not make every step applicable; unrelated knowledge must not expand "
        "user intent or require an unrelated new feature.",
        "Build an execution plan of applicable REQUIRED steps and CONDITIONAL steps whose "
        "conditions hold. Each is a completion obligation until DONE, "
        "NOT_APPLICABLE_WITH_EVIDENCE, or BLOCKED. OPTIONAL steps do not block completion.",
        "Resolve applicability for each returned Skill; prefer the most specific applicable "
        "project knowledge. Equivalent obligations from multiple Skills are one obligation, "
        "not duplicated work; duplicated copies do not weaken a REQUIRED obligation.", "",
        "Resolve source conditions and exceptions before applying an action. A limited "
        "rule does not authorize changing behavior outside its supported boundary; preserve "
        "independently valid behavior. Check the final result through the actual integration "
        "path, not only a helper or an intermediate state. For requested behavior-preservation "
        "changes, exercise already-valid inputs at every user-named entry point. Keep original "
        "regression expectations unless the user authorizes changing that behavior; test new "
        "requirements separately. For distinct generation paths, use independent test outputs "
        "and confirm the intended path ran; copying fallback output does not test "
        "fresh generation.",
        "Shared step completion criteria: fulfill the stated ACTION for the current task; "
        "the change exists and its intended integration point uses it where applicable. "
        "Address declared verification obligations and the Testing completion semantics below.",
        "Shared skip criteria: concrete current-repository evidence shows non-applicability; "
        "record NOT_APPLICABLE_WITH_EVIDENCE. Passing tests, small patches, scope pointing "
        "elsewhere, or preferring less work alone are not sufficient reasons.", "",
        "Matched scope identifies applicability and context. It does not restrict the files "
        "that may be changed. MATCHED SCOPE / CURRENT SCOPE, including current_file and "
        "current_module, are not an edit boundary or file permission boundary. If an applicable "
        "REQUIRED workflow step requires changes outside the matched scope, perform those "
        "changes. Only an explicit user restriction establishes an edit boundary; if that "
        "prevents an obligation, record BLOCKED.", "",
        "Testing completion: when an evidence-backed action requires adding or updating tests, "
        "DONE requires added or updated tests covering the current new behavior, with their "
        "paths and coverage recorded. Only running existing tests, only calling Verify, or "
        "only reporting that tests were not run does not satisfy that action. Apply this "
        "meaning only when the source action requires test changes; do not infer new project "
        "testing obligations from a generic verification step.", "",
    ]


def _completion_gate() -> list[str]:
    return ["", "# COMPLETION GATE", "", "Before considering the task complete:",
            "1. Review ALL applicable REQUIRED steps, not only steps already executed, "
            "including declared verification obligations. Check the final diff against them.",
            "2. Record every REQUIRED step ID and status: DONE, "
            "NOT_APPLICABLE_WITH_EVIDENCE, or BLOCKED, with completion evidence or the "
            "concrete current-repository reason for non-applicability.",
            "3. Resolve every CONDITIONAL step: APPLIES or NOT_APPLICABLE, with reasons. "
            "For APPLIES, resolve its completion obligation with the same statuses.",
            "4. OPTIONAL steps do not block completion. A REQUIRED step may not disappear "
            "silently; reporting that a step was not completed does not make the task complete.",
            "5. If any applicable REQUIRED or applicable CONDITIONAL obligation is unresolved "
            "or BLOCKED, task status remains INCOMPLETE / BLOCKED. Report it honestly and "
            "do not finish normally as a completed task.",
            "6. Only after all applicable obligations are resolved as DONE or "
            "NOT_APPLICABLE_WITH_EVIDENCE may the Agent finish normally."]


def _render_executable(skill: ProjectSkill | RuleGuidanceSkill, name: str) -> str:
    procedure = skill.procedure
    scope = _scope(skill.trigger.scope)
    description = f"{skill.trigger.task_description} (scope: {scope})"
    if len(description) > 1024:
        raise ValueError(
            "portable Agent Skill description exceeds specification limit (1024 chars)"
        )
    lines = ["---", f"name: {_scalar(name)}", f"description: {_scalar(description)}", "---",
             "", "# Purpose", "", f"Apply an Evidence-backed method for: {skill.description}",
             "", "# When this applies", "", f"- Scope: {scope}",
             "- Use when the requested change falls within this scope "
             "and matches the source intent.",
             "- Inspect current ownership and source exceptions before choosing a placement.",
             ""]
    workflow = isinstance(skill, ProjectSkill)
    if workflow:
        # Keep applicability before actions, but put project-specific actions
        # ahead of the shared contract in bounded first reads.
        lines += [_execution_contract()[4], ""]
    lines += ["# Decision procedure / Steps", ""]
    for index, step in enumerate(procedure.steps, 1):
        heading = step.step_id if workflow else step.instruction
        lines += [f"{index}. [{step.kind.value.upper()}] {heading}"]
        if workflow:
            lines += [f"   - Step ID: {step.step_id}",
                      f"   - TYPE: {step.kind.value.upper()}",
                      f"   - ACTION: {step.instruction}",
                      "   - DONE WHEN: This ACTION meets the shared step completion criteria."]
            if step.kind.value == "optional":
                skip = "Optional; may be omitted and does not become a completion blocker."
            else:
                skip = "The shared skip criteria establish non-applicability for this ACTION."
            lines += [f"   - SKIP ONLY IF: {skip}", f"   - WHY: {step.reason}"]
        if step.condition:
            lines.append(f"   - Condition: {step.condition}")
        if not workflow:
            lines.append(f"   - Reason: {step.reason}")
        lines.append(f"   - Evidence: {', '.join(step.evidence_ids) or 'generic reasoning step'}")
    lines += ["", "# Project constraints", ""]
    if not procedure.constraints:
        lines.append("No additional Evidence-backed project constraints were supplied.")
    for item in procedure.constraints:
        lines += [f"- [{item.enforcement.value.upper()}] {item.rule_id}: {item.statement}",
                  f"  - Scope: {_scope(item.scope)}; source status: {item.source_status.value}",
                  f"  - Support: {_metric(item.source_support)}; "
                  f"Confidence: {_metric(item.source_confidence)}",
                  f"  - Evidence: {', '.join(item.evidence_ids)}"]
        lines += [_exception(exc, "  ") for exc in item.exceptions]
    if workflow:
        contract = _execution_contract()
        lines += ["", *contract[:4], *contract[5:]]
    lines += ["", "# Verify", ""]
    for index, item in enumerate(procedure.verify, 1):
        instruction = item.instruction
        if workflow and item.step_id == "verify-placement":
            instruction = instruction.replace("each executed step", "ALL applicable REQUIRED steps")
        lines.append(f"{index}. {instruction}")
        if item.condition:
            lines.append(f"   - Condition: {item.condition}")
        refs = ', '.join(item.evidence_ids) or 'generic verification step'
        lines.append(f"   - Evidence: {refs}")
    lines += ["", "# Evidence", "", "See [references/evidence.md](references/evidence.md).",
              f"Referenced Evidence IDs: {', '.join(skill.evidence_ids)}",
              "", "# Confidence / boundaries", "",
              f"- Source status: {skill.source_status.value}",
              f"- Source Support: {_metric(skill.source_support)}",
              f"- Source Confidence: {_metric(skill.source_confidence)}",
              "- Historical tendencies are conditional preferences, "
              "never unconditional MUST rules.",
              "- Candidate or low-confidence knowledge requires checking current applicability; "
              "it does not become a confirmed project policy.",
              "- Existing ownership, stronger constraints, contrary current evidence, "
              "and documented "
              "exceptions may justify a different choice. Record that reasoning.",
              "- No future filename, command, or unobserved project convention is prescribed."]
    if isinstance(skill, RuleGuidanceSkill):
        lines += [f"- Source authority: {skill.constraint.enforcement.value}; "
                  "execute the conditional decision procedure without promoting the observation.",
                  f"- Source rule: {skill.source_rule_id}",
                  f"- Semantic method: {skill.semantic_method}",
                  f"- Semantic rationale: {skill.semantic_reason}"]
        lines += [_exception(item) for item in skill.constraint.exceptions]
    else:
        lines += _exceptions(skill.exceptions)
        for item in skill.risks:
            lines += [f"- Risk ({item.source_status.value}): {item.name}: {item.avoid_behavior} "
                      f"(Evidence: {', '.join(item.evidence_ids)})", *_exceptions(item.exceptions)]
    if procedure.excluded_rule_ids:
        lines.append("- Conflicted knowledge excluded from instructions: "
                     + ", ".join(procedure.excluded_rule_ids))
    if workflow:
        lines += _completion_gate()
    return "\n".join([*lines, ""])


def _render_evidence(evidence: tuple[Evidence, ...]) -> str:
    lines = ["# Evidence references", ""]
    for item in evidence:
        lines += [f"## {item.evidence_id}", "", f"- Kind: {item.kind.value}",
                  f"- Path: {item.location.path}"]
        if item.location.commit is not None:
            lines.append(f"- Commit: {item.location.commit}")
        if item.location.line is not None:
            lines.append(f"- Line: {item.location.line}")
        if item.location.column is not None:
            lines.append(f"- Column: {item.location.column}")
        lines += [f"- Description: {item.description}",
                  f"- Provenance: {item.provenance}"]
        if item.confidence is not None:
            lines.append(f"- Confidence: {_metric(item.confidence)}")
        lines.append("")
    return "\n".join(lines)


class AgentSkillExporter:
    """The sole knowledge input is a validated SkillRouteResult."""

    def plan(self, route: SkillRouteResult, target: CodeAgentTarget) -> AgentSkillExportPlan:
        if not isinstance(route, SkillRouteResult) or not isinstance(target, CodeAgentTarget):
            raise ValueError("a SkillRouteResult and supported target are required")
        evidence = {item.evidence_id: item for item in route.evidence}
        packages = []
        for match in route.matches:
            skill = match.skill
            selected = tuple(evidence[key] for key in skill.evidence_ids if key in evidence)
            if (tuple(item.evidence_id for item in selected) != skill.evidence_ids
                    or any(item.repository_id != route.repository_id for item in selected)):
                raise ValueError("package Evidence must exactly close source Skill references")
            name = skill_name(skill.skill_id)
            contents = {"SKILL.md": (_render_guidance(skill, name)
                                     if isinstance(skill, RuleGuidanceSkill)
                                     else _render_skill(skill, name)),
                        "references/evidence.md": _render_evidence(selected)}
            if isinstance(skill, RuleGuidanceSkill):
                contents["references/evidence.md"] += (
                    "## Semantic acceptance audit\n\n"
                    f"- Candidate: {skill.semantic_candidate_id}\n"
                    f"- Method: {skill.semantic_method}\n"
                    f"- Decision Evidence: {', '.join(skill.semantic_evidence_ids)}\n"
                    f"- Reason: {skill.semantic_reason}\n"
                )
            marker = {
                "schema": SCHEMA, "target": target.value,
                "repository_id": route.repository_id, "registry_id": route.registry_id,
                "source_snapshot_id": route.source_snapshot_id,
                "source_skill_id": skill.skill_id,
                "source_skill_version": skill.version,
                "agent_skill_name": name,
                "evidence_ids": list(skill.evidence_ids),
                "file_hashes": {key: digest(value) for key, value in contents.items()},
            }
            if isinstance(skill, RuleGuidanceSkill):
                marker["source_kind"] = "RULE"
                marker["source_rule_id"] = skill.source_rule_id
            contents[".skillforge.json"] = json.dumps(marker, ensure_ascii=False,
                                                         sort_keys=True, indent=2) + "\n"
            files = tuple(AgentSkillFile(key, value, digest(value))
                          for key, value in sorted(contents.items()))
            packages.append(AgentSkillPackage(name, skill.skill_id, skill.version, target,
                                               route.repository_id, route.registry_id,
                                               route.source_snapshot_id, skill.evidence_ids,
                                               files, skill))
        packages = tuple(sorted(packages, key=lambda item: item.agent_skill_name))
        return AgentSkillExportPlan(route.repository_id, route.registry_id,
                                    route.source_snapshot_id, target, packages,
                                    plan_fingerprint(route.repository_id, route.registry_id,
                                                     route.source_snapshot_id, target, packages))
