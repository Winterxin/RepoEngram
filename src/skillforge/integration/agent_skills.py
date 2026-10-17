"""Faithful, deterministic rendering of the Router's selected Skills."""

from __future__ import annotations

import json
from pathlib import Path

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
from skillforge.domain.project_dna import ProjectScope
from skillforge.domain.routing import SkillRouteResult
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import ProjectSkill, RuleGuidanceSkill
from skillforge.integration.knowledge_trace import KnowledgeTrace, trace_operation
from skillforge.integration.skill_instructions import render_executable, render_instruction_audit
from skillforge.integration.source_conditions import source_conditions
from skillforge.integration.source_constraints import (
    SourceConstraintExportPlan,
    plan_current_source_fallback,
)
from skillforge.integration.source_reads import SourceReadSession
from skillforge.integration.source_representations import source_representations
from skillforge.integration.source_rule_distillation import (
    ModelProtocol,
    plan_current_source_rules,
)


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


def _portable_description(trigger: object) -> str:
    description = f"{trigger.task_description} (scope: {_scope(trigger.scope)})"
    if len(description) > 1024 and len(trigger.scope.selectors) > 1:
        # Frontmatter is a discovery summary. The body retains every selector.
        scope = (
            f"{trigger.scope.kind.value}: {len(trigger.scope.selectors)} selectors; "
            "see full scope below"
        )
        description = f"{trigger.task_description} (scope: {scope})"
    if len(description) > 1024:
        raise ValueError(
            "portable Agent Skill description exceeds specification limit (1024 chars)"
        )
    return description


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
    description = _portable_description(skill.trigger)
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
    description = _portable_description(skill.trigger)
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


def _render_executable(skill: ProjectSkill | RuleGuidanceSkill, name: str) -> str:
    return render_executable(skill, name, _portable_description(skill.trigger))


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

    def plan(self, route: SkillRouteResult, target: CodeAgentTarget, *,
             repository_root: Path | None = None,
             trace: KnowledgeTrace | None = None) -> AgentSkillExportPlan:
        if not isinstance(route, SkillRouteResult) or not isinstance(target, CodeAgentTarget):
            raise ValueError("a SkillRouteResult and supported target are required")
        with trace_operation(repository_root, trace, route.repository_id, "export") as trace:
            reads = SourceReadSession(repository_root) if repository_root is not None else None
            packages = tuple(sorted(
                (self._package(route, target, match.skill, reads, trace)
                 for match in route.matches),
                key=lambda item: item.agent_skill_name,
            ))
            result = AgentSkillExportPlan(
                route.repository_id, route.registry_id, route.source_snapshot_id, target, packages,
                plan_fingerprint(route.repository_id, route.registry_id,
                                 route.source_snapshot_id, target, packages),
            )
            trace.finish("export", "EXPORTED" if packages else "NO_ROUTED_KNOWLEDGE",
                         object_id=result.plan_id,
                         output_bytes=sum(len(f.content.encode("utf-8"))
                                          for p in packages for f in p.files))
            return result

    def package(self, route: SkillRouteResult, target: CodeAgentTarget, skill_id: str, *,
                repository_root: Path | None = None,
                trace: KnowledgeTrace | None = None) -> AgentSkillPackage:
        """Render one already-routed Skill without rendering its siblings."""
        if not isinstance(route, SkillRouteResult) or not isinstance(target, CodeAgentTarget):
            raise ValueError("a SkillRouteResult and supported target are required")
        skill = next((match.skill for match in route.matches
                      if match.skill.skill_id == skill_id), None)
        if skill is None:
            raise ValueError("Skill not selected for this repository task")
        with trace_operation(repository_root, trace, route.repository_id, "export_one") as trace:
            reads = SourceReadSession(repository_root) if repository_root is not None else None
            package = self._package(route, target, skill, reads, trace)
            trace.finish("export_one", "EXPORTED", object_id=skill_id,
                         output_bytes=sum(len(f.content.encode("utf-8")) for f in package.files))
            return package

    def _package(self, route, target, skill, reads, trace=None) -> AgentSkillPackage:
        evidence = {item.evidence_id: item for item in route.evidence}
        selected = tuple(evidence[key] for key in skill.evidence_ids if key in evidence)
        if (tuple(item.evidence_id for item in selected) != skill.evidence_ids
                or any(item.repository_id != route.repository_id for item in selected)):
            raise ValueError("package Evidence must exactly close source Skill references")
        name = skill_name(skill.skill_id)
        if trace is not None:
            trace.record("selection", "ROUTED_SKILL", object_id=skill.skill_id,
                         evidence_ids=skill.evidence_ids)
        contents = {"SKILL.md": (_render_guidance(skill, name)
                                 if isinstance(skill, RuleGuidanceSkill)
                                 else _render_skill(skill, name)),
                    "references/evidence.md": _render_evidence(selected)}
        if skill.procedure is not None:
            contents["references/evidence.md"] += render_instruction_audit(skill)
        if reads is not None:
            contents["references/evidence.md"] = source_representations(
                reads.root, skill.trigger.scope, reads=reads,
            ) + source_conditions(
                reads.root, skill.trigger.scope, reads=reads,
            ) + contents["references/evidence.md"]
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
        if trace is not None:
            for file in files:
                trace.record("render", "FILE_READY", object_id=file.content_sha256,
                             path=file.relative_path,
                             output_bytes=len(file.content.encode("utf-8")))
        return AgentSkillPackage(name, skill.skill_id, skill.version, target,
                                           route.repository_id, route.registry_id,
                                           route.source_snapshot_id, skill.evidence_ids,
                                           files, skill)

    def current_source_fallback(
        self, route: SkillRouteResult, target: CodeAgentTarget, *,
        repository_root: Path, scope: ProjectScope, enabled: bool = False,
        trace: KnowledgeTrace | None = None,
    ) -> SourceConstraintExportPlan | None:
        """Opt-in, separate advisory export of explicitly scoped current source."""
        if enabled is not True:
            if trace is not None:
                trace.record("source_fallback", "DISABLED")
            return None
        if route.matches:
            if trace is not None:
                trace.record("source_fallback", "HISTORICAL_ROUTE_AVAILABLE")
            return None
        with trace_operation(repository_root, trace, route.repository_id,
                             "source_fallback") as trace:
            result = plan_current_source_fallback(
                route, target, repository_root=repository_root, scope=scope, trace=trace,
            )
            trace.finish("source_fallback", "EXPORTED" if result else "NO_PACKAGE",
                         object_id=result.plan_id if result else None,
                         output_bytes=sum(len(f.content.encode("utf-8"))
                                          for f in result.files) if result else 0)
            return result

    def current_source_rules(
        self, route: SkillRouteResult, target: CodeAgentTarget, *,
        repository_root: Path, scope: ProjectScope, enabled: bool = False,
        model: ModelProtocol | None = None,
        trace: KnowledgeTrace | None = None,
        task_aware_selection: bool = False,
    ) -> SourceConstraintExportPlan | None:
        """Opt-in, separate advisory distillation of explicitly scoped current source."""
        if enabled is not True or route.matches:
            return None
        with trace_operation(repository_root, trace, route.repository_id, "source_rules") as trace:
            trace.record("source_rules", "DISTILLATION_STARTED")
            trace.persist(repository_root)
            result = plan_current_source_rules(
                route, target, repository_root=repository_root, scope=scope,
                model=model, trace=trace, task_aware_selection=task_aware_selection,
            )
            trace.finish("source_rules", "EXPORTED" if result else "NO_SOURCE_PACKAGE",
                         object_id=result.plan_id if result else None,
                         output_bytes=sum(len(f.content.encode("utf-8"))
                                          for f in result.files) if result else 0)
            return result
