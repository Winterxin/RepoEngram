"""Deterministic recording and governance application for human feedback."""

from __future__ import annotations

from dataclasses import replace

from skillforge.domain.feedback import (
    GOVERNANCE_KINDS,
    FeedbackApplicationResult,
    FeedbackKind,
    FeedbackTargetKind,
    HumanFeedback,
    HumanFeedbackLedger,
    KnowledgeFeedbackTarget,
    RouteFeedbackTarget,
    SkillFeedbackTarget,
    SkillIdentity,
    VerifyFeedbackTarget,
)
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectRuleException,
    ProjectWorkflow,
)
from skillforge.domain.routing import SkillRouteResult
from skillforge.domain.skills import ProjectSkill
from skillforge.domain.verification import VerifyReport, VerifyTargetRef

_KNOWLEDGE_ATTR = {
    FeedbackTargetKind.PROJECT_RULE: ("rules", "rule_id", ProjectRule),
    FeedbackTargetKind.PROJECT_WORKFLOW: ("workflows", "workflow_id", ProjectWorkflow),
    FeedbackTargetKind.PROJECT_ANTI_PATTERN: (
        "anti_patterns", "anti_pattern_id", ProjectAntiPattern
    ),
}


def _knowledge_items(dna: ProjectDNA, target_kind: FeedbackTargetKind):
    collection, identifier, _ = _KNOWLEDGE_ATTR[target_kind]
    return getattr(dna, collection), identifier


class HumanFeedbackService:
    """Creates feedback from real immutable artifacts and applies only explicit decisions."""

    def record_knowledge(
        self,
        dna: ProjectDNA,
        target_kind: FeedbackTargetKind,
        knowledge_id: str,
        kind: FeedbackKind,
        rationale: str,
        *,
        exception_description: str | None = None,
        supersedes_feedback_id: str | None = None,
    ) -> HumanFeedback:
        if not isinstance(dna, ProjectDNA):
            raise ValueError("knowledge feedback requires ProjectDNA")
        if target_kind not in _KNOWLEDGE_ATTR:
            raise ValueError("target_kind must identify Project DNA knowledge")
        collection, identifier = _knowledge_items(dna, target_kind)
        if not any(getattr(item, identifier) == knowledge_id for item in collection):
            raise ValueError(f"unknown {target_kind.value} knowledge ID: {knowledge_id}")
        target = KnowledgeFeedbackTarget(
            target_kind, dna.repository_id, dna.snapshot_id, knowledge_id
        )
        return HumanFeedback(
            dna.repository_id,
            dna.snapshot_id,
            target,
            kind,
            rationale,
            exception_description,
            supersedes_feedback_id,
        )

    def record_skill(
        self,
        skill: ProjectSkill,
        kind: FeedbackKind,
        rationale: str,
        *,
        expected_skill_id: str | None = None,
        expected_version: str | None = None,
        expected_source_snapshot_id: str | None = None,
        supersedes_feedback_id: str | None = None,
    ) -> HumanFeedback:
        if not isinstance(skill, ProjectSkill):
            raise ValueError("Skill feedback requires ProjectSkill")
        for name, expected, actual in (
            ("skill_id", expected_skill_id, skill.skill_id),
            ("Skill version", expected_version, skill.version),
            ("source_snapshot_id", expected_source_snapshot_id, skill.source_snapshot_id),
        ):
            if expected is not None and expected != actual:
                raise ValueError(f"feedback {name} does not match the supplied ProjectSkill")
        target = SkillFeedbackTarget(
            skill.repository_id, skill.source_snapshot_id, skill.skill_id, skill.version, skill
        )
        return HumanFeedback(
            skill.repository_id, skill.source_snapshot_id, target, kind, rationale,
            supersedes_feedback_id=supersedes_feedback_id,
        )

    def record_route(
        self,
        route: SkillRouteResult,
        kind: FeedbackKind,
        rationale: str,
        *,
        supersedes_feedback_id: str | None = None,
    ) -> HumanFeedback:
        if not isinstance(route, SkillRouteResult):
            raise ValueError("route feedback requires SkillRouteResult")
        target = RouteFeedbackTarget(
            route.repository_id,
            route.source_snapshot_id,
            route.registry_id,
            route.request,
            tuple(sorted(SkillIdentity(item.skill.skill_id, item.skill.version)
                         for item in route.matches)),
            route,
        )
        return HumanFeedback(
            route.repository_id, route.source_snapshot_id, target, kind, rationale,
            supersedes_feedback_id=supersedes_feedback_id,
        )

    def record_verify(
        self,
        report: VerifyReport,
        kind: FeedbackKind,
        rationale: str,
        *,
        target_ref: VerifyTargetRef | None = None,
        check_id: str | None = None,
        expected_plan_id: str | None = None,
        expected_patch_snapshot_id: str | None = None,
        supersedes_feedback_id: str | None = None,
    ) -> HumanFeedback:
        if not isinstance(report, VerifyReport):
            raise ValueError("Verify feedback requires a real VerifyReport")
        if expected_plan_id is not None and expected_plan_id != report.plan_id:
            raise ValueError("unknown Verify plan_id")
        if (expected_patch_snapshot_id is not None
                and expected_patch_snapshot_id != report.patch.snapshot_id):
            raise ValueError("unknown Verify patch_snapshot_id")
        target = VerifyFeedbackTarget(
            report.repository_id,
            report.source_snapshot_id,
            report.plan_id,
            report.patch.snapshot_id,
            report,
            target_ref,
            check_id,
        )
        return HumanFeedback(
            report.repository_id, report.source_snapshot_id, target, kind, rationale,
            supersedes_feedback_id=supersedes_feedback_id,
        )

    def append(
        self, ledger: HumanFeedbackLedger, record: HumanFeedback
    ) -> HumanFeedbackLedger:
        if not isinstance(ledger, HumanFeedbackLedger):
            raise ValueError("ledger must be HumanFeedbackLedger")
        return ledger.append(record)

    def apply(
        self, dna: ProjectDNA, ledger: HumanFeedbackLedger
    ) -> FeedbackApplicationResult:
        if not isinstance(dna, ProjectDNA) or not isinstance(ledger, HumanFeedbackLedger):
            raise ValueError("apply requires ProjectDNA and HumanFeedbackLedger")
        if dna.repository_id != ledger.repository_id:
            raise ValueError("Project DNA and Feedback Ledger repository_id mismatch")

        active = ledger.active_records
        stale = tuple(
            item.feedback_id for item in active
            if item.source_snapshot_id != dna.snapshot_id
        )
        if stale:
            raise ValueError(
                "stale feedback source_snapshot_id does not match Project DNA: "
                + ", ".join(stale)
            )

        for item in active:
            if not isinstance(item.target, KnowledgeFeedbackTarget):
                continue
            collection, identifier = _knowledge_items(dna, item.target.target_kind)
            if not any(
                getattr(knowledge, identifier) == item.target.knowledge_id
                for knowledge in collection
            ):
                raise ValueError(f"unknown knowledge ID: {item.target.knowledge_id}")

        decisions: dict[tuple[object, ...], list[HumanFeedback]] = {}
        diagnostics = []
        for item in active:
            if item.kind in GOVERNANCE_KINDS:
                decisions.setdefault(item.logical_target, []).append(item)
            else:
                diagnostics.append(item.feedback_id)
        conflicts = tuple(
            records for records in decisions.values() if len(records) > 1
        )
        if conflicts:
            ids = sorted(record.feedback_id for group in conflicts for record in group)
            raise ValueError("conflicting active governance decisions: " + ", ".join(ids))

        rules = {item.rule_id: item for item in dna.rules}
        workflows = {item.workflow_id: item for item in dna.workflows}
        anti_patterns = {item.anti_pattern_id: item for item in dna.anti_patterns}
        applied = []
        for records in decisions.values():
            feedback = records[0]
            target = feedback.target
            if not isinstance(target, KnowledgeFeedbackTarget):
                raise ValueError("governance feedback must target Project DNA knowledge")
            if target.target_kind == FeedbackTargetKind.PROJECT_RULE:
                current = rules[target.knowledge_id]
            elif target.target_kind == FeedbackTargetKind.PROJECT_WORKFLOW:
                current = workflows[target.knowledge_id]
            else:
                current = anti_patterns[target.knowledge_id]
            status = current.status

            if feedback.kind == FeedbackKind.CONFIRM:
                if status in {
                    ProjectKnowledgeStatus.DEPRECATED,
                    ProjectKnowledgeStatus.MANUAL_OVERRIDE,
                }:
                    raise ValueError(f"cannot CONFIRM knowledge in {status.value} status")
                updated = replace(current, status=ProjectKnowledgeStatus.CONFIRMED)
            elif feedback.kind == FeedbackKind.DEPRECATE:
                updated = replace(current, status=ProjectKnowledgeStatus.DEPRECATED)
            elif feedback.kind == FeedbackKind.MARK_CONFLICTED:
                if status == ProjectKnowledgeStatus.DEPRECATED:
                    raise ValueError("cannot mark DEPRECATED knowledge as CONFLICTED")
                updated = replace(current, status=ProjectKnowledgeStatus.CONFLICTED)
            elif feedback.kind == FeedbackKind.MANUAL_OVERRIDE:
                if not isinstance(current, ProjectRule):
                    raise ValueError("MANUAL_OVERRIDE is only supported for ProjectRule")
                updated = replace(current, status=ProjectKnowledgeStatus.MANUAL_OVERRIDE)
            elif feedback.kind == FeedbackKind.ADD_RULE_EXCEPTION:
                if not isinstance(current, ProjectRule):
                    raise ValueError("ADD_RULE_EXCEPTION is only supported for ProjectRule")
                exception = ProjectRuleException(feedback.exception_description or "", ())
                exceptions = current.exceptions
                if exception not in exceptions:
                    exceptions = (*exceptions, exception)
                updated = replace(
                    current,
                    status=ProjectKnowledgeStatus.MANUAL_OVERRIDE,
                    exceptions=exceptions,
                )
            else:
                raise ValueError(f"unsupported governance action: {feedback.kind.value}")

            if isinstance(current, ProjectRule):
                rules[current.rule_id] = updated
            elif isinstance(current, ProjectWorkflow):
                workflows[current.workflow_id] = updated
            else:
                anti_patterns[current.anti_pattern_id] = updated
            applied.append(feedback.feedback_id)

        if applied:
            resulting_dna = ProjectDNA(
                repository_id=dna.repository_id,
                rules=tuple(rules[key] for key in sorted(rules)),
                workflows=tuple(workflows[key] for key in sorted(workflows)),
                anti_patterns=tuple(anti_patterns[key] for key in sorted(anti_patterns)),
                evidence=dna.evidence,
                schema_version=dna.schema_version,
            )
        else:
            resulting_dna = dna
        return FeedbackApplicationResult(
            dna.repository_id,
            dna.snapshot_id,
            resulting_dna.snapshot_id,
            tuple(sorted(applied)),
            tuple(sorted(diagnostics)),
            resulting_dna,
            dna,
            ledger,
        )
