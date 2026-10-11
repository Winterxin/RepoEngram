"""Build exact Skill obligations and validate caller-declared checker coverage."""

from skillforge.domain.routing import SkillRouteResult
from skillforge.domain.skills import RuleGuidanceSkill
from skillforge.domain.verification import (
    VerifyCommandSpec,
    VerifyObligation,
    VerifyObligationKind,
    VerifyPlan,
    VerifyTargetRef,
    plan_fingerprint,
)


class VerifyPlanBuilder:
    def for_skills(self, route: SkillRouteResult, skill_ids: tuple[str, ...]) -> VerifyPlan:
        """Select consumed obligations from an intact, validated canonical route."""
        seed = self.build(route)
        if (not isinstance(skill_ids, tuple) or
                any(not isinstance(key, str) or not key for key in skill_ids) or
                len(set(skill_ids)) != len(skill_ids)):
            raise ValueError("skill_ids must be a tuple of unique nonempty strings")
        ids = set(skill_ids)
        if ids - {key for key, _ in seed.selected_skills}:
            raise ValueError("Skill not selected by this route")
        obligations = tuple(o for o in seed.obligations if o.target_ref.skill_id in ids)
        evidence_ids = {key for o in obligations for key in o.evidence_ids}
        return self._create_plan(
            seed.repository_id, seed.registry_id, seed.source_snapshot_id,
            tuple(item for item in seed.selected_skills if item[0] in ids),
            obligations, (), tuple(e for e in seed.evidence if e.evidence_id in evidence_ids),
        )

    def combine(
        self, plans: tuple[VerifyPlan, ...], checks: tuple[VerifyCommandSpec, ...] = ()
    ) -> VerifyPlan:
        """Union consumed knowledge without dropping the original task obligations."""
        if not plans or any(not isinstance(plan, VerifyPlan) or plan.checks for plan in plans):
            raise ValueError("combine requires nonempty checks-free Verify plans")
        first = plans[0]
        identity = (first.repository_id, first.registry_id, first.source_snapshot_id)
        selected, obligations, evidence = {}, {}, {}
        for plan in plans:
            if (plan.repository_id, plan.registry_id, plan.source_snapshot_id) != identity:
                raise ValueError("cannot combine different knowledge snapshots")
            for mapping, items, key in (
                (selected, plan.selected_skills, lambda item: item[0]),
                (obligations, plan.obligations, lambda item: item.target_ref),
                (evidence, plan.evidence, lambda item: item.evidence_id),
            ):
                for item in items:
                    item_key = key(item)
                    if item_key in mapping and mapping[item_key] != item:
                        raise ValueError("conflicting knowledge in Verify plans")
                    mapping[item_key] = item
        seed = self._create_plan(
            *identity,
            tuple(selected[k] for k in sorted(selected)),
            tuple(obligations[k] for k in sorted(obligations)),
            (),
            tuple(evidence[k] for k in sorted(evidence)),
        )
        return self.with_checks(seed, checks)

    def build(self, route: SkillRouteResult,
              checks: tuple[VerifyCommandSpec, ...] = ()) -> VerifyPlan:
        if not isinstance(route, SkillRouteResult):
            raise ValueError("route must be a SkillRouteResult")
        if not isinstance(checks, tuple) or any(
            not isinstance(check, VerifyCommandSpec) for check in checks
        ):
            raise ValueError("checks must be a tuple of VerifyCommandSpec")
        obligations = []
        selected = []
        for match in route.matches:
            skill = match.skill
            selected.append((skill.skill_id, skill.version))
            if isinstance(skill, RuleGuidanceSkill):
                # Advisory structural guidance has no executable Verify obligation.
                continue
            constraints = {item.rule_id: item for item in skill.constraints}
            steps = {item.step_id: item for item in skill.steps}
            for rule_id in skill.verify_rule_ids:
                rule = constraints[rule_id]
                obligations.append(VerifyObligation(
                    VerifyTargetRef(VerifyObligationKind.RULE, skill.skill_id, rule_id),
                    rule.statement, None, rule.source_status.value,
                    rule.evidence_ids, rule.source_confidence,
                ))
            for condition in skill.completion_conditions:
                step = steps[condition.step_id]
                obligations.append(VerifyObligation(
                    VerifyTargetRef(VerifyObligationKind.WORKFLOW_STEP, skill.skill_id,
                                    condition.step_id),
                    step.instruction, condition.condition, skill.source_status.value,
                    condition.evidence_ids, skill.source_confidence,
                ))
            for risk in skill.risks:
                obligations.append(VerifyObligation(
                    VerifyTargetRef(VerifyObligationKind.RISK, skill.skill_id,
                                    risk.anti_pattern_id),
                    risk.description, None, risk.source_status.value,
                    risk.evidence_ids, risk.confidence,
                ))
        obligations_tuple = tuple(sorted(obligations, key=lambda item: item.target_ref))
        evidence_ids = {eid for item in obligations_tuple for eid in item.evidence_ids}
        evidence = tuple(item for item in route.evidence if item.evidence_id in evidence_ids)
        sorted_checks = tuple(sorted(checks, key=lambda check: check.check_id))
        selected_skills = tuple(sorted(selected))
        return self._create_plan(
            route.repository_id, route.registry_id, route.source_snapshot_id,
            selected_skills, obligations_tuple, sorted_checks, evidence,
        )

    def with_checks(self, seed_plan: VerifyPlan,
                    checks: tuple[VerifyCommandSpec, ...]) -> VerifyPlan:
        """Add explicit checks to a persisted seed without re-deriving obligations."""
        if not isinstance(seed_plan, VerifyPlan):
            raise ValueError("seed_plan must be a VerifyPlan")
        if seed_plan.checks:
            raise ValueError("seed_plan must not contain checks")
        if not isinstance(checks, tuple) or any(
            not isinstance(check, VerifyCommandSpec) for check in checks
        ):
            raise ValueError("checks must be a tuple of VerifyCommandSpec")
        return self._create_plan(
            seed_plan.repository_id, seed_plan.registry_id, seed_plan.source_snapshot_id,
            seed_plan.selected_skills, seed_plan.obligations,
            tuple(sorted(checks, key=lambda check: check.check_id)), seed_plan.evidence,
        )

    @staticmethod
    def _create_plan(repository_id, registry_id, source_snapshot_id,
                     selected_skills, obligations, checks, evidence) -> VerifyPlan:
        provisional = object.__new__(VerifyPlan)
        for name, value in (("repository_id", repository_id),
                            ("registry_id", registry_id),
                            ("source_snapshot_id", source_snapshot_id),
                            ("selected_skills", selected_skills),
                            ("obligations", obligations), ("checks", checks),
                            ("evidence", evidence)):
            object.__setattr__(provisional, name, value)
        return VerifyPlan(repository_id, registry_id, source_snapshot_id,
                          selected_skills, obligations, checks, evidence,
                          plan_fingerprint(provisional))
