"""Immutable, repository-local contracts for human feedback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass, field
from enum import StrEnum

from skillforge.domain.project_dna import ProjectDNA
from skillforge.domain.routing import SkillRouteResult, SkillRoutingRequest
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import ProjectSkill
from skillforge.domain.verification import VerifyReport, VerifyTargetRef


class FeedbackTargetKind(StrEnum):
    PROJECT_RULE = "PROJECT_RULE"
    PROJECT_WORKFLOW = "PROJECT_WORKFLOW"
    PROJECT_ANTI_PATTERN = "PROJECT_ANTI_PATTERN"
    PROJECT_SKILL = "PROJECT_SKILL"
    SKILL_ROUTE = "SKILL_ROUTE"
    VERIFY_RESULT = "VERIFY_RESULT"


class FeedbackKind(StrEnum):
    ACCURATE = "ACCURATE"
    INACCURATE = "INACCURATE"
    EXCEPTION = "EXCEPTION"
    PREFERRED_NEW_PATTERN = "PREFERRED_NEW_PATTERN"
    MISSING_STEP = "MISSING_STEP"
    WRONG_STEP = "WRONG_STEP"
    TOO_COARSE = "TOO_COARSE"
    TOO_FINE = "TOO_FINE"
    WRONG_ROUTE = "WRONG_ROUTE"
    MISSING_SKILL = "MISSING_SKILL"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    PERMANENT_WHITELIST = "PERMANENT_WHITELIST"
    MODULE_EXCEPTION = "MODULE_EXCEPTION"
    CONFIRM = "CONFIRM"
    DEPRECATE = "DEPRECATE"
    MARK_CONFLICTED = "MARK_CONFLICTED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    ADD_RULE_EXCEPTION = "ADD_RULE_EXCEPTION"


_KNOWLEDGE_KINDS = frozenset({
    FeedbackTargetKind.PROJECT_RULE,
    FeedbackTargetKind.PROJECT_WORKFLOW,
    FeedbackTargetKind.PROJECT_ANTI_PATTERN,
})
_KNOWLEDGE_SIGNALS = frozenset({
    FeedbackKind.ACCURATE,
    FeedbackKind.INACCURATE,
    FeedbackKind.EXCEPTION,
    FeedbackKind.PREFERRED_NEW_PATTERN,
})
_SKILL_SIGNALS = frozenset({
    FeedbackKind.MISSING_STEP,
    FeedbackKind.WRONG_STEP,
    FeedbackKind.TOO_COARSE,
    FeedbackKind.TOO_FINE,
})
_ROUTE_SIGNALS = frozenset({FeedbackKind.WRONG_ROUTE, FeedbackKind.MISSING_SKILL})
_VERIFY_SIGNALS = frozenset({
    FeedbackKind.FALSE_POSITIVE,
    FeedbackKind.FALSE_NEGATIVE,
    FeedbackKind.PERMANENT_WHITELIST,
    FeedbackKind.MODULE_EXCEPTION,
})
GOVERNANCE_KINDS = frozenset({
    FeedbackKind.CONFIRM,
    FeedbackKind.DEPRECATE,
    FeedbackKind.MARK_CONFLICTED,
    FeedbackKind.MANUAL_OVERRIDE,
    FeedbackKind.ADD_RULE_EXCEPTION,
})


def _required_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class SkillIdentity:
    skill_id: str
    version: str

    def __post_init__(self) -> None:
        _required_text(self.skill_id, "skill_id")
        _required_text(self.version, "Skill version")


@dataclass(frozen=True, slots=True)
class KnowledgeFeedbackTarget:
    target_kind: FeedbackTargetKind
    repository_id: str
    source_snapshot_id: str
    knowledge_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_kind, FeedbackTargetKind) or (
            self.target_kind not in _KNOWLEDGE_KINDS
        ):
            raise ValueError("knowledge target kind must identify Project DNA knowledge")
        _required_text(self.repository_id, "repository_id")
        _required_text(self.source_snapshot_id, "source_snapshot_id")
        _required_text(self.knowledge_id, "knowledge_id")

    @property
    def logical_identity(self) -> tuple[object, ...]:
        return (self.target_kind, self.knowledge_id)


@dataclass(frozen=True, slots=True)
class SkillFeedbackTarget:
    repository_id: str
    source_snapshot_id: str
    skill_id: str
    version: str
    source_skill: InitVar[ProjectSkill]
    target_kind: FeedbackTargetKind = field(
        default=FeedbackTargetKind.PROJECT_SKILL, init=False
    )

    def __post_init__(self, source_skill: ProjectSkill) -> None:
        _required_text(self.repository_id, "repository_id")
        _required_text(self.source_snapshot_id, "source_snapshot_id")
        _required_text(self.skill_id, "skill_id")
        _required_text(self.version, "Skill version")
        if not isinstance(source_skill, ProjectSkill) or (
            self.repository_id != source_skill.repository_id
            or self.source_snapshot_id != source_skill.source_snapshot_id
            or self.skill_id != source_skill.skill_id
            or self.version != source_skill.version
        ):
            raise ValueError("Skill target must match its source ProjectSkill")

    @property
    def logical_identity(self) -> tuple[object, ...]:
        return (self.target_kind, self.skill_id, self.version)


@dataclass(frozen=True, slots=True)
class RouteFeedbackTarget:
    repository_id: str
    source_snapshot_id: str
    registry_id: str
    routing_request: SkillRoutingRequest
    selected_skills: tuple[SkillIdentity, ...]
    source_route: InitVar[SkillRouteResult]
    target_kind: FeedbackTargetKind = field(default=FeedbackTargetKind.SKILL_ROUTE, init=False)

    def __post_init__(self, source_route: SkillRouteResult) -> None:
        _required_text(self.repository_id, "repository_id")
        _required_text(self.source_snapshot_id, "source_snapshot_id")
        _required_text(self.registry_id, "registry_id")
        if not isinstance(self.routing_request, SkillRoutingRequest):
            raise ValueError("route target requires a SkillRoutingRequest")
        if self.routing_request.repository_id != self.repository_id:
            raise ValueError("route request repository_id mismatch")
        if not isinstance(self.selected_skills, tuple) or any(
            not isinstance(item, SkillIdentity) for item in self.selected_skills
        ):
            raise ValueError("selected Skills must be SkillIdentity records")
        if self.selected_skills != tuple(sorted(set(self.selected_skills))):
            raise ValueError("selected Skill identities must be sorted and unique")
        if len({item.skill_id for item in self.selected_skills}) != len(self.selected_skills):
            raise ValueError("selected Skill IDs must be unique")
        if not isinstance(source_route, SkillRouteResult) or (
            self.repository_id != source_route.repository_id
            or self.source_snapshot_id != source_route.source_snapshot_id
            or self.registry_id != source_route.registry_id
            or self.routing_request != source_route.request
            or self.selected_skills != tuple(sorted(
                SkillIdentity(item.skill.skill_id, item.skill.version)
                for item in source_route.matches
            ))
        ):
            raise ValueError("Route target must match its source SkillRouteResult")

    @property
    def logical_identity(self) -> tuple[object, ...]:
        return (
            self.target_kind,
            self.registry_id,
            self.routing_request,
            self.selected_skills,
        )


@dataclass(frozen=True, slots=True)
class VerifyFeedbackTarget:
    repository_id: str
    source_snapshot_id: str
    plan_id: str
    patch_snapshot_id: str
    source_report: InitVar[VerifyReport]
    target_ref: VerifyTargetRef | None = None
    check_id: str | None = None
    target_kind: FeedbackTargetKind = field(default=FeedbackTargetKind.VERIFY_RESULT, init=False)

    def __post_init__(self, source_report: VerifyReport) -> None:
        for name in ("repository_id", "source_snapshot_id", "plan_id", "patch_snapshot_id"):
            _required_text(getattr(self, name), name)
        if self.target_ref is not None and not isinstance(self.target_ref, VerifyTargetRef):
            raise ValueError("target_ref must be a VerifyTargetRef")
        if self.check_id is not None:
            _required_text(self.check_id, "check_id")
        if self.target_ref is None and self.check_id is None:
            raise ValueError("Verify feedback must identify an obligation or check")
        if not isinstance(source_report, VerifyReport) or (
            self.repository_id != source_report.repository_id
            or self.source_snapshot_id != source_report.source_snapshot_id
            or self.plan_id != source_report.plan_id
            or self.patch_snapshot_id != source_report.patch.snapshot_id
        ):
            raise ValueError("Verify target must match its source VerifyReport")
        if self.target_ref is not None and self.target_ref not in {
            item.target_ref for item in source_report.obligations
        }:
            raise ValueError("unknown Verify obligation target")
        matching_check = next(
            (item for item in source_report.checks if item.check_id == self.check_id), None
        )
        if self.check_id is not None and matching_check is None:
            raise ValueError("unknown Verify check_id")
        if (self.target_ref is not None and matching_check is not None
                and self.target_ref not in matching_check.covered_refs):
            raise ValueError("Verify check does not cover the target obligation")

    @property
    def logical_identity(self) -> tuple[object, ...]:
        return (
            self.target_kind,
            self.plan_id,
            self.patch_snapshot_id,
            self.target_ref,
            self.check_id,
        )


FeedbackTarget = (
    KnowledgeFeedbackTarget | SkillFeedbackTarget | RouteFeedbackTarget | VerifyFeedbackTarget
)


def _validate_feedback_pair(kind: FeedbackKind, target: FeedbackTarget) -> None:
    target_kind = target.target_kind
    valid = (
        (kind in _KNOWLEDGE_SIGNALS and target_kind in _KNOWLEDGE_KINDS)
        or (kind in _SKILL_SIGNALS and target_kind == FeedbackTargetKind.PROJECT_SKILL)
        or (kind in _ROUTE_SIGNALS and target_kind == FeedbackTargetKind.SKILL_ROUTE)
        or (kind in _VERIFY_SIGNALS and target_kind == FeedbackTargetKind.VERIFY_RESULT)
        or (kind in {FeedbackKind.CONFIRM, FeedbackKind.DEPRECATE,
                     FeedbackKind.MARK_CONFLICTED}
            and target_kind in _KNOWLEDGE_KINDS)
        or (kind in {FeedbackKind.MANUAL_OVERRIDE, FeedbackKind.ADD_RULE_EXCEPTION}
            and target_kind == FeedbackTargetKind.PROJECT_RULE)
    )
    if not valid:
        raise ValueError(f"{kind.value} is not valid for {target_kind.value}")


@dataclass(frozen=True, slots=True)
class HumanFeedback:
    repository_id: str
    source_snapshot_id: str
    target: FeedbackTarget
    kind: FeedbackKind
    rationale: str
    exception_description: str | None = None
    supersedes_feedback_id: str | None = None
    feedback_id: str = field(init=False)

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        _required_text(self.source_snapshot_id, "source_snapshot_id")
        _required_text(self.rationale, "rationale")
        if not isinstance(self.target, (
            KnowledgeFeedbackTarget,
            SkillFeedbackTarget,
            RouteFeedbackTarget,
            VerifyFeedbackTarget,
        )):
            raise ValueError("target must be a validated Human Feedback target")
        if not isinstance(self.kind, FeedbackKind):
            raise ValueError("kind must be FeedbackKind")
        _validate_feedback_pair(self.kind, self.target)
        if (self.repository_id != self.target.repository_id
                or self.source_snapshot_id != self.target.source_snapshot_id):
            raise ValueError("feedback and target repository/snapshot identities must match")
        if self.kind == FeedbackKind.ADD_RULE_EXCEPTION:
            _required_text(self.exception_description, "exception_description")
        elif self.exception_description is not None:
            raise ValueError("exception_description is only valid for ADD_RULE_EXCEPTION")
        if self.supersedes_feedback_id is not None:
            _required_text(self.supersedes_feedback_id, "supersedes_feedback_id")
        content = {
            "schema": "human-feedback-v1",
            "repository_id": self.repository_id,
            "source_snapshot_id": self.source_snapshot_id,
            "target": self.target,
            "kind": self.kind,
            "rationale": self.rationale,
            "exception_description": self.exception_description,
            "supersedes_feedback_id": self.supersedes_feedback_id,
        }
        feedback_id = _fingerprint(content)
        if feedback_id == self.supersedes_feedback_id:
            raise ValueError("feedback cannot supersede itself")
        object.__setattr__(self, "feedback_id", feedback_id)

    @property
    def logical_target(self) -> tuple[object, ...]:
        return self.target.logical_identity

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class HumanFeedbackLedger:
    repository_id: str
    records: tuple[HumanFeedback, ...]
    ledger_id: str = field(init=False)

    def __post_init__(self) -> None:
        _required_text(self.repository_id, "repository_id")
        if not isinstance(self.records, tuple) or any(
            not isinstance(item, HumanFeedback) for item in self.records
        ):
            raise ValueError("ledger records must be a tuple of HumanFeedback")
        ids = tuple(item.feedback_id for item in self.records)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate feedback_id")
        if any(item.repository_id != self.repository_id for item in self.records):
            raise ValueError("feedback repository_id mismatch")
        records = tuple(sorted(self.records, key=lambda item: item.feedback_id))
        by_id = {item.feedback_id: item for item in records}
        for item in records:
            prior_id = item.supersedes_feedback_id
            if prior_id is None:
                continue
            prior = by_id.get(prior_id)
            if prior is None:
                raise ValueError("superseded feedback must exist in the same ledger")
            if prior.repository_id != item.repository_id:
                raise ValueError("superseded feedback repository_id mismatch")
            if prior.logical_target != item.logical_target:
                raise ValueError("superseded feedback must target the same logical artifact")
        for item in records:
            visited: set[str] = set()
            current = item
            while current.supersedes_feedback_id is not None:
                if current.feedback_id in visited:
                    raise ValueError("feedback supersession cycle")
                visited.add(current.feedback_id)
                current = by_id[current.supersedes_feedback_id]
                if current.feedback_id in visited:
                    raise ValueError("feedback supersession cycle")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "ledger_id",
            _fingerprint(("human-feedback-ledger-v1", self.repository_id, records)),
        )

    @property
    def active_records(self) -> tuple[HumanFeedback, ...]:
        superseded = {
            item.supersedes_feedback_id
            for item in self.records
            if item.supersedes_feedback_id is not None
        }
        return tuple(item for item in self.records if item.feedback_id not in superseded)

    def append(self, record: HumanFeedback) -> HumanFeedbackLedger:
        if not isinstance(record, HumanFeedback):
            raise ValueError("record must be HumanFeedback")
        if record.repository_id != self.repository_id:
            raise ValueError("feedback repository_id mismatch")
        return HumanFeedbackLedger(self.repository_id, (*self.records, record))

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class FeedbackApplicationResult:
    repository_id: str
    source_snapshot_id: str
    result_snapshot_id: str
    applied_feedback_ids: tuple[str, ...]
    diagnostic_feedback_ids: tuple[str, ...]
    resulting_project_dna: ProjectDNA
    source_dna: InitVar[ProjectDNA]
    ledger: InitVar[HumanFeedbackLedger]

    def __post_init__(
        self, source_dna: ProjectDNA, ledger: HumanFeedbackLedger
    ) -> None:
        for name in ("repository_id", "source_snapshot_id", "result_snapshot_id"):
            _required_text(getattr(self, name), name)
        for name in ("applied_feedback_ids", "diagnostic_feedback_ids"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(value, str) or not value.strip() for value in values)
                    or values != tuple(sorted(set(values)))):
                raise ValueError(f"{name} must be a sorted unique tuple of IDs")
        if set(self.applied_feedback_ids) & set(self.diagnostic_feedback_ids):
            raise ValueError("feedback cannot be both applied and diagnostic")
        if not isinstance(source_dna, ProjectDNA) or not isinstance(
            ledger, HumanFeedbackLedger
        ):
            raise ValueError("application result requires source ProjectDNA and Ledger")
        if (self.repository_id != source_dna.repository_id
                or self.repository_id != ledger.repository_id
                or self.source_snapshot_id != source_dna.snapshot_id):
            raise ValueError("application source Project DNA/Ledger identity mismatch")
        active = ledger.active_records
        if any(item.source_snapshot_id != source_dna.snapshot_id for item in active):
            raise ValueError("application feedback source snapshot is stale")
        expected_applied = tuple(sorted(
            item.feedback_id for item in active if item.kind in GOVERNANCE_KINDS
        ))
        expected_diagnostic = tuple(sorted(
            item.feedback_id for item in active if item.kind not in GOVERNANCE_KINDS
        ))
        if (self.applied_feedback_ids != expected_applied
                or self.diagnostic_feedback_ids != expected_diagnostic):
            raise ValueError("application feedback IDs must close over active Ledger records")
        if not isinstance(self.resulting_project_dna, ProjectDNA):
            raise ValueError("resulting_project_dna must be ProjectDNA")
        if (self.resulting_project_dna.repository_id != self.repository_id
                or self.resulting_project_dna.snapshot_id != self.result_snapshot_id):
            raise ValueError("result Project DNA identity mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)
