"""Independent Verify v1: immutable patch, plan, check, and report contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass
from enum import StrEnum
from pathlib import PurePath

from skillforge.domain.evidence import Confidence, Evidence
from skillforge.domain.serialization import to_json_value


def fingerprint(value: object) -> str:
    data = json.dumps(to_json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def canonical(values: tuple, name: str, key=lambda item: item) -> None:
    if not isinstance(values, tuple) or values != tuple(sorted(set(values), key=key)):
        raise ValueError(f"{name} must be sorted and unique")


_MANAGED_PATCH_FILE = re.compile(
    r"\.(?:codex|claude)/skills/skillforge-[0-9a-f]{32}/"
    r"(?:\.skillforge\.json|SKILL\.md|references/evidence\.md)"
)


def validate_patch_exclusions(paths: tuple[str, ...], *, managed: bool = False) -> None:
    """Validate exact repository-relative files excluded from a patch."""
    if (not isinstance(paths, tuple) or any(not isinstance(path, str) for path in paths)
            or paths != tuple(sorted(set(paths)))):
        raise ValueError("excluded patch paths must be a sorted unique tuple of strings")
    for path in paths:
        if (not path or path.startswith("/") or "\\" in path
                or ":" in path.split("/")[0]
                or any(part in {"", ".", ".."} for part in path.split("/"))):
            raise ValueError("excluded patch path must be canonical repository-relative")
        if managed and not _MANAGED_PATCH_FILE.fullmatch(path):
            raise ValueError("receipt exclusion is not a managed Agent Skill file")


class PatchChangeKind(StrEnum):
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    RENAMED = "RENAMED"
    COPIED = "COPIED"
    TYPE_CHANGED = "TYPE_CHANGED"
    UNTRACKED = "UNTRACKED"


@dataclass(frozen=True, slots=True)
class VerifyPatchFile:
    path: str
    kind: PatchChangeKind
    previous_path: str | None = None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        required(self.path, "patch path")
        if (self.path.startswith("/") or "\\" in self.path or
                ":" in self.path.split("/")[0] or
                any(part in {"", ".", ".."} for part in self.path.split("/"))):
            raise ValueError("patch path must be canonical repository-relative")
        if not isinstance(self.kind, PatchChangeKind):
            raise ValueError("invalid patch kind")
        if (self.previous_path is not None) != (
            self.kind in {PatchChangeKind.RENAMED, PatchChangeKind.COPIED}
        ):
            raise ValueError("previous path is required only for rename/copy")
        if self.previous_path is not None and (
            self.previous_path.startswith("/") or "\\" in self.previous_path or
            ":" in self.previous_path.split("/")[0] or
            any(part in {"", ".", ".."} for part in self.previous_path.split("/"))
        ):
            raise ValueError("previous path must be canonical repository-relative")
        if self.kind == PatchChangeKind.UNTRACKED and self.content_sha256 is None:
            raise ValueError("untracked content hash is required")


@dataclass(frozen=True, slots=True)
class GitPatchSnapshot:
    repository_id: str
    base_commit: str
    resolved_base_commit: str
    head_commit: str
    files: tuple[VerifyPatchFile, ...]
    tracked_diff_sha256: str
    snapshot_id: str

    def __post_init__(self) -> None:
        for name in ("repository_id", "base_commit", "resolved_base_commit", "head_commit",
                     "tracked_diff_sha256"):
            required(getattr(self, name), name)
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, VerifyPatchFile) for item in self.files
        ) or self.files != tuple(sorted(self.files, key=lambda f: (f.path, f.kind.value))):
            raise ValueError("patch files must be canonical")
        identities = tuple((item.path, item.kind, item.previous_path) for item in self.files)
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate patch file record")
        if self.snapshot_id != patch_fingerprint(self):
            raise ValueError("patch snapshot_id mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def patch_fingerprint(patch: GitPatchSnapshot) -> str:
    return fingerprint(("git-patch-v1", patch.repository_id, patch.resolved_base_commit,
                        patch.head_commit, patch.tracked_diff_sha256, patch.files))


class VerifyObligationKind(StrEnum):
    RULE = "RULE"
    WORKFLOW_STEP = "WORKFLOW_STEP"
    RISK = "RISK"


@dataclass(frozen=True, order=True, slots=True)
class VerifyTargetRef:
    kind: VerifyObligationKind
    skill_id: str
    ref_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VerifyObligationKind):
            raise ValueError("invalid obligation kind")
        required(self.skill_id, "skill_id")
        required(self.ref_id, "ref_id")


@dataclass(frozen=True, slots=True)
class VerifyObligation:
    target_ref: VerifyTargetRef
    description: str
    condition: str | None
    source_status: str
    evidence_ids: tuple[str, ...]
    source_confidence: Confidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_ref, VerifyTargetRef):
            raise ValueError("invalid target ref")
        required(self.description, "description")
        required(self.source_status, "source_status")
        canonical(self.evidence_ids, "obligation Evidence IDs")
        if self.source_confidence is not None and not isinstance(
            self.source_confidence, Confidence
        ):
            raise ValueError("source_confidence must be Confidence")


class VerifyCheckKind(StrEnum):
    TEST = "TEST"
    LINT = "LINT"
    TYPE_CHECK = "TYPE_CHECK"
    BUILD = "BUILD"
    CUSTOM = "CUSTOM"


class VerifyCheckStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True, slots=True)
class VerifyCommandSpec:
    check_id: str
    kind: VerifyCheckKind
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 300
    covers: tuple[VerifyTargetRef, ...] = ()

    def __post_init__(self) -> None:
        required(self.check_id, "check_id")
        if not isinstance(self.kind, VerifyCheckKind):
            raise ValueError("invalid check kind")
        if not isinstance(self.argv, tuple) or not self.argv or any(
            not isinstance(arg, str) for arg in self.argv
        ) or not self.argv[0].strip():
            raise ValueError("argv must be a nonempty tuple of strings")
        executable = PurePath(self.argv[0].replace("\\", "/")).name.casefold()
        shell_string = executable in {"sh", "bash", "zsh", "dash"} and any(
            arg.startswith("-") and "c" in arg[1:] for arg in self.argv[1:]
        )
        shell_string |= executable in {"cmd", "cmd.exe"} and any(
            arg.casefold() in {"/c", "/k"} for arg in self.argv[1:]
        )
        shell_string |= executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"} and any(
            arg.casefold() in {"-command", "-encodedcommand", "-c", "-enc"}
            for arg in self.argv[1:]
        )
        if shell_string:
            raise ValueError("shell command wrappers are not supported")
        if not isinstance(self.cwd, str) or not self.cwd or "\\" in self.cwd or (
            self.cwd != "." and (self.cwd.startswith("/") or
                                  ":" in self.cwd.split("/")[0] or
                                  any(part in {"", ".", ".."} for part in self.cwd.split("/")))
        ):
            raise ValueError("cwd must be a canonical repository-relative path")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        canonical(self.covers, "covers")
        if any(not isinstance(ref, VerifyTargetRef) for ref in self.covers):
            raise ValueError("covers must contain target refs")


@dataclass(frozen=True, slots=True)
class VerifyCheckResult:
    check_id: str
    kind: VerifyCheckKind
    status: VerifyCheckStatus
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    covered_refs: tuple[VerifyTargetRef, ...]

    def __post_init__(self) -> None:
        required(self.check_id, "check_id")
        if not isinstance(self.kind, VerifyCheckKind) or not isinstance(
            self.status, VerifyCheckStatus
        ):
            raise ValueError("invalid check result kind/status")
        if (not isinstance(self.argv, tuple) or not self.argv or
                any(not isinstance(arg, str) for arg in self.argv) or
                not self.argv[0].strip()):
            raise ValueError("result argv must be a nonempty tuple of strings")
        required(self.cwd, "result cwd")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("result stdout/stderr must be strings")
        if self.status in {VerifyCheckStatus.PASSED, VerifyCheckStatus.FAILED}:
            if type(self.exit_code) is not int or (
                self.status == VerifyCheckStatus.PASSED and self.exit_code != 0
            ) or (self.status == VerifyCheckStatus.FAILED and self.exit_code == 0):
                raise ValueError("check exit code contradicts status")
        elif self.exit_code is not None:
            raise ValueError("timeout/error cannot have exit code")
        if not isinstance(self.covered_refs, tuple) or any(
            not isinstance(ref, VerifyTargetRef) for ref in self.covered_refs
        ):
            raise ValueError("covered_refs must be a tuple of target refs")
        canonical(self.covered_refs, "covered_refs")


class VerifyFindingSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    CANDIDATE = "CANDIDATE"


@dataclass(frozen=True, slots=True)
class VerifyFinding:
    code: str
    severity: VerifyFindingSeverity
    message: str
    check_id: str | None = None
    target_ref: VerifyTargetRef | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required(self.code, "finding code")
        required(self.message, "finding message")
        if not isinstance(self.severity, VerifyFindingSeverity):
            raise ValueError("invalid finding severity")
        canonical(self.evidence_ids, "finding Evidence IDs")


class VerifyOverallStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class VerifyPlan:
    repository_id: str
    registry_id: str
    source_snapshot_id: str
    selected_skills: tuple[tuple[str, str], ...]
    obligations: tuple[VerifyObligation, ...]
    checks: tuple[VerifyCommandSpec, ...]
    evidence: tuple[Evidence, ...]
    plan_id: str

    def __post_init__(self) -> None:
        for name in ("repository_id", "registry_id", "source_snapshot_id"):
            required(getattr(self, name), name)
        canonical(self.selected_skills, "selected_skills")
        if any(not isinstance(item, tuple) or len(item) != 2 or
               any(not isinstance(part, str) or not part.strip() for part in item)
               for item in self.selected_skills):
            raise ValueError("selected skills require skill ID and version")
        if len({key for key, _ in self.selected_skills}) != len(self.selected_skills):
            raise ValueError("duplicate selected Skill ID")
        if not isinstance(self.obligations, tuple) or any(
            not isinstance(item, VerifyObligation) for item in self.obligations
        ):
            raise ValueError("obligations must be VerifyObligation tuple")
        refs = tuple(item.target_ref for item in self.obligations)
        canonical(refs, "obligations")
        if any(ref.skill_id not in {key for key, _ in self.selected_skills}
               for ref in refs):
            raise ValueError("obligation references unselected Skill")
        if not isinstance(self.checks, tuple) or any(
            not isinstance(item, VerifyCommandSpec) for item in self.checks
        ):
            raise ValueError("checks must be VerifyCommandSpec tuple")
        ids = tuple(item.check_id for item in self.checks)
        canonical(ids, "checks")
        valid = set(refs)
        if any(set(check.covers) - valid for check in self.checks):
            raise ValueError("check covers unknown obligation")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, Evidence) for item in self.evidence
        ):
            raise ValueError("Evidence must be an Evidence tuple")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        canonical(evidence_ids, "Evidence")
        if evidence_ids != tuple(sorted({eid for item in self.obligations
                                         for eid in item.evidence_ids})) or any(
            item.repository_id != self.repository_id for item in self.evidence
        ):
            raise ValueError("VerifyPlan Evidence closure mismatch")
        if self.plan_id != plan_fingerprint(self):
            raise ValueError("plan_id mismatch")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def plan_fingerprint(plan: VerifyPlan) -> str:
    return fingerprint(("verify-plan-v1", plan.repository_id, plan.registry_id,
                        plan.source_snapshot_id, plan.selected_skills, plan.obligations,
                        plan.checks, plan.evidence))


@dataclass(frozen=True, slots=True)
class PrepareVerifyReceipt:
    """Immutable prepare-time Verify seed bound to the exact Git base commit."""

    repository_id: str
    registry_id: str
    source_snapshot_id: str
    base_commit: str
    plan: VerifyPlan
    ignored_patch_paths: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        for name in ("repository_id", "registry_id", "source_snapshot_id", "base_commit"):
            required(getattr(self, name), name)
        if not isinstance(self.plan, VerifyPlan) or self.plan.checks:
            raise ValueError("receipt requires a checks-free VerifyPlan seed")
        if (self.repository_id, self.registry_id, self.source_snapshot_id) != (
            self.plan.repository_id, self.plan.registry_id, self.plan.source_snapshot_id
        ):
            raise ValueError("receipt identity does not match seed VerifyPlan")
        if (len(self.base_commit) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in self.base_commit)):
            raise ValueError("base_commit must be a full lowercase Git commit ID")
        validate_patch_exclusions(self.ignored_patch_paths, managed=True)
        expected = fingerprint(("prepare-verify-receipt-v2", self.plan.plan_id,
                                self.base_commit, self.ignored_patch_paths))
        if self.receipt_id != expected:
            raise ValueError("receipt_id mismatch")

    @classmethod
    def create(cls, plan: VerifyPlan, base_commit: str,
               ignored_patch_paths: tuple[str, ...] = ()) -> PrepareVerifyReceipt:
        if not isinstance(plan, VerifyPlan) or plan.checks:
            raise ValueError("receipt requires a checks-free VerifyPlan seed")
        validate_patch_exclusions(ignored_patch_paths, managed=True)
        receipt_id = fingerprint(("prepare-verify-receipt-v2", plan.plan_id, base_commit,
                                  ignored_patch_paths))
        return cls(plan.repository_id, plan.registry_id, plan.source_snapshot_id,
                   base_commit, plan, ignored_patch_paths, receipt_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "skillforge-prepare-verify-receipt-v2",
            "repository_id": self.repository_id,
            "registry_id": self.registry_id,
            "source_snapshot_id": self.source_snapshot_id,
            "base_commit": self.base_commit,
            "plan": self.plan.to_dict(),
            "ignored_patch_paths": list(self.ignored_patch_paths),
            "receipt_id": self.receipt_id,
        }


def validate_verify_check_results(
    plan: VerifyPlan, results: tuple[VerifyCheckResult, ...]
) -> None:
    """Require one result per planned check, in the plan's canonical order."""
    if not isinstance(results, tuple) or len(results) != len(plan.checks):
        raise ValueError("report checks must exactly match Plan checks")
    for result, spec in zip(results, plan.checks, strict=True):
        if not isinstance(result, VerifyCheckResult) or (
            result.check_id, result.kind, result.argv, result.cwd, result.covered_refs
        ) != (spec.check_id, spec.kind, spec.argv, spec.cwd, spec.covers):
            raise ValueError("check result does not match Plan check spec")


def derive_verify_findings(
    plan: VerifyPlan, results: tuple[VerifyCheckResult, ...]
) -> tuple[VerifyFinding, ...]:
    """Derive all v1 findings from the authoritative Plan and command results."""
    validate_verify_check_results(plan, results)
    obligations = {item.target_ref: item for item in plan.obligations}
    findings: list[VerifyFinding] = []
    covered: set[VerifyTargetRef] = set()
    for check in results:
        covered.update(check.covered_refs)
        if check.status != VerifyCheckStatus.PASSED:
            findings.append(VerifyFinding(
                "CHECK_" + check.status.value, VerifyFindingSeverity.ERROR,
                f"Explicit check {check.check_id} returned {check.status.value}",
                check_id=check.check_id,
            ))
            for ref in check.covered_refs:
                findings.append(VerifyFinding(
                    "COVERED_CHECK_" + check.status.value,
                    VerifyFindingSeverity.ERROR,
                    f"Check {check.check_id} did not establish obligation {ref.ref_id}",
                    check_id=check.check_id, target_ref=ref,
                    evidence_ids=obligations[ref].evidence_ids,
                ))
    for item in plan.obligations:
        if item.target_ref not in covered:
            findings.append(VerifyFinding(
                "UNVERIFIED_OBLIGATION", VerifyFindingSeverity.CANDIDATE,
                f"No explicit checker covers {item.target_ref.ref_id}",
                target_ref=item.target_ref, evidence_ids=item.evidence_ids,
            ))
    return tuple(findings)


def derive_verify_overall_status(
    plan: VerifyPlan, results: tuple[VerifyCheckResult, ...],
    findings: tuple[VerifyFinding, ...],
) -> VerifyOverallStatus:
    """Freeze FAILED / INCOMPLETE / PASSED precedence for v1."""
    validate_verify_check_results(plan, results)
    if any(item.severity == VerifyFindingSeverity.ERROR for item in findings):
        return VerifyOverallStatus.FAILED
    covered = {ref for result in results for ref in result.covered_refs}
    if not results or covered != {item.target_ref for item in plan.obligations}:
        return VerifyOverallStatus.INCOMPLETE
    return VerifyOverallStatus.PASSED


@dataclass(frozen=True, slots=True)
class VerifyReport:
    repository_id: str
    registry_id: str
    source_snapshot_id: str
    plan_id: str
    patch: GitPatchSnapshot
    checks: tuple[VerifyCheckResult, ...]
    obligations: tuple[VerifyObligation, ...]
    findings: tuple[VerifyFinding, ...]
    overall_status: VerifyOverallStatus
    evidence: tuple[Evidence, ...]
    plan: InitVar[VerifyPlan]

    def __post_init__(self, plan: VerifyPlan) -> None:
        if not isinstance(plan, VerifyPlan):
            raise ValueError("VerifyReport requires its original VerifyPlan")
        if (self.repository_id != plan.repository_id or
                self.registry_id != plan.registry_id or
                self.source_snapshot_id != plan.source_snapshot_id or
                self.plan_id != plan.plan_id):
            raise ValueError("report Plan identity mismatch")
        if self.repository_id != self.patch.repository_id:
            raise ValueError("report repository mismatch")
        if self.obligations != plan.obligations:
            raise ValueError("report obligations must exactly match Plan")
        validate_verify_check_results(plan, self.checks)
        expected_findings = derive_verify_findings(plan, self.checks)
        if self.findings != expected_findings:
            raise ValueError("report findings do not match canonical derivation")
        expected_status = derive_verify_overall_status(plan, self.checks, expected_findings)
        if self.overall_status != expected_status:
            raise ValueError("report overall status mismatch")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, Evidence) for item in self.evidence
        ):
            raise ValueError("report Evidence must be an Evidence tuple")
        expected = tuple(sorted({eid for item in self.obligations for eid in item.evidence_ids}
                                | {eid for item in self.findings for eid in item.evidence_ids}))
        if (tuple(item.evidence_id for item in self.evidence) != expected or
                any(item.repository_id != self.repository_id for item in self.evidence)):
            raise ValueError("report Evidence closure mismatch")
        plan_evidence = {item.evidence_id: item for item in plan.evidence}
        if any(item.evidence_id in plan_evidence and
               item != plan_evidence[item.evidence_id] for item in self.evidence):
            raise ValueError("report Evidence differs from Plan Evidence")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


# Backward-compatible name for the former unimplemented placeholder severity.
FindingSeverity = VerifyFindingSeverity
