import json

import pytest

from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind
from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import (
    ProjectSkill,
    SkillCompletionCondition,
    SkillParameter,
    SkillStep,
    SkillStepKind,
    SkillTrigger,
)
from skillforge.domain.verification import VerifyFinding, VerifyFindingSeverity
from skillforge.domain.workflow import Workflow, WorkflowStep


def test_domain_contracts_serialize_to_plain_json():
    confidence = Confidence(0.75, "observed_frequency_v1", Support(3, 4))
    evidence = Evidence(
        "ev1",
        "repo1",
        EvidenceType.GIT,
        SourceLocation("src/a.py", "abc", 4),
        "Changed route",
        "git_diff",
        confidence,
    )
    rule = ProjectRule(
        "rule1",
        "repo1",
        RuleType.ARCHITECTURE,
        "Use service layer",
        ProjectScope(ProjectScopeKind.DIRECTORY, ("src/api/**",)),
        ProjectKnowledgeStatus.CANDIDATE,
        (evidence.evidence_id,),
        Support(3, 4),
        confidence,
    )
    change = ChangeSet(
        "repo1",
        "abc",
        "2026-09-24T00:00:00+00:00",
        (FileChange("src/a.py", FileChangeKind.MODIFIED),),
        ("ev1",),
    )
    workflow = Workflow(
        "wf1",
        "repo1",
        "add route",
        ProjectScope(ProjectScopeKind.REPOSITORY),
        (WorkflowStep("step1", "Add test", ("ev1",)),),
        (),
        ("ev1",),
        Support(3, 4),
        confidence,
    )
    skill = ProjectSkill(
        "skill1",
        "repo1",
        "snapshot1",
        "1",
        "Add route",
        SkillTrigger("new route", ProjectScope(ProjectScopeKind.REPOSITORY), ("ev1",)),
        (SkillParameter("name", "Route name"),),
        (SkillStep("step1", "Add test", SkillStepKind.REQUIRED, None, "required", ("ev1",)),),
        "wf1",
        ProjectKnowledgeStatus.CANDIDATE,
        Support(3, 4),
        confidence,
        ("ev1",),
        (),
        (),
        (),
        (SkillCompletionCondition("step1", None, ("ev1",)),),
        (),
        ("ev1",),
    )
    finding = VerifyFinding("MANUAL_REVIEW", VerifyFindingSeverity.WARNING, "Check route")
    payload = to_json_value((evidence, rule, change, workflow, skill, finding))
    assert json.loads(json.dumps(payload))[0]["kind"] == "git"
    assert payload[0]["confidence"]["method"] == "observed_frequency_v1"
    assert payload[1]["confidence"]["support"] == {"supporting": 3, "examined": 4}
    assert payload[2]["files"][0]["kind"] == "modified"


def test_evidence_and_rule_invariants():
    with pytest.raises(ValueError):
        SourceLocation("../outside.py")
    with pytest.raises(ValueError):
        SourceLocation("C:/outside.py")
    with pytest.raises(ValueError):
        Confidence(1.2, "frequency")
    with pytest.raises(ValueError):
        Support(5, 4)
    with pytest.raises(ValueError):
        ProjectRule(
            "r",
            "repo",
            RuleType.BUSINESS,
            "High risk rule",
            ProjectScope(ProjectScopeKind.REPOSITORY),
            ProjectKnowledgeStatus.CONFIRMED,
            (),
        )
