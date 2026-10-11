"""Portable rendering and canonical export contract tests."""

from dataclasses import replace

import pytest
from test_skill_registry import compilation, skill

from skillforge.domain.agent_integration import (
    AgentSkillFile,
    CodeAgentTarget,
    digest,
    skill_name,
)
from skillforge.domain.evidence import Confidence, Support
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus as Status,
)
from skillforge.domain.project_dna import ProjectRuleException, RuleType
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.skills import (
    SkillCompletionCondition,
    SkillConstraint,
    SkillConstraintEnforcement,
    SkillException,
    SkillRisk,
    SkillStep,
    SkillStepKind,
)
from skillforge.integration import AgentSkillExporter
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def route(skills, limit=5):
    registry = SkillRegistryBuilder().build(compilation(tuple(skills)))
    return SkillRouter().route(registry, SkillRoutingRequest("repo", "create route", limit=limit))


def complex_skill():
    original = skill("a")
    ids = ("a",)
    steps = (
        SkillStep("first", "Inspect", SkillStepKind.REQUIRED, None, "source reason", ids),
        SkillStep("second", "Consider", SkillStepKind.OPTIONAL, None, "optional reason", ids),
        SkillStep("third", "Change", SkillStepKind.CONDITIONAL, "when enabled",
                  "conditional reason", ids),
    )
    required = SkillConstraint("required", RuleType.TESTING, "Check observed result",
                               original.trigger.scope, Status.CONFIRMED,
                               SkillConstraintEnforcement.REQUIRED,
                               (ProjectRuleException("rule exception", ids),), ids,
                               Support(2, 3), Confidence(0.7, "rule", Support(2, 3)))
    advisory = SkillConstraint("advisory", RuleType.CONVENTION, "Prefer stable naming",
                               original.trigger.scope, Status.CANDIDATE,
                               SkillConstraintEnforcement.ADVISORY, (), ids)
    risk = SkillRisk("risk", "Repeat edit", "Observed repeated edit", "Avoid blind repeat",
                     original.trigger.scope, Status.CANDIDATE, Confidence(0.5, "risk"),
                     (SkillException("risk exception", ids),), ids)
    return replace(original, steps=steps, constraints=(required, advisory), risks=(risk,),
                   exceptions=(SkillException("workflow exception", ids),),
                   completion_conditions=(SkillCompletionCondition("first", None, ids),
                                          SkillCompletionCondition("third", "when enabled", ids)),
                   verify_rule_ids=("required",))


def test_portable_rendering_preserves_all_semantics_and_evidence():
    routed = route((complex_skill(),))
    codex = AgentSkillExporter().plan(routed, CodeAgentTarget.CODEX)
    claude = AgentSkillExporter().plan(routed, CodeAgentTarget.CLAUDE_CODE)
    package = codex.packages[0]
    files = {item.relative_path: item.content for item in package.files}
    markdown = files["SKILL.md"]
    assert markdown == next(item.content for item in claude.packages[0].files
                            if item.relative_path == "SKILL.md")
    assert markdown.index("Inspect") < markdown.index("Consider") < markdown.index("Change")
    for text in ("[REQUIRED]", "[OPTIONAL]", "[CONDITIONAL]", "when enabled",
                 "source reason", "optional reason", "conditional reason",
                 "Required constraints", "Advisory constraints", "Check observed result",
                 "Prefer stable naming", "rule exception", "Repeat edit", "risk exception",
                 "workflow exception", "Completion conditions", "Verification references",
                 '"supporting": 2', '"score": 0.7', "Source status: candidate"):
        assert text in markdown
    assert "not executable verification commands" in markdown
    assert "pytest " not in markdown and "ruff " not in markdown
    assert files["references/evidence.md"].count("## a") == 1
    assert "- Provenance: fixture" in files["references/evidence.md"]
    assert package.evidence_ids == ("a",)
    assert codex == AgentSkillExporter().plan(routed, CodeAgentTarget.CODEX)


def test_top_k_and_per_skill_evidence_isolation():
    routed = route(tuple(skill(str(index)) for index in range(10)), limit=3)
    plan = AgentSkillExporter().plan(routed, CodeAgentTarget.CODEX)
    assert len(plan.packages) == 3
    assert {item.source_skill_id for item in plan.packages} == {"0", "1", "2"}
    for package in plan.packages:
        evidence = next(item.content for item in package.files
                        if item.relative_path == "references/evidence.md")
        assert evidence.count("## ") == 1
        assert f"## {package.source_skill_id}" in evidence


def test_frontmatter_quotes_special_characters_without_injection():
    special = 'do: # "quoted"\n---\nUnicode 中文'
    item = skill("a", task=special)
    registry = SkillRegistryBuilder().build(compilation((item,)))
    routed = SkillRouter().route(registry, SkillRoutingRequest("repo", special))
    markdown = AgentSkillExporter().plan(routed, CodeAgentTarget.CODEX).packages[0].files[1].content
    assert markdown.startswith('---\nname: "skillforge-')
    frontmatter = markdown.split("\n---\n", 1)[0]
    assert '\\n---\\nUnicode 中文' in frontmatter
    assert frontmatter.count("\n") == 2


def test_portable_name_and_description_length_boundaries():
    import re

    generated = skill_name("a")
    assert 1 <= len(generated) <= 64
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", generated)
    suffix = " (scope: repository: (whole repository))"

    def export(task):
        item = skill("a", task=task)
        registry = SkillRegistryBuilder().build(compilation((item,)))
        routed = SkillRouter().route(registry, SkillRoutingRequest("repo", task))
        return AgentSkillExporter().plan(routed, CodeAgentTarget.CODEX)

    valid = export("x" * (1024 - len(suffix)))
    markdown = next(file.content for file in valid.packages[0].files
                    if file.relative_path == "SKILL.md")
    assert len(markdown.split("\n", 3)[2].removeprefix('description: "').removesuffix('"')) == 1024
    assert valid.packages[0].agent_skill_name == generated
    with pytest.raises(ValueError, match="description exceeds specification limit"):
        export("x" * (1025 - len(suffix)))


def test_direct_contract_rejects_bad_files_packages_and_plans():
    plan = AgentSkillExporter().plan(route((skill("a"), skill("b"))), CodeAgentTarget.CODEX)
    package = plan.packages[0]
    for path in ("/absolute", "../outside", "a/../b", "a\\b", "C:/outside"):
        with pytest.raises(ValueError):
            AgentSkillFile(path, "x", digest("x"))
    with pytest.raises(ValueError):
        AgentSkillFile("SKILL.md", "x", "bad")
    for changes in (
        {"agent_skill_name": "skillforge-other"},
        {"source_skill_version": "v2"},
        {"target": "codex"},
        {"evidence_ids": ()},
        {"evidence_ids": ("a", "extra")},
        {"files": package.files + (package.files[0],)},
    ):
        with pytest.raises(ValueError):
            replace(package, **changes)
    for changes in ({"plan_id": "fake"}, {"target": CodeAgentTarget.CLAUDE_CODE},
                    {"packages": tuple(reversed(plan.packages))},
                    {"packages": plan.packages + plan.packages[:1]}):
        with pytest.raises(ValueError):
            replace(plan, **changes)
