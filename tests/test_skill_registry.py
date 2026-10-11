"""Canonical Registry integration and direct-construction tests."""

from dataclasses import replace

import pytest

from skillforge.domain.evidence import Confidence, Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.project_dna import ProjectKnowledgeStatus, ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRegistry, registry_fingerprint
from skillforge.domain.skills import (
    ProjectSkill,
    SkillCompilationResult,
    SkillSkipReason,
    SkillTrigger,
    SkippedWorkflow,
)
from skillforge.repository.skill_registry import SkillRegistryBuilder


def evidence(key: str, repo: str = "repo") -> Evidence:
    return Evidence(key, repo, EvidenceType.GIT, SourceLocation("src/a.py"), key, "fixture")


def skill(key: str, *, repo: str = "repo", snapshot: str = "snap", version: str = "v1",
          task: str = "create route", scope: ProjectScope | None = None) -> ProjectSkill:
    return ProjectSkill(
        key, repo, snapshot, version, key,
        SkillTrigger(task, scope or ProjectScope(ProjectScopeKind.REPOSITORY), (key,)),
        (), (), key, ProjectKnowledgeStatus.CANDIDATE, Support(1, 1),
        Confidence(0.5, "fixture"), (key,), (), (), (), (), (), (key,),
    )


def compilation(skills: tuple[ProjectSkill, ...]) -> SkillCompilationResult:
    return SkillCompilationResult(
        "repo", "snap", skills,
        (SkippedWorkflow("skip-a", SkillSkipReason.DEPRECATED),
         SkippedWorkflow("skip-b", SkillSkipReason.UNKNOWN_SCOPE)),
        tuple(evidence(key) for key in sorted(
            {key for item in skills for key in item.evidence_ids}
        )),
    )


def registry(*keys: str) -> SkillRegistry:
    return SkillRegistryBuilder().build(compilation(tuple(skill(key) for key in keys)))


def test_builder_only_includes_compiled_skills_and_exact_evidence():
    result = registry("a", "b", "c")
    assert tuple(item.skill_id for item in result.skills) == ("a", "b", "c")
    assert tuple(item.evidence_id for item in result.evidence) == ("a", "b", "c")
    assert not hasattr(result, "skipped_workflows")


def test_actual_compiler_to_registry_integration_with_three_skills_and_two_skips():
    from test_skill_compiler import dna, workflow

    from skillforge.repository.skill_compiler import SkillCompiler

    def named(key, status=ProjectKnowledgeStatus.CANDIDATE):
        source = workflow(status)
        return replace(source, workflow_id=key, draft=replace(source.draft, draft_id=key))

    workflows = tuple(named(f"wf-{index}") for index in range(3))
    workflows += (named("wf-skip-a", ProjectKnowledgeStatus.DEPRECATED),
                  named("wf-skip-b", ProjectKnowledgeStatus.CONFLICTED))
    compiled = SkillCompiler().compile(dna(workflows=workflows))
    result = SkillRegistryBuilder().build(compiled)
    assert len(compiled.skills) == len(result.skills) == 3
    assert len(compiled.skipped_workflows) == 2
    assert tuple(item.evidence_id for item in result.evidence) == tuple(
        item.evidence_id for item in compiled.evidence
    )


def test_registry_is_order_independent_and_version_sensitive():
    first = SkillRegistryBuilder().build(compilation((skill("b"), skill("a"))))
    second = SkillRegistryBuilder().build(compilation((skill("a"), skill("b"))))
    assert first.to_dict() == second.to_dict()
    assert first.registry_id == second.registry_id
    changed = SkillRegistryBuilder().build(compilation((skill("b", version="v2"), skill("a"))))
    assert changed.registry_id != first.registry_id


@pytest.mark.parametrize("skills,evidence_items", [
    ((skill("a"), skill("a")), (evidence("a"),)),
    ((skill("a", repo="other"),), (evidence("a"),)),
    ((skill("a", snapshot="old"),), (evidence("a"),)),
    ((skill("a"),), ()),
    ((skill("a"),), (evidence("a"), evidence("z"))),
    ((skill("a"),), (evidence("a"), evidence("a"))),
    ((skill("a"),), (evidence("a", "other"),)),
    ((skill("b"), skill("a")), (evidence("a"), evidence("b"))),
    ((skill("a"), skill("b")), (evidence("b"), evidence("a"))),
])
def test_direct_registry_rejects_noncanonical_content(skills, evidence_items):
    identity = registry_fingerprint("repo", "snap", skills)
    with pytest.raises(ValueError):
        SkillRegistry("repo", "snap", identity, skills, evidence_items)


def test_registry_rejects_conflicting_evidence_id_and_wrong_fingerprint():
    original = evidence("a")
    conflict = replace(original, description="different")
    with pytest.raises(ValueError):
        SkillRegistry("repo", "snap", registry_fingerprint("repo", "snap", (skill("a"),)),
                      (skill("a"),), (original, conflict))
    with pytest.raises(ValueError, match="registry_id"):
        replace(registry("a"), registry_id="bad")


def test_builder_requires_compilation_result():
    with pytest.raises(TypeError):
        SkillRegistryBuilder().build(object())
