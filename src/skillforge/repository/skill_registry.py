"""Build an immutable, repository-local Registry from compiled Skills only."""

from skillforge.domain.routing import SkillRegistry, registry_fingerprint
from skillforge.domain.skills import SkillCompilationResult


class SkillRegistryBuilder:
    def build(self, compilation: SkillCompilationResult) -> SkillRegistry:
        if not isinstance(compilation, SkillCompilationResult):
            raise TypeError("SkillRegistryBuilder requires SkillCompilationResult")
        skills = tuple(sorted(compilation.skills, key=lambda skill: skill.skill_id))
        evidence = tuple(sorted(compilation.evidence, key=lambda item: item.evidence_id))
        return SkillRegistry(
            compilation.repository_id,
            compilation.source_snapshot_id,
            registry_fingerprint(compilation.repository_id, compilation.source_snapshot_id, skills),
            skills,
            evidence,
        )
