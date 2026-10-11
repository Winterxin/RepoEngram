"""Route compiled Skills with lexical task and explicit exact-scope signals only."""

from skillforge.domain.routing import (
    SkillRegistry,
    SkillRouteResult,
    SkillRoutingRequest,
    match_skill,
)


class SkillRouter:
    def route(self, registry: SkillRegistry, request: SkillRoutingRequest) -> SkillRouteResult:
        if not isinstance(registry, SkillRegistry):
            raise TypeError("SkillRouter requires SkillRegistry")
        if not isinstance(request, SkillRoutingRequest):
            raise TypeError("SkillRouter requires SkillRoutingRequest")
        if request.repository_id != registry.repository_id:
            raise ValueError("Routing repository_id mismatch")
        candidates = (match_skill(skill, request) for skill in registry.skills)
        matches = tuple(sorted((match for match in candidates if match is not None),
                               key=lambda match: match.rank_key)[:request.limit])
        selected = {key for match in matches for key in match.skill.evidence_ids}
        evidence = tuple(item for item in registry.evidence if item.evidence_id in selected)
        return SkillRouteResult(registry.repository_id, registry.registry_id,
                                registry.source_snapshot_id, request, matches, evidence, registry)
