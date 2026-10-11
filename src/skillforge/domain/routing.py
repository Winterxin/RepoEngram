"""Canonical repository Skill snapshots and transparent, deterministic routing contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass
from enum import StrEnum

from skillforge.domain.evidence import Evidence
from skillforge.domain.project_dna import ProjectScopeKind
from skillforge.domain.serialization import to_json_value
from skillforge.domain.skills import PortableProjectSkill, ProjectSkill, RuleGuidanceSkill

REGISTRY_SCHEMA = "skill-registry-v1"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_INTENT_IDENTIFIER = re.compile(r"\b_+[A-Za-z]\w*\b")
# Language, location and generic editing words do not identify a workflow action.
# Keep domain words and exact symbols; this is a lexical signal, not semantic proof.
_INTENT_NOISE = frozenset("""
a an the and or of to in on for with from by at as is are be into that this it its
when while then if only all any each using use used do does should must can may
add change changes update updates fix implement improve ensure preserve retain
keep existing current new requested relevant project repository module modules
file files code test tests testing output text markdown english chinese alongside
instead based non helper implementation context language available result
guard guarded filter filtered supported unsupported claim claims public explicit
make making not without
""".split())


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _evidence_closure(skills: tuple[PortableProjectSkill, ...], evidence: tuple[Evidence, ...],
                      repository_id: str) -> None:
    if not isinstance(evidence, tuple) or any(not isinstance(item, Evidence) for item in evidence):
        raise ValueError("Evidence must be a tuple of Evidence")
    ids = tuple(item.evidence_id for item in evidence)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("Evidence IDs must be sorted and unique")
    if any(item.repository_id != repository_id for item in evidence):
        raise ValueError("Evidence repository_id mismatch")
    expected = tuple(sorted({key for skill in skills for key in skill.evidence_ids}))
    if ids != expected:
        raise ValueError("Evidence must exactly close selected Skill references")


def registry_fingerprint(repository_id: str, source_snapshot_id: str,
                         skills: tuple[PortableProjectSkill, ...]) -> str:
    """Fingerprint only schema, repository, snapshot and ordered Skill versions."""
    payload = (REGISTRY_SCHEMA, repository_id, source_snapshot_id,
               tuple((skill.skill_id, skill.version) for skill in skills))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    repository_id: str
    source_snapshot_id: str
    registry_id: str
    skills: tuple[PortableProjectSkill, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _required(self.repository_id, "repository_id")
        _required(self.source_snapshot_id, "source_snapshot_id")
        if not isinstance(self.skills, tuple) or any(
            not isinstance(skill, (ProjectSkill, RuleGuidanceSkill)) for skill in self.skills
        ):
            raise ValueError("Registry skills must be a tuple of ProjectSkill")
        ids = tuple(skill.skill_id for skill in self.skills)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Registry Skill IDs must be sorted and unique")
        if any(skill.repository_id != self.repository_id or
               skill.source_snapshot_id != self.source_snapshot_id for skill in self.skills):
            raise ValueError("Registry Skill repository/snapshot mismatch")
        _evidence_closure(self.skills, self.evidence, self.repository_id)
        if self.registry_id != registry_fingerprint(
            self.repository_id, self.source_snapshot_id, self.skills
        ):
            raise ValueError("registry_id does not match canonical Registry content")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


@dataclass(frozen=True, slots=True)
class SkillRoutingRequest:
    repository_id: str
    task_description: str
    current_file: str | None = None
    current_directory: str | None = None
    current_module: str | None = None
    current_file_type: str | None = None
    current_domain: str | None = None
    limit: int = 5
    current_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.repository_id, "repository_id")
        _required(self.task_description, "task_description")
        for name in ("current_file", "current_directory", "current_module",
                     "current_file_type", "current_domain"):
            value = getattr(self, name)
            if value is not None:
                _required(value, name)
        if (isinstance(self.limit, bool) or not isinstance(self.limit, int)
                or not 1 <= self.limit <= 5):
            raise ValueError("limit must be an integer from 1 to 5")
        if (not isinstance(self.current_modules, tuple) or len(self.current_modules) > 5
                or any(not isinstance(item, str) or not item.strip()
                       for item in self.current_modules)
                or self.current_modules != tuple(sorted(set(self.current_modules)))):
            raise ValueError("current_modules must be a sorted unique tuple of at most 5 names")
        if self.current_module is not None and self.current_modules:
            raise ValueError("use current_module or current_modules, not both")


class SkillRouteReason(StrEnum):
    TASK_LEXICAL_MATCH = "TASK_LEXICAL_MATCH"
    TASK_EXACT_MATCH = "TASK_EXACT_MATCH"
    FILE_SCOPE_MATCH = "FILE_SCOPE_MATCH"
    DIRECTORY_SCOPE_MATCH = "DIRECTORY_SCOPE_MATCH"
    MODULE_SCOPE_MATCH = "MODULE_SCOPE_MATCH"
    FILE_TYPE_SCOPE_MATCH = "FILE_TYPE_SCOPE_MATCH"
    DOMAIN_SCOPE_MATCH = "DOMAIN_SCOPE_MATCH"
    REPOSITORY_SCOPE = "REPOSITORY_SCOPE"


_SCOPE_CONTEXT = {
    ProjectScopeKind.FILE: ("current_file", SkillRouteReason.FILE_SCOPE_MATCH),
    ProjectScopeKind.DIRECTORY: ("current_directory", SkillRouteReason.DIRECTORY_SCOPE_MATCH),
    ProjectScopeKind.MODULE: ("current_module", SkillRouteReason.MODULE_SCOPE_MATCH),
    ProjectScopeKind.FILE_TYPE: ("current_file_type", SkillRouteReason.FILE_TYPE_SCOPE_MATCH),
    ProjectScopeKind.DOMAIN: ("current_domain", SkillRouteReason.DOMAIN_SCOPE_MATCH),
}


def _lex_words(value: str) -> tuple[str, ...]:
    return tuple(word.casefold() for part in _CAMEL.split(value)
                 for word in _WORD.findall(part))


def task_tokens(value: str) -> tuple[str, ...]:
    """Sorted unique words after punctuation, underscore and identifier splitting."""
    return tuple(sorted(set(_lex_words(value))))


def _scope_conflicts(skill: PortableProjectSkill, request: SkillRoutingRequest) -> bool:
    """An explicit context outside the declared Scope cannot be rescued by words."""
    scope = skill.trigger.scope
    if scope.kind == ProjectScopeKind.REPOSITORY:
        return False
    field, _ = _SCOPE_CONTEXT[scope.kind]
    contexts = _scope_context(request, field)
    return bool(contexts) and not set(contexts).intersection(scope.selectors)


def _scope_context(request: SkillRoutingRequest, field: str) -> tuple[str, ...]:
    if field == "current_module" and request.current_modules:
        return request.current_modules
    value = getattr(request, field)
    return () if value is None else (value,)


def workflow_intent_tokens(skill: PortableProjectSkill) -> tuple[str, ...]:
    """Content of executable actions, excluding generic edits and declared locations."""
    if not isinstance(skill, ProjectSkill) or skill.procedure is None:
        return ()
    location = {word for selector in skill.trigger.scope.selectors
                for word in task_tokens(selector)}
    # Plural mentions of the declared location are still location signals, not
    # evidence that another operation in that module uses this workflow.
    location |= {word + "s" for word in tuple(location) if word.isascii()}
    location |= {word[:-1] + "ies" for word in tuple(location)
                 if word.isascii() and word.endswith("y")}
    content = " ".join((skill.trigger.task_description,
                        *(text for step in skill.procedure.steps
                          for text in (step.instruction, step.condition or ""))))
    return tuple(sorted(_intent_tokens(content) - location))


def _intent_tokens(value: str) -> set[str]:
    # A symbol naming a fallback helper is not evidence that another helper in
    # that module implements the fallback workflow. Match complete identifiers.
    symbols = {symbol.casefold() for symbol in _INTENT_IDENTIFIER.findall(value)}
    words = task_tokens(_INTENT_IDENTIFIER.sub(" ", value))
    return symbols | {word for word in words if len(word) > 2 and word not in _INTENT_NOISE}


def _workflow_intent_conflicts(skill: PortableProjectSkill, request: SkillRoutingRequest) -> bool:
    # Scope-based rule discovery remains intact. Do not discard embedded constraints
    # or risks merely because their hosting workflow action is not lexically matched.
    if (not isinstance(skill, ProjectSkill) or skill.procedure is None
            or skill.constraints or skill.risks
            or _lex_words(request.task_description) == _lex_words(skill.trigger.task_description)):
        return False
    intent = set(workflow_intent_tokens(skill))
    query = _intent_tokens(request.task_description)
    overlap = intent.intersection(query)
    if skill.trigger.scope.kind == ProjectScopeKind.REPOSITORY:
        # Explicit module context describes location, not the requested action.
        # A repository-wide workflow must not qualify solely because it repeats
        # the package/module name. Do not infer context from free task text.
        location = {word for module in _scope_context(request, "current_module")
                    for word in task_tokens(module)}
        overlap -= location
    if not intent or overlap:
        return False
    # English-only lexical knowledge cannot reject a translated task.
    # Unknown cross-language applicability retains scope-based discovery.
    if any(not word.isascii() for word in query) and all(word.isascii() for word in intent):
        return False
    return True


@dataclass(frozen=True, slots=True)
class SkillRouteMatch:
    skill: PortableProjectSkill
    matched_tokens: tuple[str, ...]
    trigger_token_count: int
    matched_token_count: int
    trigger_coverage: float
    exact_task_match: bool
    scope_match: bool
    match_reasons: tuple[SkillRouteReason, ...]

    @property
    def rank_key(self) -> tuple[int, int, float, int, str]:
        has_task = self.matched_token_count > 0
        tier = 0 if has_task and self.scope_match else 1 if has_task else 2
        return (tier, -int(self.exact_task_match), -self.trigger_coverage,
                -self.matched_token_count, self.skill.skill_id)


def match_skill(
    skill: PortableProjectSkill, request: SkillRoutingRequest
) -> SkillRouteMatch | None:
    if _scope_conflicts(skill, request) or _workflow_intent_conflicts(skill, request):
        return None
    task = task_tokens(request.task_description)
    trigger = task_tokens(skill.trigger.task_description)
    matched = tuple(sorted(set(task) & set(trigger)))
    exact = bool(trigger) and _lex_words(request.task_description) == _lex_words(
        skill.trigger.task_description
    )
    # Shared editing/function words are not a task signal for a plain executable
    # workflow. Exact tasks and embedded policy/risk discovery remain intact.
    if (isinstance(skill, ProjectSkill) and skill.procedure is not None
            and not skill.constraints and not skill.risks and not exact):
        matched = tuple(word for word in matched if word not in _INTENT_NOISE)
    scope = skill.trigger.scope
    specific = False
    reasons = []
    if matched:
        reasons.append(SkillRouteReason.TASK_LEXICAL_MATCH)
    if exact:
        reasons.append(SkillRouteReason.TASK_EXACT_MATCH)
    if scope.kind == ProjectScopeKind.REPOSITORY:
        reasons.append(SkillRouteReason.REPOSITORY_SCOPE)
    else:
        field, reason = _SCOPE_CONTEXT[scope.kind]
        specific = bool(set(_scope_context(request, field)).intersection(scope.selectors))
        if specific:
            reasons.append(reason)
    if not matched and not specific:
        return None
    return SkillRouteMatch(skill, matched, len(trigger), len(matched),
                           len(matched) / len(trigger) if trigger else 0.0,
                           exact, specific, tuple(reasons))


@dataclass(frozen=True, slots=True)
class SkillRouteResult:
    repository_id: str
    registry_id: str
    source_snapshot_id: str
    request: SkillRoutingRequest
    matches: tuple[SkillRouteMatch, ...]
    evidence: tuple[Evidence, ...]
    registry: InitVar[SkillRegistry]

    def __post_init__(self, registry: SkillRegistry) -> None:
        if not isinstance(registry, SkillRegistry):
            raise ValueError("Route result requires its Registry for validation")
        if (self.repository_id != registry.repository_id or
            self.registry_id != registry.registry_id or
            self.source_snapshot_id != registry.source_snapshot_id):
            raise ValueError("Route result Registry identity mismatch")
        if (not isinstance(self.request, SkillRoutingRequest)
                or self.request.repository_id != self.repository_id):
            raise ValueError("Route request repository mismatch")
        if not isinstance(self.matches, tuple) or len(self.matches) > self.request.limit or any(
            not isinstance(match, SkillRouteMatch) for match in self.matches
        ):
            raise ValueError("Route matches must be a bounded tuple")
        skills = {skill.skill_id: skill for skill in registry.skills}
        ids = tuple(match.skill.skill_id for match in self.matches)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate selected Skill")
        for match in self.matches:
            if (skills.get(match.skill.skill_id) != match.skill or
                match_skill(match.skill, self.request) != match):
                raise ValueError("Route match is not a valid Registry candidate")
        if self.matches != tuple(sorted(self.matches, key=lambda match: match.rank_key)):
            raise ValueError("Route matches must use canonical rank order")
        eligible = (match_skill(skill, self.request) for skill in registry.skills)
        expected = tuple(sorted((match for match in eligible if match is not None),
                                key=lambda match: match.rank_key)[:self.request.limit])
        if self.matches != expected:
            raise ValueError("Route matches must be the Registry's canonical Top-K")
        _evidence_closure(tuple(match.skill for match in self.matches), self.evidence,
                          self.repository_id)
        registry_evidence = {item.evidence_id: item for item in registry.evidence}
        if any(registry_evidence.get(item.evidence_id) != item for item in self.evidence):
            raise ValueError("Route Evidence differs from Registry Evidence")

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def routing_trace(registry: SkillRegistry, route: SkillRouteResult) -> dict[str, object]:
    """Serialize every deterministic candidate signal without influencing selection."""
    if (route.registry_id != registry.registry_id
            or route.repository_id != registry.repository_id):
        raise ValueError("route and registry identity mismatch")
    ranked = sorted(
        (match for skill in registry.skills if (match := match_skill(skill, route.request))
         is not None), key=lambda match: match.rank_key
    )
    selected = [match.skill.skill_id for match in route.matches]
    if selected != [match.skill.skill_id for match in ranked[:route.request.limit]]:
        raise ValueError("trace selection differs from canonical route")
    rank = {match.skill.skill_id: index + 1 for index, match in enumerate(ranked)}
    matches = {match.skill.skill_id: match for match in ranked}
    candidates = []
    for skill in registry.skills:
        match = matches.get(skill.skill_id)
        candidates.append({
            "skill_id": skill.skill_id,
            "trigger_tokens": list(task_tokens(skill.trigger.task_description)),
            "workflow_intent_tokens": list(workflow_intent_tokens(skill)),
            "scope_kind": skill.trigger.scope.kind.value,
            "scope_selectors": list(skill.trigger.scope.selectors),
            "matched_tokens": list(match.matched_tokens) if match else [],
            "exact_task_match": match.exact_task_match if match else False,
            "scope_match": match.scope_match if match else False,
            "match_reasons": [reason.value for reason in match.match_reasons] if match else [],
            "trigger_coverage": match.trigger_coverage if match else 0.0,
            "rank": rank.get(skill.skill_id),
            "rank_key": list(match.rank_key) if match else None,
            "selected": skill.skill_id in selected,
            "non_selection_reason": (
                None if skill.skill_id in selected else
                "EXPLICIT_SCOPE_MISMATCH" if _scope_conflicts(skill, route.request) else
                "WORKFLOW_INTENT_MISMATCH" if _workflow_intent_conflicts(skill, route.request) else
                "NO_MATCH" if match is None else "BELOW_TOP_K"
            ),
        })
    return {
        "schema": "skillforge-router-decision-trace-v1",
        "algorithm": "lexical-workflow-intent-top-k-v6",
        "repository_id": registry.repository_id,
        "registry_id": registry.registry_id,
        "source_snapshot_id": registry.source_snapshot_id,
        "request": to_json_value(route.request),
        "task_tokens": list(task_tokens(route.request.task_description)),
        "candidate_skill_ids": [skill.skill_id for skill in registry.skills],
        "ranked_skill_ids": [match.skill.skill_id for match in ranked],
        "selected_skill_ids": selected,
        "candidates": candidates,
    }
