"""Lexical, scope, ordering, isolation and selected Evidence routing tests."""

from dataclasses import replace

import pytest
from test_skill_registry import compilation, evidence, skill

from skillforge.domain.project_dna import ProjectKnowledgeStatus, ProjectScope, ProjectScopeKind
from skillforge.domain.routing import (
    SkillRouteReason,
    SkillRouteResult,
    SkillRoutingRequest,
    match_skill,
    routing_trace,
    task_tokens,
)
from skillforge.repository.skill_router import SkillRouter


def route(skills, request=None):
    from skillforge.repository.skill_registry import SkillRegistryBuilder

    registry = SkillRegistryBuilder().build(compilation(tuple(skills)))
    return registry, SkillRouter().route(
        registry, request or SkillRoutingRequest("repo", "create route")
    )


def scoped(key, kind, selector, task="unrelated task"):
    return skill(key, task=task, scope=ProjectScope(kind, (selector,)))


def test_lexical_normalization_overlap_exact_and_no_synonyms():
    item = skill("a", task="CreateRoute_endpoint")
    registry, result = route((item,), SkillRoutingRequest("repo", "create-route ENDPOINT"))
    match = result.matches[0]
    assert match.matched_tokens == ("create", "endpoint", "route")
    assert match.trigger_token_count == match.matched_token_count == 3
    assert match.trigger_coverage == 1.0
    assert match.exact_task_match
    assert match.match_reasons == (
        SkillRouteReason.TASK_LEXICAL_MATCH,
        SkillRouteReason.TASK_EXACT_MATCH,
        SkillRouteReason.REPOSITORY_SCOPE,
    )
    assert task_tokens("route, CREATE") == ("create", "route")
    related = SkillRouter().route(registry, SkillRoutingRequest("repo", "implement endpoint"))
    assert related.matches[0].matched_tokens == ("endpoint",)
    _, empty = route((skill("a", task="implement endpoint"),),
                     SkillRoutingRequest("repo", "create route"))
    assert empty.matches == empty.evidence == ()
    _, reordered = route((skill("a", task="create route"),),
                         SkillRoutingRequest("repo", "route create"))
    assert not reordered.matches[0].exact_task_match


@pytest.mark.parametrize("kind,field,selector", [
    (ProjectScopeKind.FILE, "current_file", "src/a.py"),
    (ProjectScopeKind.DIRECTORY, "current_directory", "src"),
    (ProjectScopeKind.MODULE, "current_module", "pkg.api"),
    (ProjectScopeKind.FILE_TYPE, "current_file_type", "py"),
    (ProjectScopeKind.DOMAIN, "current_domain", "orders"),
])
def test_explicit_exact_scope_only(kind, field, selector):
    item = scoped("a", kind, selector)
    _, yes = route((item,), SkillRoutingRequest("repo", "create route", **{field: selector}))
    assert yes.matches[0].scope_match
    _, no = route((item,), SkillRoutingRequest("repo", "create route", **{field: "wrong"}))
    assert no.matches == ()
    _, absent = route((item,), SkillRoutingRequest("repo", "create route"))
    assert absent.matches == ()


def test_no_directory_or_file_type_inference_from_file_and_repo_scope_not_eligible():
    items = (scoped("directory", ProjectScopeKind.DIRECTORY, "src"),
             scoped("type", ProjectScopeKind.FILE_TYPE, "py"),
             skill("repo-scope", task="unrelated task"))
    _, result = route(items, SkillRoutingRequest("repo", "create route", current_file="src/a.py"))
    assert result.matches == result.evidence == ()


@pytest.mark.parametrize("kind,field,selector", [
    (ProjectScopeKind.FILE, "current_file", "src/a.py"),
    (ProjectScopeKind.DIRECTORY, "current_directory", "src"),
    (ProjectScopeKind.MODULE, "current_module", "pkg.api"),
    (ProjectScopeKind.FILE_TYPE, "current_file_type", "py"),
    (ProjectScopeKind.DOMAIN, "current_domain", "orders"),
])
def test_explicit_scope_conflict_overrides_even_exact_task(kind, field, selector):
    local = scoped("local", kind, selector, task="create route")
    global_rule = skill("global", task="create route")
    request = SkillRoutingRequest("repo", "create route", **{field: "other"})
    registry, result = route((local, global_rule), request)
    assert tuple(match.skill.skill_id for match in result.matches) == ("global",)
    assert tuple(item.evidence_id for item in result.evidence) == ("global",)
    candidate = next(row for row in routing_trace(registry, result)["candidates"]
                     if row["skill_id"] == "local")
    assert candidate["non_selection_reason"] == "EXPLICIT_SCOPE_MISMATCH"
    assert not candidate["selected"] and candidate["rank"] is None
    # Unknown context retains task discovery; no inferred scope or hidden narrowing.
    _, unknown = route((local,), SkillRoutingRequest("repo", "create route"))
    assert unknown.matches[0].exact_task_match and not unknown.matches[0].scope_match
    _, explicit = route((local,), SkillRoutingRequest("repo", "unrelated task",
                                                     **{field: selector}))
    assert explicit.matches[0].scope_match


def test_explicit_scope_accepts_any_declared_selector():
    item = skill("a", scope=ProjectScope(ProjectScopeKind.MODULE, ("pkg.api", "pkg.other")))
    _, result = route((item,), SkillRoutingRequest("repo", "create route",
                                                  current_module="pkg.other"))
    assert result.matches[0].scope_match


def test_rank_tiers_subranking_and_stable_tie_break():
    items = (
        scoped("c", ProjectScopeKind.FILE, "src/a.py"),
        skill("b", task="create endpoint"),
        scoped("a", ProjectScopeKind.FILE, "src/a.py", task="create route"),
        skill("d", task="create route"),
        skill("e", task="create route"),
    )
    request = SkillRoutingRequest("repo", "create route", current_file="src/a.py")
    _, result = route(items, request)
    assert tuple(match.skill.skill_id for match in result.matches) == ("a", "d", "e", "b", "c")
    _, reverse = route(tuple(reversed(items)), request)
    assert reverse.to_dict()["matches"] == result.to_dict()["matches"]
    assert result.matches[0].skill.source_status == ProjectKnowledgeStatus.CANDIDATE


def test_source_status_and_confidence_do_not_affect_rank():
    candidate = skill("a")
    confirmed = replace(skill("b"), source_status=ProjectKnowledgeStatus.CONFIRMED)
    _, result = route((confirmed, candidate))
    assert tuple(match.skill.skill_id for match in result.matches) == ("a", "b")


def test_coverage_then_matched_count_within_tier():
    items = (skill("low", task="create route extra"),
             skill("high", task="create"),
             skill("two", task="route create extra extra2"))
    _, result = route(items)
    assert tuple(match.skill.skill_id for match in result.matches) == ("high", "low", "two")
    _, tied_coverage = route((skill("one", task="create extra"),
                               skill("two", task="create route extra another")))
    assert tuple(match.skill.skill_id for match in tied_coverage.matches) == ("two", "one")


def test_top_k_evidence_isolation_and_zero_match():
    items = tuple(skill(str(i), task="create route") for i in range(7))
    registry, default = route(items)
    assert len(default.matches) == 5
    assert tuple(item.evidence_id for item in default.evidence) == ("0", "1", "2", "3", "4")
    selected = SkillRouter().route(registry, SkillRoutingRequest("repo", "create route", limit=3))
    assert len(selected.matches) == 3
    assert tuple(item.evidence_id for item in selected.evidence) == ("0", "1", "2")
    assert len(registry.evidence) == 7
    assert "registry" not in selected.to_dict()
    assert len(selected.to_dict()["evidence"]) == 3


def test_decision_trace_covers_zero_one_top_k_ties_and_scope():
    import json

    registry, empty = route((), SkillRoutingRequest("repo", "create route"))
    assert routing_trace(registry, empty)["candidates"] == []
    registry, single = route((skill("only", task="create route"),))
    assert routing_trace(registry, single)["selected_skill_ids"] == ["only"]
    items = (skill("a", task="create route"),
             scoped("b", ProjectScopeKind.FILE, "src/a.py", task="unrelated"),
             *(skill(str(i), task="create route") for i in range(6)))
    request = SkillRoutingRequest("repo", "create route", current_file="src/a.py", limit=3)
    registry, result = route(items, request)
    trace = routing_trace(registry, result)
    assert trace == json.loads(json.dumps(trace))
    assert trace["selected_skill_ids"] == [match.skill.skill_id for match in result.matches]
    assert len(trace["candidate_skill_ids"]) == 8
    assert sum(item["non_selection_reason"] == "BELOW_TOP_K"
               for item in trace["candidates"]) == 5
    scoped_candidate = next(item for item in trace["candidates"] if item["skill_id"] == "b")
    assert scoped_candidate["scope_match"] is True
    assert scoped_candidate["matched_tokens"] == []
    assert trace == routing_trace(registry, SkillRouter().route(registry, request))


@pytest.mark.parametrize("limit", [0, 6, -1, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        SkillRoutingRequest("repo", "task", limit=limit)


def test_request_and_repository_validation():
    registry, _ = route((skill("a"),))
    for kwargs in ({"repository_id": " "}, {"task_description": ""}):
        with pytest.raises(ValueError):
            SkillRoutingRequest(kwargs.get("repository_id", "repo"),
                                kwargs.get("task_description", "task"))
    with pytest.raises(ValueError, match="repository_id"):
        SkillRouter().route(registry, SkillRoutingRequest("other", "create route"))


def test_direct_result_rejects_identity_match_order_duplicates_and_evidence():
    registry, result = route((skill("a"), skill("b")))
    def construct(**changes):
        data = dict(repository_id=result.repository_id, registry_id=result.registry_id,
                    source_snapshot_id=result.source_snapshot_id, request=result.request,
                    matches=result.matches, evidence=result.evidence, registry=registry)
        data.update(changes)
        return SkillRouteResult(**data)

    for changes in (
        {"repository_id": "other"}, {"registry_id": "bad"},
        {"source_snapshot_id": "old"},
        {"matches": result.matches + result.matches},
        {"matches": tuple(reversed(result.matches))},
        {"matches": (result.matches[0],), "evidence": result.evidence},
        {"matches": result.matches, "evidence": (evidence("a"),)},
        {"matches": result.matches, "evidence": result.evidence + result.evidence},
        {"matches": (replace(result.matches[0], matched_tokens=("fake",)),)},
        {"matches": (result.matches[1],), "evidence": (evidence("b"),)},
    ):
        with pytest.raises(ValueError):
            construct(**changes)
    with pytest.raises(ValueError):
        construct(matches=result.matches * 3)
    with pytest.raises(ValueError):
        construct(matches=(match_skill(skill("outside"), result.request),), evidence=())
