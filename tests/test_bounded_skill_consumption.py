"""Exact multi-module discovery and bounded session reads; no model fixtures."""

import json

import pytest
from test_autonomous_mcp import installed  # noqa: F401
from test_prepare_evolution_baseline import prepared, repository  # noqa: F401
from test_skill_router import route, scoped

from skillforge.domain.project_dna import ProjectScopeKind
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.integration.mcp_server import ProjectTools


def test_modules_are_exact_alternatives_with_one_global_top_k():
    items = tuple(scoped(str(i), ProjectScopeKind.MODULE, f"pkg.unit{i}")
                  for i in range(6))
    request = SkillRoutingRequest("repo", "create route", limit=3,
                                  current_modules=("pkg.unit0", "pkg.unit2", "pkg.unit4"))
    _, result = route(items, request)
    assert {m.skill.skill_id for m in result.matches} == {"0", "2", "4"}
    assert all(m.scope_match for m in result.matches)
    assert result.request.current_modules == request.current_modules


def test_multi_scope_does_not_infer_modules_or_rescue_outside_scope():
    item = scoped("a", ProjectScopeKind.MODULE, "pkg.unit", task="create route")
    _, wrong = route((item,), SkillRoutingRequest("repo", "create route",
                                                current_modules=("pkg", "pkg.unit.child")))
    assert not wrong.matches


@pytest.mark.parametrize("modules", [[], ["pkg.a"], ("",), ("pkg.a", "pkg.a"),
                                     ("pkg.b", "pkg.a"), tuple(f"pkg.u{i}" for i in range(6))])
def test_domain_multi_scope_requires_bounded_canonical_tuple(modules):
    with pytest.raises(ValueError, match="current_modules"):
        SkillRoutingRequest("repo", "task", current_modules=modules)


def test_singular_and_plural_cannot_conflict():
    with pytest.raises(ValueError, match="current_module"):
        SkillRoutingRequest("repo", "task", current_module="pkg.a",
                            current_modules=("pkg.b",))


def test_mcp_rejects_joined_module_without_replacing_active_task(installed):  # noqa: F811
    root, _, _ = installed
    service = ProjectTools(root)
    service.call("get_related_workflow", {"task": "Implement a route"})
    original = service.route
    with pytest.raises(ValueError, match="current_modules"):
        service.call("get_related_workflow", {"task": "Implement a route", "scope": {
            "current_module": "app.api.route0 and app.api.route1"}})
    assert service.route is original


def test_mcp_canonicalizes_modules_and_repeat_reads_are_compact(installed):  # noqa: F811
    root, _, _ = installed
    service = ProjectTools(root)
    args = {"task": "Implement a route", "scope": {
        "current_modules": ["app.api.route1", "app.api.route0"]}}
    related = service.call("get_related_workflow", args)
    assert related["skills"]
    assert service.route.request.current_modules == ("app.api.route0", "app.api.route1")
    selected = related["skills"][0]["skill_id"]
    first = service.call("get_skill", {"skill_id": selected})
    assert first["files"]["SKILL.md"] and not first["already_read"]
    service.call("get_related_workflow", args)
    assert selected in service.read_ids
    again = service.call("get_skill", {"skill_id": selected})
    assert again["already_read"] and again["files"] == {}
    assert again["version"] == first["version"]
    assert len(json.dumps(again)) < len(json.dumps(first))
    refreshed = service.call("get_skill", {"skill_id": selected, "refresh": True})
    assert refreshed["files"] == first["files"]
    service.call("get_related_workflow", {"task": "Different task"})
    assert not service.read_ids


@pytest.mark.parametrize("modules", [[], "pkg.a", [""], [1], ["pkg.a", "pkg.a"],
                                     [f"pkg.u{i}" for i in range(6)]])
def test_mcp_rejects_invalid_plural_scope_before_routing(installed, modules):  # noqa: F811
    root, _, _ = installed
    with pytest.raises(ValueError, match="current_modules"):
        ProjectTools(root).call("get_related_workflow", {
            "task": "Implement a route", "scope": {"current_modules": modules}})
