"""Additional project-rule searches are consumable within the current task."""

import subprocess
import sys

import pytest
from test_incremental_evolution_v2 import full, models
from test_incremental_evolution_v2 import rich_repo as rich_repo

from skillforge.integration.autonomous import install
from skillforge.domain.skills import ProjectSkill
from skillforge.integration.mcp_server import ProjectTools
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.verify import VerifyPlanBuilder


@pytest.fixture
def service(rich_repo):
    prepared = full(rich_repo, models())
    baseline = load_baseline(rich_repo, prepared.evolution_baseline_path)
    install(rich_repo, baseline.source_dna, baseline.source_compilation, baseline.head_commit)
    return ProjectTools(rich_repo)


def test_read_extra_search_skill_and_preserve_primary_obligations(service, monkeypatch):
    primary = service.call(
        "get_related_workflow",
        {"task": "unrelated", "scope": {"current_module": "app.api0.route0"}},
    )
    assert primary["skills"]
    for item in primary["skills"]:
        service.call("get_skill", {"skill_id": item["skill_id"]})
    expected = VerifyPlanBuilder().build(service.route)
    search = service.call(
        "search_project_rules",
        {"query": "unrelated", "scope": {"current_module": "app.api1.route0"}},
    )
    ids = {item["skill_id"] for item in primary["skills"]}
    extra = [item["skill_id"] for item in search["skills"] if item["skill_id"] not in ids]
    assert extra
    read = service.call("get_skill", {"skill_id": extra[0]})
    assert read["files"]["SKILL.md"]
    observed = []

    class Verifier:
        def verify(self, root, base, plan):
            observed.append(plan)
            raise RuntimeError("inspection only")

    monkeypatch.setattr("skillforge.integration.mcp_server.IndependentVerifier", Verifier)
    with pytest.raises(RuntimeError, match="inspection"):
        service.call("verify_patch", {})
    assert set(expected.obligations) <= set(observed[0].obligations)
    assert extra[0] in {key for key, _ in observed[0].selected_skills}
    assert ids <= {key for key, _ in observed[0].selected_skills}


def test_read_one_of_multiple_search_results_consumes_only_that_skill(service, monkeypatch):
    primary = service.call("get_related_workflow", {
        "task": "unrelated", "scope": {"current_module": "app.api0.route0"},
    })
    primary_ids = {item["skill_id"] for item in primary["skills"]}
    for skill_id in primary_ids:
        service.call("get_skill", {"skill_id": skill_id})
    seed = VerifyPlanBuilder().build(service.route)
    # Query genuine registered triggers to obtain a canonical multi-result route.
    query = " ".join(skill.trigger.task_description for skill in service.registry.skills
                     if isinstance(skill, ProjectSkill))
    search = service.call("search_project_rules", {"query": query})
    extras = [item["skill_id"] for item in search["skills"]
              if item["skill_id"] not in primary_ids]
    assert len(extras) >= 2
    selected = extras[0]
    observed = []

    class Verifier:
        def verify(self, root, base, plan):
            observed.append(plan)
            raise RuntimeError("inspection only")

    monkeypatch.setattr("skillforge.integration.mcp_server.IndependentVerifier", Verifier)
    # A real multi-result search alone grants access, not new task obligations.
    with pytest.raises(RuntimeError, match="inspection"):
        service.call("verify_patch", {})
    assert set(seed.obligations) == set(observed[-1].obligations)
    assert {key for key, _ in observed[-1].selected_skills} == primary_ids
    observed.clear()
    assert service.call("get_skill", {"skill_id": selected})["files"]["SKILL.md"]
    with pytest.raises(RuntimeError, match="inspection"):
        service.call("verify_patch", {})
    assert {key for key, _ in observed[0].selected_skills} == primary_ids | {selected}
    assert set(seed.obligations) <= set(observed[0].obligations)
    assert not (set(extras[1:]) & {o.target_ref.skill_id for o in observed[0].obligations})
    assert selected in {o.target_ref.skill_id for o in observed[0].obligations}
    # Selection remains a valid canonical route; no router result is fabricated.
    route = service.consumed_routes[selected]
    assert len(route.matches) >= 2
    builder = VerifyPlanBuilder()
    with pytest.raises(ValueError, match="not selected"):
        builder.for_skills(route, ("foreign",))
    with pytest.raises(ValueError, match="unique"):
        builder.for_skills(route, (selected, selected))


def test_search_does_not_grant_foreign_access_or_replace_task(service):
    with pytest.raises(ValueError, match="before"):
        service.call("get_skill", {"skill_id": "foreign"})
    service.call("get_related_workflow", {"task": "unrelated"})
    service.call(
        "search_project_rules",
        {"query": "unrelated", "scope": {"current_module": "app.api1.route0"}},
    )
    with pytest.raises(ValueError, match="not selected"):
        service.call("get_skill", {"skill_id": "foreign"})
    service.call("get_related_workflow", {"task": "a different unrelated task"})
    assert not service.search_routes and not service.consumed_routes


def test_previously_selected_skill_cannot_be_read_after_head_changes(service):
    primary = service.call(
        "get_related_workflow",
        {"task": "unrelated", "scope": {"current_module": "app.api1.route0"}},
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "advance HEAD"],
        cwd=service.root,
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="stale"):
        service.call("get_skill", {"skill_id": primary["skills"][0]["skill_id"]})


@pytest.mark.parametrize("tool_name", ("get_evidence", "verify_patch"))
def test_consumed_knowledge_cannot_verify_or_supply_evidence_after_head_changes(service, tool_name):
    primary = service.call("get_related_workflow", {
        "task": "unrelated", "scope": {"current_module": "app.api1.route0"}})
    for item in primary["skills"]:
        service.call("get_skill", {"skill_id": item["skill_id"]})
    skill_id = primary["skills"][0]["skill_id"]
    assert service.call("get_evidence", {"rule_id": skill_id})
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "advance HEAD"],
                   cwd=service.root, check=True, capture_output=True)
    arguments = {"rule_id": skill_id} if tool_name == "get_evidence" else {}
    with pytest.raises(ValueError, match="stale"):
        service.call(tool_name, arguments)


@pytest.mark.parametrize("multiple_results", (False, True))
def test_rule_search_is_readable_through_real_stdio(service, multiple_results):
    import json
    import os

    search = ({"query": " ".join(skill.trigger.task_description
                                  for skill in service.registry.skills
                                  if isinstance(skill, ProjectSkill))}
              if multiple_results else
              {"query": "unrelated", "scope": {"current_module": "app.api1.route0"}})

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_related_workflow", "arguments": {"task": "unrelated"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_project_rules",
                "arguments": search,
            },
        },
    ]
    # A deterministic client reads the actual routed ID, never an injected ID.
    process = subprocess.Popen(
        [sys.executable, "-m", "skillforge.integration.mcp_server", "--repo", str(service.root)],
        cwd=service.root,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        responses = []
        for request in requests:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            responses.append(json.loads(process.stdout.readline()))
        searched = json.loads(responses[-1]["result"]["content"][0]["text"])
        if multiple_results:
            assert len(searched["skills"]) >= 2
        skill_id = searched["skills"][0]["skill_id"]
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "get_skill", "arguments": {"skill_id": skill_id}},
                }
            )
            + "\n"
        )
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["result"]["isError"] is False
    finally:
        process.communicate(timeout=10)


def test_combining_plans_preserves_nonempty_behavior_test_obligations(service):
    service.call(
        "get_related_workflow",
        {"task": "unrelated", "scope": {"current_module": "nebula.entry"}},
    )
    builder = VerifyPlanBuilder()
    seed = builder.build(service.route)
    assert seed.obligations
    combined = builder.combine((seed, seed))
    assert combined == seed
    foreign = builder._create_plan(
        seed.repository_id,
        "foreign-registry",
        seed.source_snapshot_id,
        seed.selected_skills,
        seed.obligations,
        (),
        seed.evidence,
    )
    with pytest.raises(ValueError, match="different knowledge snapshots"):
        builder.combine((seed, foreign))
