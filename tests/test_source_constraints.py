"""Opt-in current-source observations stay bounded and separate from routed policy."""

import json

from test_skill_registry import compilation, skill

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.integration import AgentSkillExporter
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def _route(*, matches=False):
    skills = (skill("history", task="create route"),) if matches else ()
    registry = SkillRegistryBuilder().build(compilation(skills))
    return SkillRouter().route(
        registry, SkillRoutingRequest("repo", "create route" if matches else "unrelated task")
    )


def _scope(path="src/example.py"):
    return ProjectScope(ProjectScopeKind.FILE, (path,))


def _export(root, route=None, *, scope=None):
    return AgentSkillExporter().current_source_fallback(
        route or _route(), CodeAgentTarget.CODEX, repository_root=root,
        scope=scope or _scope(), enabled=True,
    )


def _files(plan):
    return {item.relative_path: item.content for item in plan.files}


def test_default_off_and_selected_history_route_do_not_touch_filesystem(tmp_path, monkeypatch):
    # A nonexistent root makes even resolving/reading it an error if the fast path regresses.
    missing_root = tmp_path / "does-not-exist"

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled or history route inspected the filesystem")

    monkeypatch.setattr("pathlib.Path.resolve", forbidden)
    assert AgentSkillExporter().current_source_fallback(
        _route(), CodeAgentTarget.CODEX, repository_root=missing_root, scope=_scope()
    ) is None
    assert AgentSkillExporter().current_source_fallback(
        _route(matches=True), CodeAgentTarget.CODEX, repository_root=missing_root,
        scope=_scope(), enabled=True,
    ) is None


def test_branch_predicates_effects_and_zero_semantics_are_observations(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text(
        "def normalize(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    elif value == 0:\n"
        "        return 0\n"
        "    else:\n"
        "        return value\n",
        encoding="utf-8",
    )

    plan = _export(tmp_path)
    markdown = _files(plan)["SKILL.md"]
    assert "if value is None" in markdown
    assert "elif (if value is None was false)" in markdown
    assert "value == 0" in markdown and "return 0" in markdown
    assert "else (if value == 0 was false)" in markdown
    assert "unconditional" not in markdown
    assert "Predicates are shown as source expressions and were not evaluated." in markdown


def test_ast_observation_does_not_execute_module_side_effects(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "def run():\n    return 0\n",
        encoding="utf-8",
    )

    plan = _export(tmp_path)
    assert plan is not None
    assert not marker.exists()
    assert "return 0" in _files(plan)["SKILL.md"]


def test_sensitive_and_test_eval_paths_are_excluded(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "evals").mkdir()
    (tmp_path / "src/safe.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (tmp_path / "src/api_token.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests/test_hidden.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    (tmp_path / "evals/answer.py").write_text("def f():\n    return 3\n", encoding="utf-8")

    scope = ProjectScope(ProjectScopeKind.DIRECTORY, ("src",))
    markdown = _files(_export(tmp_path, scope=scope))["SKILL.md"]
    assert "safe.py" in markdown
    assert "api_token.py" not in markdown
    assert "tests/" not in markdown and "evals/" not in markdown
    assert "return 2" not in markdown and "return 3" not in markdown


def test_scope_cannot_escape_repository_root(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("def f():\n    return 'outside'\n", encoding="utf-8")
    scope = ProjectScope(ProjectScopeKind.FILE, ("../" + outside.name,))
    assert _export(tmp_path, scope=scope) is None


def test_symlinked_source_is_not_followed(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("def f():\n    return 'outside'\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    link = src / "linked.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return

    plan = _export(tmp_path, scope=ProjectScope(ProjectScopeKind.DIRECTORY, ("src",)))
    assert plan is None


def test_output_and_input_bounds_are_enforced(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for index in range(24):
        (src / f"m{index:02}.py").write_text(
            f"def f{index}():\n    return {index}\n", encoding="utf-8"
        )

    plan = _export(tmp_path, scope=ProjectScope(ProjectScopeKind.DIRECTORY, ("src",)))
    marker = json.loads(_files(plan)[".skillforge.json"])
    markdown = _files(plan)["SKILL.md"]
    evidence = _files(plan)["references/evidence.md"]
    assert len(marker["file_hashes"]) <= 16
    assert len(markdown.encode("utf-8")) < 32 * 1024
    assert len(evidence.encode("utf-8")) < 2 * 1024
    assert len([line for line in markdown.splitlines() if line.startswith("- ")]) <= 49


def test_output_is_deterministic_and_hash_tracks_current_source(tmp_path):
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("def f():\n    return 0\n", encoding="utf-8")
    first = _export(tmp_path)
    again = _export(tmp_path)
    assert first == again
    first_files = _files(first)
    first_marker = json.loads(first_files[".skillforge.json"])
    assert first_marker["source_kind"] == "CURRENT_SOURCE_OBSERVATION"
    assert first_marker["schema"] == "skillforge-current-source-observation-v1"
    assert "source_skill_id" not in first_marker
    assert "historical workflow" in first_files["SKILL.md"]

    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    changed = _export(tmp_path)
    changed_marker = json.loads(_files(changed)[".skillforge.json"])
    assert changed_marker["file_hashes"] != first_marker["file_hashes"]
    assert changed.plan_id != first.plan_id
    assert "return 1" in _files(changed)["SKILL.md"]
