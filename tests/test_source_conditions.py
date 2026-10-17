"""Scope and meaning of current condition hints must remain source-bound."""
from pathlib import Path

from test_agent_skill_rendering import route
from test_skill_registry import skill

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.source_conditions import source_conditions


def write(root: Path, code: str):
    (root / "consumer.py").write_text(code, encoding="utf8")
    return ProjectScope(ProjectScopeKind.MODULE, ("consumer",))


def test_distinct_condition_producers_are_not_collapsed(tmp_path):
    scope = write(tmp_path, """def states(data):
    observed = data.get('observation') is not None
    return {'can_review': observed, 'can_project': bool(data.get('projection'))}
def consume(flags):
    return flags.get('can_review'), flags.get('can_project')
""")
    text = source_conditions(tmp_path, scope)
    assert "'can_review'" in text and "'can_project'" in text
    assert "observation" in text and "projection" in text
    assert "consumer.py:" in text and "file SHA256" in text
    assert "not evaluated truth values" in text


def test_ambiguous_binding_and_non_boolean_values_not_promoted(tmp_path):
    scope = write(tmp_path, """def states(data):
    flag = bool(data)
    if data:
        flag = data
    return {'flag': flag, 'value': data.get('value'), 'text': data.get('text') or 'fallback'}
def consume(flags):
    return flags.get('flag'), flags.get('value'), flags.get('text')
""")
    assert source_conditions(tmp_path, scope) == ""


def test_nested_function_binding_is_not_reused_for_outer_dictionary(tmp_path):
    scope = write(tmp_path, """def states(data):
    flag = data
    def inner():
        flag = bool(data)
        return {}
    return {'flag': flag}
def consume(flags):
    return flags.get('flag')
""")
    assert source_conditions(tmp_path, scope) == ""


def test_outside_root_and_oversize_source_are_not_read(tmp_path):
    assert source_conditions(tmp_path, ProjectScope(ProjectScopeKind.FILE, ("../other.py",))) == ""
    scope = write(tmp_path, "#" * 262145)
    assert source_conditions(tmp_path, scope) == ""


def test_condition_context_closes_canonical_package_hashes(tmp_path):
    scope = write(tmp_path, """def states(data):
    return {'enabled': bool(data)}
def consume(flags):
    return flags.get('enabled')
""")
    routed = route((skill("a", scope=scope),))
    exporter = AgentSkillExporter()
    plain = exporter.plan(routed, CodeAgentTarget.CODEX)
    current = exporter.plan(routed, CodeAgentTarget.CODEX, repository_root=tmp_path)
    assert current.plan_id != plain.plan_id
    assert current.packages[0].evidence_ids == plain.packages[0].evidence_ids
    assert len(current.packages[0].files) == 3
    markdown = next(f.content for f in current.packages[0].files
                    if f.relative_path == "references/evidence.md")
    assert "'enabled'" in markdown
