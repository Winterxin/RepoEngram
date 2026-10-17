"""Current-source context must preserve scope, provenance, and export identity."""

import subprocess

import pytest
from test_agent_skill_rendering import route
from test_skill_registry import skill

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.source_representations import source_representations


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "consumer.py").write_text("def consume(row): return row.get('amount')\n")
    (tmp_path / "producer.py").write_text(
        "def produce(measured): return {'amount': measured, 'quantity': measured}\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def scope():
    return ProjectScope(ProjectScopeKind.MODULE, ("consumer",))


def test_related_same_variable_assignment_is_source_anchored(repository):
    text = source_representations(repository, scope())
    assert "producer.py:1" in text and "file SHA256" in text
    assert "'amount', 'quantity'" in text and "'measured'" in text
    assert "not global interchangeability" in text


def test_helper_lookup_arguments_also_locate_producer_representations(repository):
    (repository / "consumer.py").write_text(
        "def consume(profile): return lookup(profile, 'amount', 'fallback')\n"
    )
    assert "'amount', 'quantity'" in source_representations(repository, scope())


def test_constants_dynamic_expressions_overwrites_and_unrelated_keys_not_equivalent(repository):
    (repository / "producer.py").write_text(
        "def produce(x):\n"
        " return [{'amount': 0, 'quantity': 0}, {'amount': f(), 'quantity': f()},\n"
        " {'amount': x, 'quantity': x, 'amount': 0},\n"
        " {'amount': x, 'quantity': x, **extra}, {'other': x, 'different': x},\n"
        " {'amount': x, 'quantity': x, key(): 0},\n"
        " {'amount': x, 'side_effect': mutate(), 'quantity': x}]\n"
    )
    assert source_representations(repository, scope()) == ""


def test_untracked_and_test_sources_are_excluded(repository):
    (repository / "producer.py").write_text("pass\n")
    (repository / "untracked.py").write_text("x={'amount': v, 'quantity': v}\n")
    (repository / "tests").mkdir()
    (repository / "tests/test_data.py").write_text("x={'amount': v, 'quantity': v}\n")
    subprocess.run(["git", "add", "tests"], cwd=repository, check=True)
    assert source_representations(repository, scope()) == ""


def test_context_is_bounded_and_requires_actual_matching_consumer(repository):
    (repository / "producer.py").write_text(
        "\n".join(f"x{i}={{'amount': value, 'alias{i}': value}}" for i in range(50))
    )
    assert source_representations(repository, scope()).count("file SHA256") == 24
    assert source_representations(
        repository, ProjectScope(ProjectScopeKind.MODULE, ("missing",))
    ) == ""
    assert source_representations(
        repository, ProjectScope(ProjectScopeKind.FILE, ("../outside.py",))
    ) == ""


def test_export_context_is_optional_and_changes_canonical_content_identity(repository):
    routed = route((skill("a", scope=scope()),))
    exporter = AgentSkillExporter()
    plain = exporter.plan(routed, CodeAgentTarget.CODEX)
    contextual = exporter.plan(routed, CodeAgentTarget.CODEX, repository_root=repository)
    assert plain.plan_id != contextual.plan_id
    assert plain.packages[0].evidence_ids == contextual.packages[0].evidence_ids
    assert len(contextual.packages[0].files) == 3
    evidence = next(f.content for f in contextual.packages[0].files
                    if f.relative_path == "references/evidence.md")
    assert "producer.py:1" in evidence
    assert evidence.index("producer.py:1") < evidence.index("- Provenance:")
    (repository / "producer.py").write_text(
        "def produce(count): return {'amount': count, 'quantity': count}\n"
    )
    updated = exporter.plan(routed, CodeAgentTarget.CODEX, repository_root=repository)
    assert contextual.plan_id != updated.plan_id
    assert contextual.packages[0].source_skill_version == updated.packages[0].source_skill_version


def test_export_shares_reads_without_changing_context_or_retaining_stale_source(
    repository, monkeypatch,
):
    from pathlib import Path

    from skillforge.integration.source_conditions import source_conditions

    routed = route((skill("a", scope=scope()), skill("b", scope=scope())))
    expected_context = source_representations(repository, scope()) + source_conditions(
        repository, scope(),
    )
    original = Path.read_bytes
    observed = []

    def read(path):
        observed.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    exporter = AgentSkillExporter()
    first = exporter.plan(routed, CodeAgentTarget.CODEX, repository_root=repository)
    assert observed.count("consumer.py") == observed.count("producer.py") == 1
    for package in first.packages:
        evidence = next(f.content for f in package.files
                        if f.relative_path == "references/evidence.md")
        assert evidence.startswith(expected_context)
        assert exporter.package(
            routed, CodeAgentTarget.CODEX, package.source_skill_id, repository_root=repository,
        ) == package
    (repository / "producer.py").write_text(
        "def produce(count): return {'amount': count, 'quantity': count}\n"
    )
    second = exporter.plan(routed, CodeAgentTarget.CODEX, repository_root=repository)
    assert first.plan_id != second.plan_id
