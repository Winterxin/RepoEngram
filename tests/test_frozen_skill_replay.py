"""File-only replay preparation checks using a real, clean Git source fixture."""

import hashlib
import json
import subprocess

import pytest
from test_skill_registry import compilation, skill

from evals.frozen_skill_replay import ReplayPreparationError, prepare_frozen_skill_replay
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.integration import AgentSkillExporter
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def _source_repo(root):
    root.mkdir(parents=True)
    (root / "sample").mkdir()
    (root / "sample" / "flags.py").write_bytes(
        b"def enabled(flag: bool):\n"
        b"    values = {'enabled': bool(flag)}\n"
        b"    if values.get('enabled'):\n"
        b"        return True\n"
        b"    return False\n"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.test"],
                   check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "sample/flags.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _route(*, matched=True):
    scope = ProjectScope(ProjectScopeKind.MODULE, ("sample.flags",))
    skills = (skill("source-route", task="create route", scope=scope),) if matched else ()
    registry = SkillRegistryBuilder().build(compilation(skills))
    request = SkillRoutingRequest(
        "repo", "create route" if matched else "unrelated request",
        current_module="sample.flags",
    )
    return SkillRouter().route(registry, request)


def _make_reference(package, directory, *, wrong_identity=False):
    directory.mkdir(parents=True)
    for item in package.files:
        path = directory / item.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.content.encode("utf-8"))
    marker_path = directory / ".skillforge.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if wrong_identity:
        marker["source_skill_id"] = "other-skill"
    skill_path = directory / "SKILL.md"
    body = skill_path.read_text(encoding="utf-8")
    body += "\n# Shared execution protocol\nPrevious transport protocol wording.\n"
    skill_path.write_bytes(body.encode("utf-8"))
    marker["file_hashes"]["SKILL.md"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_prepares_exact_utf8_package_with_source_conditions(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    output = tmp_path / "bundle"
    manifest = prepare_frozen_skill_replay(
        _route(), source, head, output, trace_reference="trace.jsonl",
    )

    assert manifest["delivery"] == "file"
    assert manifest["source_head"] == head
    trace_path = source / manifest["trace_reference"]
    assert trace_path.is_file()
    assert manifest["caller_trace_reference"] == "trace.jsonl"
    package = manifest["packages"][0]
    package_root = source / package["installed_at"]
    evidence = (package_root / "references" / "evidence.md").read_bytes()
    assert b"Current source conditions" in evidence
    assert b"bool(flag)" in evidence
    assert b"\r\n" not in evidence
    for row in package["files"]:
        raw = (package_root / row["path"]).read_bytes()
        assert len(raw) == row["utf8_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (output / "replay-manifest.json").is_file()


def test_reference_must_exist_and_match_before_knowledge_is_written(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    output = tmp_path / "bundle"
    with pytest.raises(ReplayPreparationError, match="REFERENCE_PACKAGE_MISSING"):
        prepare_frozen_skill_replay(_route(), source, head, output,
                                    reference_package=tmp_path / "missing")
    assert not (source / ".runtime" / "knowledge").exists()
    reason = json.loads((output / "replay-manifest.json").read_text(encoding="utf-8"))
    assert reason == {"schema": 1, "status": "REJECTED", "reason": "REFERENCE_PACKAGE_MISSING"}


def test_reference_allows_shared_protocol_edit_with_frozen_identity_and_evidence(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    route = _route()
    package = AgentSkillExporter().plan(
        route, CodeAgentTarget.CODEX, repository_root=source,
    ).packages[0]
    reference = tmp_path / "reference"
    _make_reference(package, reference)
    manifest = prepare_frozen_skill_replay(
        route, source, head, tmp_path / "bundle", reference_package=reference,
    )
    assert manifest["reference_comparison"] == "MATCHED"


def test_windows_reference_newlines_do_not_change_project_evidence(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    route = _route()
    package = AgentSkillExporter().plan(
        route, CodeAgentTarget.CODEX, repository_root=source,
    ).packages[0]
    reference = tmp_path / "reference"
    _make_reference(package, reference)
    for relative in ("SKILL.md", "references/evidence.md"):
        path = reference / relative
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    manifest = prepare_frozen_skill_replay(
        route, source, head, tmp_path / "bundle", reference_package=reference,
    )
    assert manifest["reference_comparison"] == "MATCHED"


def test_omitted_source_conditions_stop_replay_before_actor_knowledge_is_written(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    route = _route()
    # Reproduce the prior omission: export without repository_root.
    incomplete = AgentSkillExporter().plan(route, CodeAgentTarget.CODEX).packages[0]
    reference = tmp_path / "reference"
    _make_reference(incomplete, reference)
    with pytest.raises(ReplayPreparationError, match="REFERENCE_EVIDENCE_MISMATCH"):
        prepare_frozen_skill_replay(route, source, head, tmp_path / "bundle",
                                    reference_package=reference)
    assert not (source / ".runtime" / "knowledge").exists()


def test_wrong_head_and_no_match_fail_without_model_or_knowledge(tmp_path, monkeypatch):
    source = tmp_path / "source"
    head = _source_repo(source)

    def forbidden_model(*args, **kwargs):
        raise AssertionError("replay preparation must not invoke a model")

    monkeypatch.setattr("skillforge.integration.agent_skills.plan_current_source_rules",
                        forbidden_model)
    with pytest.raises(ReplayPreparationError, match="SOURCE_HEAD_MISMATCH"):
        prepare_frozen_skill_replay(_route(), source, "0" * 40, tmp_path / "wrong-head")
    with pytest.raises(ReplayPreparationError, match="NO_ROUTED_KNOWLEDGE"):
        prepare_frozen_skill_replay(_route(matched=False), source, head, tmp_path / "no-match")
    assert not (source / ".runtime" / "knowledge").exists()
    for bundle in (tmp_path / "wrong-head", tmp_path / "no-match"):
        report = json.loads((bundle / "replay-manifest.json").read_text(encoding="utf-8"))
        assert report["status"] == "REJECTED"
        assert "task" not in json.dumps(report).lower()


def test_nonempty_bundle_and_reference_identity_mismatch_are_non_destructive(tmp_path):
    source = tmp_path / "source"
    head = _source_repo(source)
    nonempty = tmp_path / "occupied"
    nonempty.mkdir()
    sentinel = nonempty / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ReplayPreparationError, match="OUTPUT_BUNDLE_NOT_EMPTY"):
        prepare_frozen_skill_replay(_route(), source, head, nonempty)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    package = AgentSkillExporter().plan(
        _route(), CodeAgentTarget.CODEX, repository_root=source,
    ).packages[0]
    reference = tmp_path / "bad-reference"
    _make_reference(package, reference, wrong_identity=True)
    with pytest.raises(ReplayPreparationError, match="REFERENCE_IDENTITY_MISMATCH"):
        prepare_frozen_skill_replay(_route(), source, head, tmp_path / "bad-ref-bundle",
                                    reference_package=reference)
    assert not (source / ".runtime" / "knowledge").exists()
