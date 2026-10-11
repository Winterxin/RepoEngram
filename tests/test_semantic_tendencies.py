"""Generic history convention adapter, gate, selection and advisory compilation."""

from dataclasses import replace

import pytest
from test_structural_roles import _repo

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.semantic_tendency import SemanticTendencyResponse, TendencyDecisionKind
from skillforge.integration.agent_skills import _render_guidance
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_evaluation import (
    select_semantic_candidates,
    semantic_selection_report,
)
from skillforge.repository.semantic_tendencies import (
    SemanticTendencyEvaluator,
    build_semantic_tendency_request,
    validate_semantic_tendency_response,
)
from skillforge.repository.semantic_tendency_prompt import build_semantic_tendency_prompt
from skillforge.repository.semantic_workflow_prompt import build_semantic_workflow_prompt
from skillforge.repository.skill_compiler import SkillCompiler


@pytest.fixture
def inputs(tmp_path):
    return _repo(tmp_path, [(f"nebula/plugins/unit_{i}.py",) for i in range(3)],
                 initial=("nebula/__init__.py", "nebula/plugins/__init__.py"))


def request_for(inputs):
    _, _, history, facts, _, candidates = inputs
    candidate, = candidates.candidates
    return build_semantic_tendency_request(history, facts, candidates, candidate)


def response_for(request, decision=TendencyDecisionKind.SUPPORTED_TENDENCY):
    return SemanticTendencyResponse(
        request.context.repository_id, request.context.candidate_id, decision,
        "Independent additions share a local structural choice in the supplied history.",
        tuple(item.evidence_id for item in request.evidence),
    )


def test_request_recovers_commits_paths_modules_and_closed_evidence(inputs):
    request = request_for(inputs)
    assert request.parent_package == "nebula.plugins"
    assert request.structural_scope == "nebula/plugins"
    assert request.context.current_modules == tuple(f"nebula.plugins.unit_{i}" for i in range(3))
    commits = {item.commit_id: item for item in request.context.supporting_commits}
    refs = {item.evidence_id for item in request.evidence}
    for event in request.structural_events:
        changed, = commits[event.commit_id].relevant_changed_files
        assert changed.path == event.path and changed.kind == "added"
        assert set(event.evidence_ids) <= refs
        assert set(changed.evidence_ids) <= refs


def test_tendency_has_separate_criteria_and_no_workflow_steps(inputs):
    request = request_for(inputs)
    prompt = build_semantic_tendency_prompt(request)
    assert "No multi-step process or fixed execution order is needed" in prompt
    assert "SUPPORTED_TENDENCY" in prompt
    accepted = validate_semantic_tendency_response(request, response_for(request))
    assert accepted.rule.type == "convention"
    workflow_prompt = build_semantic_workflow_prompt(request.context)
    assert "SUPPORTED_TENDENCY" not in workflow_prompt
    assert "WORKFLOW" in workflow_prompt


@pytest.mark.parametrize("decision", [
    TendencyDecisionKind.INSUFFICIENT_EVIDENCE, TendencyDecisionKind.INCOHERENT,
])
def test_rejection_remains_rejection(inputs, decision):
    request = request_for(inputs)
    response = response_for(request, decision)
    assert validate_semantic_tendency_response(request, response) == response


def test_single_event_cannot_be_accepted(inputs):
    request = request_for(inputs)
    request = replace(request, structural_events=request.structural_events[:1])
    with pytest.raises(ValueError, match="independent"):
        validate_semantic_tendency_response(request, response_for(request))


def test_mixed_roles_cannot_be_accepted(inputs):
    request = request_for(inputs)
    events = (*request.structural_events[:-1],
              replace(request.structural_events[-1], path="other/unit.py"))
    request = replace(request, structural_events=events)
    with pytest.raises(ValueError, match="different structural roles"):
        validate_semantic_tendency_response(request, response_for(request))


@pytest.mark.parametrize("reason", ["Always create modules.", "Modules must be separate."])
def test_policy_claims_are_rejected(inputs, reason):
    request = request_for(inputs)
    with pytest.raises(ValueError, match="policy"):
        validate_semantic_tendency_response(request, replace(response_for(request), reason=reason))


def test_counter_changes_visible_and_canonical_wording_bounded(inputs):
    root, _, _, _, _, _ = inputs
    from skillforge.repository.change_patterns import ChangePatternMiner
    from skillforge.repository.git_history import GitHistoryMiner
    from skillforge.repository.static_analysis import PythonStaticAnalyzer
    from skillforge.repository.workflow_candidates import WorkflowCandidateMiner

    # Make this an actual modification, rather than an identical file write.
    (root / "nebula/plugins/unit_0.py").write_text("value = 1\n", encoding="utf-8")
    from test_structural_roles import _git
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "extend existing capability")
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    candidate, = candidates.candidates
    request = build_semantic_tendency_request(history, facts, candidates, candidate)
    assert any(file.kind == "modified" for commit in request.examined_changes
               for file in commit.relevant_changed_files)
    accepted = validate_semantic_tendency_response(request, response_for(request))
    assert "does not establish a universal choice" in accepted.rule.statement
    assert accepted.rule.support.examined == len(history.changesets)


def test_selection_allows_role_and_records_limit_rejection(inputs):
    from test_structural_roles import _git

    from skillforge.repository.change_patterns import ChangePatternMiner
    from skillforge.repository.git_history import GitHistoryMiner
    from skillforge.repository.static_analysis import PythonStaticAnalyzer
    from skillforge.repository.workflow_candidates import WorkflowCandidateMiner

    root = inputs[0]
    for index in range(4):
        for path in ("README.md", "notes.txt", "changelog.md"):
            (root / path).write_text(f"revision {index}\n", encoding="utf-8")
        _git(root, "add", "README.md", "notes.txt", "changelog.md")
        _git(root, "commit", "-qm", f"documentation update {index}")
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    result = WorkflowCandidateMiner().mine(history, facts, patterns)
    competitor, role = result.candidates
    assert competitor.kind == ChangePatternKind.FILE_CO_CHANGE
    assert role.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE
    assert select_semantic_candidates(result, limit=1) == (competitor,)
    assert select_semantic_candidates(result, limit=2) == (competitor, role)
    report = semantic_selection_report(result, limit=1)
    assert report[0]["selected"] and not report[1]["selected"]


def test_prepare_dispatches_role_before_dna_and_router(inputs):
    from skillforge.domain.agent_integration import CodeAgentTarget
    from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer

    captured = []

    class StopAtSemantic(Exception):
        pass

    class Model:
        def complete(self, request):
            captured.append(request)
            raise StopAtSemantic("stop before DNA / Router")

    with pytest.raises(Exception, match="stop before DNA"):
        SkillForgePreparer(tendency_model=Model()).prepare(
            PrepareRequest(inputs[0], "inspect package", CodeAgentTarget.CODEX),
        )
    assert len(captured) == 1
    assert captured[0].parent_package == "nebula.plugins"


def test_advisory_dna_compilation_preserves_scope_and_evidence(inputs):
    _, _, history, facts, _, candidates = inputs

    class Model:
        def complete(self, request):
            return response_for(request)

    result = SemanticTendencyEvaluator().evaluate(
        history, facts, candidates, Model(), selected_candidates=candidates.candidates,
    )
    dna = ProjectDNABuilder().build(candidates.repository_id, semantic_tendencies=result)
    compiled = SkillCompiler().compile(dna, semantic_tendencies=result)
    skill, = compiled.skills
    assert dna.workflows == ()
    assert skill.source_status == "candidate"
    assert skill.constraint.enforcement == "advisory"
    assert skill.source_scope.selectors == ("nebula/plugins",)
    assert skill.evidence_ids == dna.rules[0].evidence_ids
    assert "advisory" in _render_guidance(skill, "history-convention").lower()


def test_missing_cited_events_are_rejected(inputs):
    request = request_for(inputs)
    response = replace(response_for(request), evidence_ids=(request.evidence[0].evidence_id,))
    with pytest.raises(ValueError, match="every supporting event"):
        validate_semantic_tendency_response(request, response)


def test_event_citation_lookup_is_exact_and_does_not_decide(inputs):
    import json
    request = request_for(inputs)
    context = json.loads(build_semantic_tendency_prompt(request).split("CANDIDATE_CONTEXT:\n", 1)[1])
    options = context.pop("event_citation_options")
    assert context == request.to_dict()
    sources = {item.evidence_id: item for item in request.evidence}
    assert len(options) == len(request.structural_events)
    for index, (option, event) in enumerate(zip(options, request.structural_events, strict=True)):
        expected = sorted(key for key in event.evidence_ids if sources[key].provenance == "git_diff"
                          and sources[key].path == event.path and sources[key].commit_id == event.commit_id)
        assert option == {"event_index": index, "evidence_ids": expected}
        assert expected
    # Missing citations still fail; no lookup fills the response on behalf of the model.
    with pytest.raises(ValueError, match="every supporting event"):
        validate_semantic_tendency_response(request, replace(response_for(request),
            evidence_ids=(options[0]["evidence_ids"][0],)))


def test_unknown_entities_in_reason_are_rejected(inputs):
    request = request_for(inputs)
    response = replace(response_for(request), reason="Observed choice in invented.package.")
    with pytest.raises(ValueError, match="unknown module"):
        validate_semantic_tendency_response(request, response)


def test_tendency_cli_reuses_isolated_transport(inputs):
    import json
    import subprocess
    from pathlib import Path

    from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
    from skillforge.repository.semantic_reference_schema import bind_response_schema
    from skillforge.repository.semantic_tendency_prompt import tendency_citation_schema

    request = request_for(inputs)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs["input"] == build_semantic_tendency_prompt(request)
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert Path(kwargs["cwd"]) != inputs[0]
        schema = Path(command[command.index("--output-schema") + 1])
        bound = json.loads(schema.read_text(encoding="utf-8"))
        assert bound == bind_response_schema(tendency_citation_schema(request), request.to_dict())
        assert set(bound["$defs"]["evidence"]["enum"]) == {
            item.evidence_id for item in request.evidence
        }
        output = Path(command[command.index("-o") + 1])
        response = response_for(request)
        output.write_text(json.dumps({
            "decision": response.decision, "reason": response.reason,
            "evidence_ids": response.evidence_ids,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    response = CodexCliTendencySemanticModel(runner=runner).complete(request)
    from skillforge.repository.semantic_tendencies import bind_tendency_decision
    assert bind_tendency_decision(request, response) == response_for(request) and len(calls) == 1
