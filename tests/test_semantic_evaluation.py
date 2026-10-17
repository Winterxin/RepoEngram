"""Offline checks for the Codex semantic evaluation boundary."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_semantic_workflows as fixtures

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.semantic_workflow import SemanticDecisionKind, SemanticDecisionResponse
from skillforge.domain.workflow import WorkflowCandidateMember
from skillforge.repository import semantic_evaluation
from skillforge.repository.codex_cli_semantic import (
    CodexCliSemanticError,
    CodexCliSemanticModel,
    parse_semantic_workflow_response,
)
from skillforge.repository.semantic_workflow_prompt import CLASSIFICATION_PROTOCOL
from skillforge.repository.semantic_workflows import build_semantic_workflow_request


def _request():
    history, static, candidates = fixtures._fixture()
    return build_semantic_workflow_request(history, static, candidates, candidates.candidates[0])


def _decision_payload(request):
    data = fixtures.FakeModel().complete(request).to_dict()
    for field in ("repository_id", "candidate_id", "confidence_score"):
        del data["workflow"][field]
    return data


def _current_protocol_payload(request):
    data = _decision_payload(request)
    data["classification_protocol"] = CLASSIFICATION_PROTOCOL
    workflow = data["workflow"]
    by_step = {}
    for kind in ("required", "optional", "conditional"):
        for item in workflow.pop(f"{kind}_steps"):
            assert item["step_id"] not in by_step
            by_step[item["step_id"]] = (kind, item)
    for step in workflow["ordered_steps"]:
        kind, item = by_step[step["step_id"]]
        step.update({
            "classification_kind": kind,
            "classification_reason": item["reason"],
            "classification_evidence_ids": item["evidence_ids"],
            "classification_condition": item["condition"],
        })
    return data


def _runner_with_response(request, mutate=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert "decision" in schema["required"]
        workflow = schema["properties"]["workflow"]
        assert workflow["properties"]["scope"]["properties"]["kind"]["enum"]
        payload = _decision_payload(request)
        if mutate:
            mutate(payload)
        Path(command[command.index("-o") + 1]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run, calls


def _selection_candidate(candidate_id, units, support, *, kind=ChangePatternKind.MODULE_CO_CHANGE,
                         commits=("commit-1", "commit-2")):
    members = tuple(
        WorkflowCandidateMember(
            value=value,
            current_module=value if namespace == "module" else None,
        )
        for namespace, value in units
    )
    current_modules = tuple(sorted({
        member.current_module for member in members if member.current_module is not None
    }))
    return SimpleNamespace(
        candidate_id=candidate_id,
        kind=kind,
        members=members,
        current_modules=current_modules,
        supporting_commits=tuple(commits),
        support=SimpleNamespace(supporting=support),
    )


def test_command_schema_isolation_and_parsing():
    request = _request()
    runner, calls = _runner_with_response(request)
    response = CodexCliSemanticModel(runner=runner).complete(request)
    assert isinstance(response, SemanticDecisionResponse)
    assert response.candidate_id == request.candidate_id
    command, kwargs = calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("-m") + 1] == "gpt-6-luna"
    assert {
        "--output-schema", "-o", "--ephemeral", "--ignore-user-config",
        "--ignore-rules", "read-only",
    } <= set(command)
    assert command[-1] == "-"
    assert Path(kwargs["cwd"]) == Path(command[command.index("-C") + 1])
    assert kwargs["input"].count("CANDIDATE_CONTEXT:") == 1
    assert str(Path.cwd()) not in kwargs["input"]
    assert kwargs["timeout"] == 180


@pytest.mark.parametrize("failure,expected", [
    ("missing", "not found"),
    ("timeout", "timed out"),
    ("nonzero", "exit 1"),
    ("no_output", "did not create"),
    ("malformed", "malformed JSON"),
    ("unparseable", "cannot be parsed"),
])
def test_adapter_failures_are_clear(failure, expected):
    def run(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("codex")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            return subprocess.CompletedProcess(command, 1, "", "not logged in")
        if failure == "malformed":
            Path(command[command.index("-o") + 1]).write_text("{", encoding="utf-8")
        if failure == "unparseable":
            Path(command[command.index("-o") + 1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(CodexCliSemanticError, match=expected):
        CodexCliSemanticModel(runner=run).complete(_request())


def test_nonzero_error_does_not_repeat_prompt():
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, "", kwargs["input"] + "\nERROR: authentication failed"
        )

    with pytest.raises(CodexCliSemanticError, match="authentication failed") as captured:
        CodexCliSemanticModel(runner=run).complete(_request())
    assert "CANDIDATE_CONTEXT" not in str(captured.value)


def test_structured_response_rejects_wrong_nested_shape():
    request = _request()
    data = _decision_payload(request)
    data["workflow"]["ordered_steps"][0]["unexpected"] = "x"
    with pytest.raises(ValueError, match="step must be an object"):
        parse_semantic_workflow_response(data)


def test_v2_classification_maps_one_conditional_step_and_legacy_still_parses():
    request = _request()
    data = _current_protocol_payload(request)
    step = data["workflow"]["ordered_steps"][0]
    step["classification_kind"] = "conditional"
    step["classification_condition"] = "when the evidenced contract is present"
    parsed = parse_semantic_workflow_response(data)
    workflow = parsed.workflow
    assert workflow is not None
    assert workflow.conditional_steps[0].step_id == step["step_id"]
    assert workflow.conditional_steps[0].condition == step["classification_condition"]
    assert all(item.step_id != step["step_id"] for item in workflow.required_steps)
    assert all(item.step_id != step["step_id"] for item in workflow.optional_steps)

    legacy = parse_semantic_workflow_response(_decision_payload(request))
    assert legacy.workflow is not None
    assert legacy.workflow.required_steps or legacy.workflow.optional_steps


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda step: step.update(classification_kind="surprise"), "classification_kind"),
    (lambda step: step.update(
        classification_kind="required", classification_condition="always"
    ), "require a null condition"),
    (lambda step: step.update(classification_evidence_ids=[]), "must not be empty"),
    (lambda step: step.update(evidence_ids=[]), "must not be empty"),
    (lambda step: step.update(classification_kind="conditional", classification_condition=""),
     "non-empty condition"),
])
def test_v2_rejects_invalid_kind_conditions_and_evidence(mutation, message):
    request = _request()
    data = _current_protocol_payload(request)
    mutation(data["workflow"]["ordered_steps"][0])
    with pytest.raises(ValueError, match=message):
        parse_semantic_workflow_response(data)


def test_v2_rejects_unknown_protocol_and_duplicate_step_ids():
    request = _request()
    data = _current_protocol_payload(request)
    data["classification_protocol"] = "future-protocol"
    with pytest.raises(ValueError, match="unsupported classification_protocol"):
        parse_semantic_workflow_response(data)

    data = _current_protocol_payload(request)
    steps = data["workflow"]["ordered_steps"]
    steps[1]["step_id"] = steps[0]["step_id"]
    with pytest.raises(ValueError, match="duplicate step_id"):
        parse_semantic_workflow_response(data)


def test_legacy_duplicate_classification_still_reaches_existing_rejection():
    from skillforge.repository.semantic_workflows import _validate_response

    request = _request()
    legacy = _decision_payload(request)
    source_group = next(
        name for name in ("required_steps", "optional_steps", "conditional_steps")
        if legacy["workflow"][name]
    )
    target_group = next(name for name in ("required_steps", "optional_steps")
                        if name != source_group)
    duplicate = dict(legacy["workflow"][source_group][0])
    duplicate["condition"] = None
    legacy["workflow"][target_group].append(duplicate)
    parsed = parse_semantic_workflow_response(legacy)
    with pytest.raises(ValueError, match="multiple required/optional/conditional"):
        _validate_response(request, parsed.workflow)


def test_selection_preserves_miner_order_and_top_n():
    _, _, result = fixtures._fixture()
    selected = semantic_evaluation.select_semantic_candidates(
        result, kind=ChangePatternKind.MODULE_CO_CHANGE, limit=2
    )
    assert selected == tuple(
        candidate for candidate in result.candidates
        if candidate.kind == ChangePatternKind.MODULE_CO_CHANGE
    )[:2]


def test_selection_collapses_matching_file_module_candidates_and_reports_why():
    _, _, original = fixtures._fixture()
    base = original.candidates[0]
    file_candidate = replace(
        base, candidate_id="file-candidate", kind=ChangePatternKind.FILE_CO_CHANGE,
    )
    module_candidate = replace(
        base, candidate_id="module-candidate", kind=ChangePatternKind.MODULE_CO_CHANGE,
    )
    independent = replace(
        base,
        candidate_id="independent-candidate",
        current_modules=("pkg.unrelated",),
        members=(replace(base.members[0], value="pkg.unrelated", current_module="pkg.unrelated"),),
        supporting_commits=("different-commit",),
    )
    result = replace(original, candidates=(file_candidate, module_candidate, independent))

    selected = semantic_evaluation.select_semantic_candidates(result, limit=2)
    assert tuple(item.candidate_id for item in selected) == (
        "module-candidate", "independent-candidate",
    )
    report = {item["candidate_id"]: item for item in
              semantic_evaluation.semantic_selection_report(result, limit=2)}
    assert report["file-candidate"]["selected"] is False
    assert "module-candidate" in report["file-candidate"]["reason"]
    assert "module-level representative" in report["module-candidate"]["reason"]
    assert report["independent-candidate"]["selected"] is True
    assert result.evidence == original.evidence

    duplicate_only = replace(original, candidates=(file_candidate, module_candidate))
    duplicate_only_selected = semantic_evaluation.select_semantic_candidates(
        duplicate_only, limit=2,
    )
    assert tuple(item.candidate_id for item in duplicate_only_selected) == (
        "file-candidate", "module-candidate",
    )


def test_selection_does_not_collapse_when_only_module_overlap_matches():
    _, _, original = fixtures._fixture()
    base = original.candidates[0]
    first = replace(base, candidate_id="first")
    same_commits_different_modules = replace(
        base,
        candidate_id="different-modules",
        current_modules=("pkg.other",),
        members=(replace(base.members[0], value="pkg.other", current_module="pkg.other"),),
    )
    same_modules_different_commits = replace(
        base, candidate_id="different-commits", supporting_commits=("other-commit",),
    )
    result = replace(
        original,
        candidates=(first, same_commits_different_modules, same_modules_different_commits),
    )

    selected = semantic_evaluation.select_semantic_candidates(result, limit=3)
    assert tuple(item.candidate_id for item in selected) == (
        "first", "different-modules", "different-commits",
    )


def test_mixed_python_and_unmapped_ui_member_prevents_file_module_collapse():
    _, _, original = fixtures._fixture()
    base = original.candidates[0]
    module_member = WorkflowCandidateMember("pkg.core", "pkg.core")
    file_candidate = replace(
        base,
        candidate_id="python-and-ui-file",
        kind=ChangePatternKind.FILE_CO_CHANGE,
        members=(
            WorkflowCandidateMember("src/pkg/core.py", "pkg.core"),
            WorkflowCandidateMember("web/ui.js", None),
        ),
        current_modules=("pkg.core",),
    )
    module_candidate = replace(
        base,
        candidate_id="python-only-module",
        kind=ChangePatternKind.MODULE_CO_CHANGE,
        members=(module_member,),
        current_modules=("pkg.core",),
        supporting_commits=file_candidate.supporting_commits,
    )
    result = replace(original, candidates=(file_candidate, module_candidate))

    selected = semantic_evaluation.select_semantic_candidates(result, limit=2)
    assert tuple(item.candidate_id for item in selected) == (
        "python-and-ui-file", "python-only-module",
    )


def test_equal_support_but_different_commits_does_not_replace_file_candidate():
    _, _, original = fixtures._fixture()
    base = original.candidates[0]
    file_candidate = replace(
        base, candidate_id="file-first", kind=ChangePatternKind.FILE_CO_CHANGE,
    )
    module_candidate = replace(
        base,
        candidate_id="module-different-commits",
        kind=ChangePatternKind.MODULE_CO_CHANGE,
        supporting_commits=tuple(f"other-{index}" for index in
                                 range(len(base.supporting_commits))),
    )
    result = replace(original, candidates=(file_candidate, module_candidate))

    selected = semantic_evaluation.select_semantic_candidates(result, limit=2)
    assert tuple(item.candidate_id for item in selected) == (
        "file-first", "module-different-commits",
    )


def test_duplicate_slots_keep_original_workflow_and_tendency_quotas():
    _, _, original = fixtures._fixture()
    base = original.candidates[0]
    file_candidate = replace(
        base, candidate_id="file-first", kind=ChangePatternKind.FILE_CO_CHANGE,
    )
    tendency = replace(
        base, candidate_id="tendency-in-original-head",
        kind=ChangePatternKind.ADDED_PYTHON_MODULE_ROLE,
    )
    module_candidate = replace(
        base, candidate_id="module-equivalent", kind=ChangePatternKind.MODULE_CO_CHANGE,
    )
    result = type("CandidateResult", (), {
        "candidates": (file_candidate, tendency, module_candidate),
    })()

    for limit in (1, 2, 3):
        selected = semantic_evaluation.select_semantic_candidates(result, limit=limit)
        expected_channels = {
            channel: sum(
                semantic_evaluation._selection_channel(candidate) == channel
                for candidate in (file_candidate, tendency, module_candidate)[:limit]
            )
            for channel in ("workflow", "tendency")
        }
        actual_channels = {
            channel: sum(
                semantic_evaluation._selection_channel(candidate) == channel
                for candidate in selected
            )
            for channel in ("workflow", "tendency")
        }
        assert actual_channels == expected_channels
    assert {candidate.candidate_id for candidate in selected} == {
        "file-first", "tendency-in-original-head", "module-equivalent",
    }


def test_selection_keeps_strongest_first_then_admits_disjoint_module_region():
    first = _selection_candidate("strong", (("module", "pkg.a"),), 10)
    overlapping = _selection_candidate("overlap", (("module", "pkg.a"),), 9)
    disjoint = _selection_candidate("disjoint", (("module", "pkg.z"),), 2)
    result = SimpleNamespace(candidates=(first, overlapping, disjoint))

    selected = semantic_evaluation.select_semantic_candidates(result, limit=2)
    assert tuple(candidate.candidate_id for candidate in selected) == ("strong", "disjoint")


def test_selection_prefers_new_member_ratio_over_raw_new_member_count():
    shared = tuple(("module", f"pkg.shared{index}") for index in range(8))
    first = _selection_candidate("strong", shared, 10)
    high_ratio = _selection_candidate(
        "high-ratio", (shared[0], ("module", "pkg.one-new")), 9,
    )
    more_new_members_lower_ratio = _selection_candidate(
        "more-members-lower-ratio",
        (*shared, ("module", "pkg.new-a"), ("module", "pkg.new-b")), 8,
    )
    result = SimpleNamespace(
        candidates=(first, high_ratio, more_new_members_lower_ratio),
    )

    selected = semantic_evaluation.select_semantic_candidates(result, limit=2)
    assert tuple(candidate.candidate_id for candidate in selected) == ("strong", "high-ratio")


def test_evaluation_records_validator_rejection_and_serializes(tmp_path, monkeypatch):
    history, static, candidates = fixtures._fixture()
    monkeypatch.setattr(semantic_evaluation.GitHistoryMiner, "mine", lambda self, path: history)
    monkeypatch.setattr(
        semantic_evaluation.PythonStaticAnalyzer, "analyze", lambda self, path: static
    )
    monkeypatch.setattr(semantic_evaluation.ChangePatternMiner, "mine", lambda self, h, s: None)
    monkeypatch.setattr(
        semantic_evaluation.WorkflowCandidateMiner, "mine", lambda self, h, s, p: candidates
    )

    class RejectedModel:
        def complete(self, request):
            model = fixtures.FakeModel(
                lambda req: replace(
                    fixtures._valid_response(req), name_evidence_ids=("unknown-evidence",)
                )
            )
            return model.complete(request)

    report = semantic_evaluation.evaluate_semantic_workflows(
        tmp_path, limit=1, model=RejectedModel()
    )
    assert report["selected_count"] == 1
    assert report["entries"][0]["error"]["stage"] == "semantic_validator"
    assert report["entries"][0]["decision"] == "WORKFLOW"
    assert report["entries"][0]["model_response"]["workflow"]["name"]
    assert report["entries"][0]["candidate"]["candidate_id"]
    output = semantic_evaluation.write_semantic_evaluation_report(
        report, tmp_path / "reports" / "semantic-eval.json"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_evaluation_keeps_original_candidate_and_accepted_draft(tmp_path, monkeypatch):
    history, static, candidates = fixtures._fixture()
    monkeypatch.setattr(semantic_evaluation.GitHistoryMiner, "mine", lambda self, path: history)
    monkeypatch.setattr(
        semantic_evaluation.PythonStaticAnalyzer, "analyze", lambda self, path: static
    )
    monkeypatch.setattr(semantic_evaluation.ChangePatternMiner, "mine", lambda self, h, s: None)
    monkeypatch.setattr(
        semantic_evaluation.WorkflowCandidateMiner, "mine", lambda self, h, s, p: candidates
    )
    report = semantic_evaluation.evaluate_semantic_workflows(
        tmp_path, limit=1, model=fixtures.FakeModel()
    )
    assert report["model"] == "gpt-6-luna"
    entry = report["entries"][0]
    assert entry["status"] == "accepted_by_validator"
    assert entry["decision"] == "WORKFLOW"
    assert entry["candidate"]["candidate_id"] == entry["semantic"]["candidate_id"]
    assert entry["semantic"]["ordered_steps"]
    assert entry["evidence_ids"]


def test_evaluation_report_preserves_semantic_rejection(tmp_path, monkeypatch):
    history, static, candidates = fixtures._fixture()
    monkeypatch.setattr(semantic_evaluation.GitHistoryMiner, "mine", lambda self, path: history)
    monkeypatch.setattr(
        semantic_evaluation.PythonStaticAnalyzer, "analyze", lambda self, path: static
    )
    monkeypatch.setattr(semantic_evaluation.ChangePatternMiner, "mine", lambda self, h, s: None)
    monkeypatch.setattr(
        semantic_evaluation.WorkflowCandidateMiner, "mine", lambda self, h, s, p: candidates
    )

    class RejectionModel:
        def complete(self, request):
            return replace(
                fixtures.FakeModel().complete(request),
                decision=SemanticDecisionKind.INCOHERENT,
                reason="The supplied examples describe distinct engineering tasks",
                workflow=None,
            )

    report = semantic_evaluation.evaluate_semantic_workflows(
        tmp_path, limit=1, model=RejectionModel()
    )
    entry = report["entries"][0]
    assert entry["decision"] == "INCOHERENT"
    assert entry["status"] == "semantic_rejected"
    assert entry["semantic"] is None
    assert entry["rejection"]["confidence"]["method"] == "llm_semantic_workflow_v1"
