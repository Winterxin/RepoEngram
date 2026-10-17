"""Offline checks for Codex Anti-pattern evaluation and its CLI."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import test_semantic_antipatterns as fixtures

from skillforge import cli
from skillforge.domain.semantic_antipattern import SemanticAntiPatternDecisionKind
from skillforge.repository import semantic_antipattern_evaluation as evaluation
from skillforge.repository.codex_cli_antipattern import (
    CodexCliAntiPatternError,
    CodexCliAntiPatternSemanticModel,
    parse_semantic_antipattern_response,
)
from skillforge.repository.semantic_antipattern_prompt import build_semantic_antipattern_prompt
from skillforge.repository.semantic_antipatterns import (
    SemanticAntiPatternAbstracter,
    build_semantic_antipattern_request,
)


def _history_and_candidates():
    history = fixtures._history(
        {"paths": ("src/a.py",), "message": "Fix parser input"},
        {"paths": ("src/a.py",), "message": "Follow-up implementation"},
        {"paths": ("src/b.py",), "message": "Fix unrelated item"},
    )
    return history, fixtures._candidate_result(history)


def _request():
    history, candidates = _history_and_candidates()
    return build_semantic_antipattern_request(history, candidates, candidates.candidates[0])


def _runner(request, decision=SemanticAntiPatternDecisionKind.ANTI_PATTERN):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        schema_path = Path(command[command.index("--output-schema") + 1])
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["properties"]["anti_pattern"]["properties"]["scope"]
        assert '"uniqueItems"' not in schema_path.read_text(encoding="utf-8")
        payload = fixtures._valid_response(request, decision).to_dict()
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    return run, calls


@pytest.mark.parametrize("decision", [
    SemanticAntiPatternDecisionKind.ANTI_PATTERN,
    SemanticAntiPatternDecisionKind.INCOHERENT,
    SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
])
def test_adapter_command_isolation_prompt_and_structured_parse(decision):
    request = _request()
    runner, calls = _runner(request, decision)
    response = CodexCliAntiPatternSemanticModel(runner=runner).complete(request)
    assert response.decision == decision
    assert (response.anti_pattern is None) == (decision != "ANTI_PATTERN")
    command, kwargs = calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("-m") + 1] == "gpt-6-luna"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert {"--ignore-user-config", "--ignore-rules", "--ephemeral",
            "--skip-git-repo-check", "--output-schema", "-o"} <= set(command)
    assert command[-1] == "-"
    assert Path(kwargs["cwd"]) == Path(command[command.index("-C") + 1])
    assert Path(kwargs["cwd"]) != Path.cwd()
    assert kwargs["input"] == build_semantic_antipattern_prompt(request)
    assert "src/b.py" not in kwargs["input"]
    assert kwargs["timeout"] == 180


@pytest.mark.parametrize(("failure", "message"), [
    ("missing", "not found"), ("timeout", "timed out"),
    ("nonzero", "exit 1"), ("usage", "usage limit"),
    ("no_output", "did not create"), ("malformed", "malformed JSON"),
    ("unparseable", "cannot be parsed"),
    ("multiline", "unsupported keyword uniqueItems"),
])
def test_adapter_failure_handling(failure, message):
    def run(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("codex")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure in {"nonzero", "usage"}:
            stderr = "ERROR: authentication failed" if failure == "nonzero" else "usage limit"
            return subprocess.CompletedProcess(command, 1, "", stderr)
        if failure == "multiline":
            stderr = (
                'ERROR: {\n  "error": {\n'
                '    "message": "Invalid schema: unsupported keyword uniqueItems"\n'
                '  }\n}'
            )
            return subprocess.CompletedProcess(command, 1, "", stderr)
        if failure in {"malformed", "unparseable"}:
            content = "{" if failure == "malformed" else "{}"
            Path(command[command.index("-o") + 1]).write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(CodexCliAntiPatternError, match=message):
        CodexCliAntiPatternSemanticModel(runner=run).complete(_request())


def test_nonzero_diagnostic_is_bounded_and_retains_multiline_error():
    def run(command, **kwargs):
        stderr = (
            kwargs["input"] + "\nERROR: {\n"
            '  "message": "Invalid schema: unsupported keyword uniqueItems"\n'
            "  " + "x" * 3000 + "\n}"
        )
        return subprocess.CompletedProcess(command, 1, "", stderr)

    with pytest.raises(CodexCliAntiPatternError) as captured:
        CodexCliAntiPatternSemanticModel(runner=run).complete(_request())
    message = str(captured.value)
    assert "unsupported keyword uniqueItems" in message
    assert "CANDIDATE_CONTEXT" not in message
    assert len(message) < 1350


def test_parser_checks_nested_shape_and_types_without_semantic_validation():
    data = fixtures._valid_response(_request()).to_dict()
    data["anti_pattern"]["scope"]["selectors"].append("fake.py")
    assert parse_semantic_antipattern_response(data).anti_pattern.scope.selectors[-1] == "fake.py"
    data["anti_pattern"]["exceptions"] = [{"description": "x", "evidence_ids": [], "extra": 1}]
    with pytest.raises(ValueError, match="exception must be an object"):
        parse_semantic_antipattern_response(data)
    data = fixtures._valid_response(_request()).to_dict()
    data["confidence_score"] = True
    with pytest.raises(ValueError, match="must be a number"):
        parse_semantic_antipattern_response(data)


@pytest.mark.parametrize(("field", "message"), [
    ("name_evidence_ids", "duplicate Evidence IDs"),
    ("scope_selectors", "scope selectors contain duplicates"),
])
def test_validator_still_rejects_duplicate_model_arrays(field, message):
    history, candidates = _history_and_candidates()
    request = build_semantic_antipattern_request(
        history, candidates, candidates.candidates[0]
    )
    payload = fixtures._valid_response(request).to_dict()
    anti_pattern = payload["anti_pattern"]
    if field == "scope_selectors":
        selectors = anti_pattern["scope"]["selectors"]
        anti_pattern["scope"]["selectors"] = selectors + selectors
    else:
        evidence_ids = anti_pattern[field]
        anti_pattern[field] = evidence_ids + evidence_ids

    class ParsedModel:
        def complete(self, _request):
            return parse_semantic_antipattern_response(payload)

    with pytest.raises(ValueError, match=message):
        SemanticAntiPatternAbstracter().abstract(
            history, candidates, ParsedModel(),
            selected_candidates=(candidates.candidates[0],),
        )


def _patch_mining(monkeypatch, history, candidates):
    monkeypatch.setattr(
        evaluation.GitHistoryMiner, "mine", lambda self, path, *, limit: history
    )
    monkeypatch.setattr(
        evaluation.NegativePatternMiner, "mine", lambda self, value: candidates
    )


def test_evaluation_selection_request_closure_and_serialization(tmp_path, monkeypatch):
    history, candidates = _history_and_candidates()
    _patch_mining(monkeypatch, history, candidates)
    assert evaluation.select_semantic_antipattern_candidates(candidates, limit=1) == (
        candidates.candidates[0],
    )
    with pytest.raises(ValueError, match="positive"):
        evaluation.select_semantic_antipattern_candidates(candidates, limit=0)
    model = fixtures.FakeModel()
    report = evaluation.evaluate_semantic_antipatterns(
        tmp_path, history_limit=12, limit=1, model=model
    )
    assert report["history_limit"] == 12
    assert report["selected_count"] == 1
    entry = report["entries"][0]
    assert entry["status"] == "accepted_by_validator"
    assert entry["draft"]["candidate_id"] == entry["candidate"]["candidate_id"]
    assert entry["request"] == model.requests[0].to_dict()
    assert entry["model_response"] == fixtures._valid_response(model.requests[0]).to_dict()
    assert "src/b.py" not in json.dumps(entry["request"])
    output = evaluation.write_semantic_antipattern_evaluation_report(
        report, tmp_path / "reports" / "antipattern-eval.json"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("decision", [
    SemanticAntiPatternDecisionKind.INCOHERENT,
    SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
])
def test_evaluation_preserves_legal_rejections(tmp_path, monkeypatch, decision):
    history, candidates = _history_and_candidates()
    _patch_mining(monkeypatch, history, candidates)
    model = fixtures.FakeModel(lambda request: fixtures._valid_response(request, decision))
    report = evaluation.evaluate_semantic_antipatterns(tmp_path, limit=2, model=model)
    assert report["selected_count"] == 2
    assert all(entry["status"] == "semantic_rejected" for entry in report["entries"])
    assert all(entry["rejection"]["decision"] == decision for entry in report["entries"])
    assert all(entry["validation_error"] is None for entry in report["entries"])


def test_evaluation_separates_validation_and_model_errors(tmp_path, monkeypatch):
    history, candidates = _history_and_candidates()
    _patch_mining(monkeypatch, history, candidates)

    class InvalidModel:
        def complete(self, request):
            return replace(fixtures._valid_response(request), decision_evidence_ids=("fake",))

    report = evaluation.evaluate_semantic_antipatterns(tmp_path, limit=1, model=InvalidModel())
    entry = report["entries"][0]
    assert entry["status"] == "validation_error"
    assert entry["model_response"] is not None
    assert entry["validation_error"]["stage"] == "semantic_validator"

    class FailedModel:
        def complete(self, request):
            raise CodexCliAntiPatternError("offline test failure")

    report = evaluation.evaluate_semantic_antipatterns(tmp_path, limit=1, model=FailedModel())
    assert report["entries"][0]["status"] == "model_error"
    assert report["entries"][0]["validation_error"]["stage"] == "codex_cli"


def test_evaluation_empty_candidates_and_limits(tmp_path, monkeypatch):
    history, candidates = _history_and_candidates()
    empty = replace(candidates, candidates=(), signals=(), evidence=())
    _patch_mining(monkeypatch, history, empty)

    class UnusedModel:
        def complete(self, request):
            pytest.fail("model must not be called")

    report = evaluation.evaluate_semantic_antipatterns(tmp_path, model=UnusedModel())
    assert report["selected_count"] == 0 and report["entries"] == []
    with pytest.raises(ValueError, match="positive"):
        evaluation.evaluate_semantic_antipatterns(tmp_path, limit=0, model=UnusedModel())
    with pytest.raises(ValueError, match="nonnegative"):
        evaluation.evaluate_semantic_antipatterns(
            tmp_path, history_limit=-1, model=UnusedModel()
        )


def test_cli_provider_output_and_exit_codes(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as missing:
        cli.main(["antipattern-eval", str(tmp_path)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as invalid:
        cli.main(["antipattern-eval", str(tmp_path), "--provider", "other"])
    assert invalid.value.code == 2

    report = {"selected_count": 2, "entries": [
        {"status": "semantic_rejected"}, {"status": "semantic_rejected"},
    ]}
    seen = []

    def evaluate(path, *, history_limit, limit):
        seen.append((path, history_limit, limit))
        return report

    monkeypatch.setattr(cli, "evaluate_semantic_antipatterns", evaluate)
    output = tmp_path / "custom.json"
    monkeypatch.setattr(
        cli, "write_semantic_antipattern_evaluation_report",
        lambda value, path: seen.append((value, path)) or path,
    )
    args = ["antipattern-eval", str(tmp_path), "--provider", "codex-cli",
            "--history-limit", "12", "--limit", "2", "--output", str(output)]
    assert cli.main(args) == 0
    assert seen == [(tmp_path, 12, 2), (report, output)]
    assert "Semantic rejections: 2" in capsys.readouterr().out
    report["entries"][1]["status"] = "validation_error"
    assert cli.main(args) == 1
    report["entries"][1]["status"] = "model_error"
    assert cli.main(args) == 1
