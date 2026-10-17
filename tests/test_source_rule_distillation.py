"""Focused contracts for evidence-bound current-source rule distillation."""

import json
from pathlib import Path

import pytest
from test_skill_registry import compilation, skill

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.integration import AgentSkillExporter
from skillforge.integration.source_rule_distillation import (
    SourceRuleValidationError,
    _bind_line_references,
    _resolve_codex_executable,
)
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def _route(*, matches=False, task=None):
    skills = (skill("historical", task="create route"),) if matches else ()
    registry = SkillRegistryBuilder().build(compilation(skills))
    task = task or ("create route" if matches else
                    "write the requested feature using the supplied oracle")
    return SkillRouter().route(registry, SkillRoutingRequest("repo", task))


def _scope(*paths):
    return ProjectScope(ProjectScopeKind.DIRECTORY, tuple(paths or ("src",)))


def _write(root: Path, relative: str, contents: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


class FakeModel:
    """Build evidence from this call's actual payload, never canned source IDs/quotes."""

    def __init__(self, *, mutate=None, transform=None, condition="when value is zero",
                 action="return the explicit zero result", exceptions=None, ordering=None):
        self.calls = []
        self.mutate = mutate
        self.transform = transform
        self.condition = condition
        self.action = action
        self.exceptions = ["None remains distinct from zero"] if exceptions is None else exceptions
        self.ordering = ["check None before zero"] if ordering is None else ordering

    def complete(self, request):
        self.calls.append(request)
        sources = request["sources"]
        eps = [{"source_id": item["source_id"], "symbol": item["symbol"]}
               for item in sources if item["symbol"] != "module"]
        eps = eps[:2] or [{"source_id": sources[0]["source_id"], "symbol": sources[0]["symbol"]}]
        evidence = [{"source_id": ep["source_id"],
                     "quote": next(s["text"].strip()[:180] for s in sources
                                   if s["source_id"] == ep["source_id"] and s["text"].strip())}
                    for ep in eps]
        raw = {"rules": [{"condition": self.condition, "action": self.action,
                          "exceptions": self.exceptions, "ordering": self.ordering,
                          "entrypoints": eps, "evidence": evidence}]}
        if self.mutate:
            self.mutate()
        return self.transform(raw, request) if self.transform else raw


@pytest.mark.parametrize("numbered", [False, True])
def test_response_schema_requires_evidenced_entrypoints_like_runtime_validator(numbered):
    from skillforge.integration.source_rule_distillation import (
        _line_reference_schema,
        source_rule_response_schema,
    )

    schema = _line_reference_schema(()) if numbered else source_rule_response_schema()
    fields = schema["properties"]["rules"]["items"]["properties"]
    assert fields["evidence"]["minItems"] == 1
    if numbered:
        assert "entrypoints" not in fields
        assert fields["evidence"]["maxItems"] <= 12
    else:
        assert fields["entrypoints"]["minItems"] == 1


def test_line_protocol_derives_only_quoted_entrypoints_without_mutating_response():
    import copy

    from skillforge.integration.source_evidence_binding import bind_evidenced_entrypoints

    sources = ({"source_id": "s1", "symbol": "Service.apply", "text": "one\ntwo"},
               {"source_id": "s2", "symbol": "unquoted", "text": "other"})
    raw = {"rules": [{"condition": "a condition", "action": "an action", "exceptions": [],
                      "ordering": [], "evidence": [
                          {"source_id": "s1", "start_line": 1, "end_line": 1},
                          {"source_id": "s1", "start_line": 2, "end_line": 2}]}]}
    saved = copy.deepcopy(raw)
    bound = bind_evidenced_entrypoints(_bind_line_references(raw, sources), sources)
    assert bound["rules"][0]["entrypoints"] == [{"source_id": "s1", "symbol": "Service.apply"}]
    assert raw == saved


def _files(plan):
    return {item.relative_path: item.content for item in plan.files}


def _export(root, model, *, route=None, scope=None, enabled=True):
    return AgentSkillExporter().current_source_rules(
        route or _route(), CodeAgentTarget.CODEX, repository_root=root,
        scope=scope or _scope(), enabled=enabled, model=model,
    )


def test_default_adapter_resolves_windows_shim_to_installed_native_cli(tmp_path):
    shim = tmp_path / "npm" / "codex.cmd"
    shim.parent.mkdir(parents=True)
    shim.touch()
    native = shim.parent / (
        "node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/"
        "vendor/x86_64-pc-windows-msvc/bin/codex.exe"
    )
    native.parent.mkdir(parents=True)
    native.touch()

    def which(name):
        return None if name == "codex.exe" else str(shim)

    assert _resolve_codex_executable(which=which, windows=True) == str(native)


def test_line_references_bind_exact_crlf_source_and_reject_bad_ranges():
    sources = ({"source_id": "s1", "text": "def run():\r\n    return 1\r\n"},)
    response = {"rules": [{"evidence": [
        {"source_id": "s1", "start_line": 1, "end_line": 2}
    ]}]}
    bound = _bind_line_references(response, sources)
    assert bound["rules"][0]["evidence"][0]["quote"] == "def run():\n    return 1"
    with pytest.raises(SourceRuleValidationError, match="exceeds"):
        _bind_line_references({"rules": [{"evidence": [
            {"source_id": "s1", "start_line": 1, "end_line": 3}
        ]}]}, sources)
    with pytest.raises(SourceRuleValidationError, match="invalid"):
        _bind_line_references({"rules": [{"evidence": [
            {"source_id": "foreign", "start_line": 1, "end_line": 1}
        ]}]}, sources)


def test_long_line_range_is_lossless_bounded_and_does_not_mutate_model_response():
    import copy

    text = "长😀\\\"" * 400 + "\r\nreturn result"
    sources = ({"source_id": "s1", "text": text},)
    raw = {"rules": [{"evidence": [
        {"source_id": "s1", "start_line": 1, "end_line": 2}
    ]}]}
    original = copy.deepcopy(raw)
    bound = _bind_line_references(raw, sources)
    quotes = bound["rules"][0]["evidence"]
    assert len(quotes) > 1
    assert all(0 < len(item["quote"]) <= 500 for item in quotes)
    assert "".join(item["quote"] for item in quotes) == text.replace("\r\n", "\n")
    assert raw == original


def test_line_protocol_states_bounds_without_increasing_instruction_bytes():
    from skillforge.integration.source_rule_distillation import _line_reference_prompt, _Request

    legacy = (
        "Distill reusable behavior shown in the numbered source lines. Return JSON matching the "
        "schema. For each evidence item select source_id and inclusive 1-based "
        "start_line/end_line; "
        "never copy or paraphrase source text. Rules are advisory CURRENT_SOURCE_RULE_CANDIDATEs, "
        "not proof, confirmed history, or normative policy. Preserve conditions, exceptions, and "
        "branch ordering; generalize only when supplied entrypoints support it.\nRequest:\n"
    )
    prefix = _line_reference_prompt(_Request("repo", "candidate", ())).split("Request:\n")[0]
    assert "500-character" in prefix and "16 resulting quotes" in prefix
    assert "same-source_id evidence" in prefix
    assert len(prefix.encode("utf-8")) <= min(
        420, len(legacy.split("Request:\n")[0].encode("utf-8"))
    )


@pytest.mark.parametrize("numbered", [False, True])
def test_prompt_protocol_preserves_source_payload_and_existing_instruction_budget(numbered):
    from skillforge.integration.source_rule_distillation import (
        _line_reference_prompt,
        _Request,
        source_rule_prompt,
    )

    source = {"source_id": "s1", "path": "src/示例.py", "sha256": "abc",
              "symbol": "wrap", "text": "def wrap(value):\n\t# 保留条件\n\n\treturn value\n"}
    request = _Request("repo", "candidate", (source,))
    prompt = (_line_reference_prompt if numbered else source_rule_prompt)(request)
    prefix, payload = prompt.split("Request:\n", 1)
    expected = {**source}
    if numbered:
        expected["text"] = "\n".join(
            f"{index}: {line}" for index, line in enumerate(source["text"].splitlines(), 1)
        )
    assert json.loads(payload) == {
        "repository_id": "repo", "candidate_id": "candidate", "sources": [expected]
    }
    assert source["text"].endswith("return value\n")
    assert len(prefix.encode("utf-8")) <= (420 if numbered else 511)


def test_line_ranges_reject_overflow_without_dropping_evidence_or_mutating_response():
    import copy

    sources = ({"source_id": "s1", "text": "x" * 8001},)
    raw = {"rules": [{"evidence": [
        {"source_id": "s1", "start_line": 1, "end_line": 1}
    ]}]}
    original = copy.deepcopy(raw)
    with pytest.raises(SourceRuleValidationError, match="item limit"):
        _bind_line_references(raw, sources)
    assert raw == original


def test_bound_long_evidence_still_passes_full_exact_source_and_entrypoint_validator():
    from skillforge.integration.source_rule_distillation import _validate

    source = {"source_id": "s1", "symbol": "apply", "text": "x" * 1501}
    raw = {"rules": [{"condition": "when the source condition applies",
                      "action": "preserve the supported result", "exceptions": [],
                      "ordering": [], "entrypoints": [{"source_id": "s1", "symbol": "apply"}],
                      "evidence": [{"source_id": "s1", "start_line": 1, "end_line": 1}]}]}
    bound = _bind_line_references(raw, (source,))
    assert len(_validate(bound, [source])) == 1
    bound["rules"][0]["evidence"][0]["quote"] = "invented quote"
    with pytest.raises(SourceRuleValidationError, match="exact supplied"):
        _validate(bound, [source])


def test_default_off_and_historical_match_return_before_filesystem_or_model(monkeypatch, tmp_path):
    class Forbidden:
        def complete(self, request):
            raise AssertionError("model called on early return")

    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem inspected on early return")

    monkeypatch.setattr(Path, "resolve", forbidden)
    missing = tmp_path / "missing"
    assert _export(missing, Forbidden(), enabled=False) is None
    assert _export(missing, Forbidden(), route=_route(matches=True)) is None


def test_model_receives_only_explicitly_scoped_current_source_not_request_or_oracle(tmp_path):
    _write(tmp_path, "src/main.py", "def normalize(value):\n    return value\n")
    _write(tmp_path, "tests/oracle.py", "ORACLE_PRIVATE = 'should not be sent'\n")
    _write(tmp_path, "evals/task.json", '{"task": "private prompt"}')
    _write(tmp_path, "src/api_token.py", "TOKEN = 'secret-value'\ndef f(): return 2\n")
    _write(tmp_path, "other-repo/src/private.py",
           "OTHER_REPO_PRIVATE = 'outside the requested repository scope'\n")
    model = FakeModel()

    _export(tmp_path, model, route=_route(task="private task description oracle"))

    request = model.calls[0]
    payload = json.dumps(request)
    assert set(request) == {"repository_id", "candidate_id", "sources"}
    assert "private task description" not in payload
    assert "ORACLE_PRIVATE" not in payload and "should not be sent" not in payload
    assert "private prompt" not in payload and "secret-value" not in payload
    assert "outside the requested repository scope" not in payload
    assert all(item["path"] == "src/main.py" for item in request["sources"])


def test_route_task_selects_source_but_never_enters_model_request_or_trace(tmp_path):
    from skillforge.integration.knowledge_trace import KnowledgeTrace

    _write(tmp_path, "src/policy.py",
           "def round_amount(value):\n    if value is None: return 0\n    return round(value)\n\n"
           "def reserve_seat(status):\n    if status != 'available': return False\n"
           "    return True\n")
    task = "reserve seat available status PRIVATE_QUERY_MUST_STAY_LOCAL"
    model, trace = FakeModel(), KnowledgeTrace("repo")
    plan = AgentSkillExporter().current_source_rules(
        _route(task=task), CodeAgentTarget.CODEX, repository_root=tmp_path,
        scope=_scope(), enabled=True, model=model, trace=trace, task_aware_selection=True,
    )
    assert plan is not None
    request = model.calls[0]
    assert request["sources"][0]["symbol"] == "reserve_seat"
    assert set(request) == {"repository_id", "candidate_id", "sources"}
    assert all(set(s) == {"source_id", "path", "sha256", "symbol", "text"}
               for s in request["sources"])
    assert "PRIVATE_QUERY_MUST_STAY_LOCAL" not in json.dumps(request)
    assert "PRIVATE_QUERY_MUST_STAY_LOCAL" not in json.dumps(trace.events)
    assert any(event["reason"] == "TASK_RANKED" for event in trace.events)
    assert any(event["reason"] == "SELECTED" for event in trace.events)

    default_model = FakeModel()
    default_plan = AgentSkillExporter().current_source_rules(
        _route(task=task), CodeAgentTarget.CODEX, repository_root=tmp_path,
        scope=_scope(), enabled=True, model=default_model,
    )
    assert default_plan is not None
    assert default_model.calls[0]["sources"][0]["symbol"] == "round_amount"


@pytest.mark.parametrize("tamper", ["unknown_source", "nonmatching_quote", "fabricated_entrypoint"])
def test_unknown_sources_quotes_and_entrypoints_are_rejected(tmp_path, tamper):
    _write(tmp_path, "src/main.py", "def normalize(value):\n    return value\n")

    def corrupt(raw, request):
        rule = raw["rules"][0]
        if tamper == "unknown_source":
            rule["evidence"][0]["source_id"] = "s-foreign"
        elif tamper == "nonmatching_quote":
            rule["evidence"][0]["quote"] = "return an invented result"
        else:
            rule["entrypoints"][0]["symbol"] = "related_but_unprovided"
        return raw

    with pytest.raises(SourceRuleValidationError):
        _export(tmp_path, FakeModel(transform=corrupt))


def test_each_of_two_entrypoints_requires_its_own_exact_quote(tmp_path):
    _write(tmp_path, "src/one.py", "def normalize(value):\n    return value\n")
    _write(tmp_path, "src/two.py", "def convert(value):\n    return value\n")

    def missing_second_quote(raw, request):
        raw["rules"][0]["evidence"] = raw["rules"][0]["evidence"][:1]
        return raw

    with pytest.raises(SourceRuleValidationError, match="each entrypoint"):
        _export(tmp_path, FakeModel(transform=missing_second_quote))

    plan = _export(tmp_path, FakeModel())
    body = _files(plan)["SKILL.md"]
    assert "normalize" in body and "convert" in body


def test_rule_preserves_condition_action_exceptions_order_and_advisory_status(tmp_path):
    _write(tmp_path, "src/values.py", "def normalize(value):\n"
           "    if value is None: return None\n    if value == 0: return 0\n    return value\n")
    plan = _export(tmp_path, FakeModel())
    files = _files(plan)
    body = files["SKILL.md"]
    evidence = files["references/evidence.md"]
    assert "when value is zero" in body
    assert "return the explicit zero result" in body
    assert "None remains distinct from zero" in body
    assert "check None before zero" in body
    assert "CURRENT_SOURCE_RULE_CANDIDATE" in body
    assert "not proof, confirmed history, or normative policy" in body
    marker = json.loads(files[".skillforge.json"])
    assert marker["source_kind"] == "CURRENT_SOURCE_RULE_CANDIDATE"
    assert marker["source_hashes"]["src/values.py"]
    assert "src/values.py" in evidence


def test_crlf_source_accepts_lf_model_quote_and_documents_exact_normalization(tmp_path):
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir(parents=True)
    raw = b"def normalize(value):\r\n    return value\r\n"
    source.write_bytes(raw)

    def normalize_model_quotes(response, request):
        for evidence in response["rules"][0]["evidence"]:
            evidence["quote"] = evidence["quote"].replace("\r\n", "\n").replace("\r", "\n")
        return response

    plan = _export(tmp_path, FakeModel(transform=normalize_model_quotes))
    files = _files(plan)
    assert "line endings normalized to LF" in files["references/evidence.md"]
    marker = json.loads(files[".skillforge.json"])
    import hashlib

    assert marker["source_hashes"]["src/main.py"] == hashlib.sha256(raw).hexdigest()


def test_stale_source_during_model_call_is_rejected(tmp_path):
    source = _write(tmp_path, "src/main.py", "def normalize(value):\n    return value\n")
    model = FakeModel(mutate=lambda: source.write_text(
        "def normalize(value):\n    return 'changed'\n", encoding="utf-8"))
    with pytest.raises(SourceRuleValidationError, match="source changed"):
        _export(tmp_path, model)


def test_generated_helper_does_not_claim_coverage_of_requested_entrypoint(tmp_path):
    _write(tmp_path, 'src/main.py', 'def _check(value):\n    return value is not None\n\n'
           'def apply_state(value):\n    return _check(value)\n')

    def helper_only(raw, request):
        source = next(s for s in request['sources'] if s['symbol'] == '_check')
        rule = raw['rules'][0]
        rule['entrypoints'] = [dict(source_id=source['source_id'], symbol='_check')]
        rule['evidence'] = [dict(source_id=source['source_id'], quote=source['text'])]
        return raw

    plan = _export(tmp_path, FakeModel(transform=helper_only),
                   route=_route(task='Fix apply_state'))
    marker = json.loads(_files(plan)['.skillforge.json'])
    assert marker['entrypoint_coverage']['requested'] == [['src/main.py', 'apply_state']]
    assert marker['entrypoint_coverage']['missing_from_generation'] == [
        ['src/main.py', 'apply_state']]
    assert marker['entrypoint_coverage']['dropped_by_budget'] == []
    assert marker['entrypoint_coverage']['delivered'] == []


def test_zero_and_missing_values_remain_distinct_in_generic_export(tmp_path):
    _write(tmp_path, "src/values.py", "def normalize(value):\n"
           "    if value is None: return None\n    if value == 0: return 0\n    return value\n")
    model = FakeModel(condition="when value == 0", action="return 0",
                      exceptions=["None returns None"], ordering=["test None before zero"])
    plan = _export(tmp_path, model)
    body = _files(plan)["SKILL.md"]
    assert "when value == 0" in body and "return 0" in body
    assert "None returns None" in body and "test None before zero" in body
    assert "Campus" not in body and "oracle" not in body


def test_export_bounds_complete_rules_and_records_drops_without_mutating_response(tmp_path):
    import copy

    from skillforge.integration.knowledge_trace import KnowledgeTrace

    _write(tmp_path, "src/main.py", "def apply(value):\n    return value\n")
    captured = {}

    def verbose(response, request):
        template = response["rules"][0]
        response["rules"] = [
            {**copy.deepcopy(template), "condition": f"case {index} " + "c" * 350,
             "action": "a" * 350, "exceptions": ["e" * 300] * 3,
             "ordering": ["o" * 300] * 3}
            for index in range(8)
        ]
        captured["original"] = copy.deepcopy(response)
        return response

    model, trace = FakeModel(transform=verbose), KnowledgeTrace("repo")
    plan = AgentSkillExporter().current_source_rules(
        _route(), CodeAgentTarget.CODEX, repository_root=tmp_path, scope=_scope(),
        enabled=True, model=model, trace=trace,
    )
    files = _files(plan)
    docs = [files["SKILL.md"], files["references/evidence.md"]]
    assert sum(len(value.encode("utf8")) for value in docs) <= 8192
    assert len(json.dumps(docs, ensure_ascii=True, separators=(",", ":")).encode()) <= 8192
    marker = json.loads(files[".skillforge.json"])
    assert marker["response"] == captured["original"]
    kept = marker["delivery"]["retained_rule_indexes"]
    assert 0 < len(kept) < 8
    assert len(kept) + len(marker["delivery"]["dropped_rules"]) == 8
    assert any(event["reason"] == "RULE_EXCEEDS_BYTE_BUDGET" for event in trace.events)


def test_plan_id_tracks_source_guidance_evidence_and_model_payload_changes(tmp_path):
    source = _write(tmp_path, "src/main.py",
                    "def normalize(value):\n    normalized = value.strip()\n"
                    "    return normalized\n")
    first = _export(tmp_path, FakeModel())
    changed_guidance = _export(tmp_path, FakeModel(action="preserve the input result"))
    assert first.plan_id != changed_guidance.plan_id
    assert (_files(first)["references/evidence.md"]
            == _files(changed_guidance)["references/evidence.md"])

    def choose_other_exact_quote(raw, request):
        source_text = request["sources"][0]["text"]
        quote = "return normalized"
        assert quote in source_text
        raw["rules"][0]["evidence"][0]["quote"] = quote
        return raw

    changed_evidence = _export(tmp_path, FakeModel(transform=choose_other_exact_quote))
    assert changed_evidence.plan_id != first.plan_id
    assert _files(changed_evidence)["SKILL.md"] == _files(first)["SKILL.md"]
    assert (_files(changed_evidence)["references/evidence.md"]
            != _files(first)["references/evidence.md"])

    source.write_text("def normalize(value):\n    normalized = value.strip()\n"
                      "    return normalized.lower()\n", encoding="utf-8")
    changed_source = _export(tmp_path, FakeModel())
    assert changed_guidance.plan_id != changed_source.plan_id
    marker = json.loads(_files(changed_source)[".skillforge.json"])
    assert marker["source_hashes"]["src/main.py"] != json.loads(
        _files(first)[".skillforge.json"])["source_hashes"]["src/main.py"]
