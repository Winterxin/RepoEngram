"""Whole, bounded source observations reach the original workflow contract."""

import ast
from dataclasses import replace

import pytest
from test_git_history import _commit, _init_repo, _write
from test_semantic_workflows import FakeModel

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.evidence import SourceLocation
from skillforge.domain.semantic_workflow import SemanticDecisionKind, SemanticDecisionResponse
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.semantic_workflow_prompt import build_semantic_workflow_prompt
from skillforge.repository.semantic_workflows import (
    SemanticWorkflowAbstracter,
    build_semantic_workflow_request,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner
from skillforge.repository.workflow_code_evidence import (
    MAX_CODE_CONTEXT_BYTES,
    MAX_IMPLEMENTATION_BYTES,
    WorkflowCodeEvidence,
    _reference_size,
)


def _implementations(facts):
    return {item.evidence_id: item for item in facts.evidence
            if item.description.startswith("Python implementation ")}


def test_nonimportable_source_keeps_facts_without_breaking_module_evidence(tmp_path):
    _write(tmp_path, "example-case.py", "def example(value):\n    return value\n")
    _write(tmp_path, "valid.py", "def transform(value):\n    return value\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    example = next(module for module in facts.modules if module.path == "example-case.py")
    assert example.module is None
    assert example.symbols
    implementations = _implementations(facts)
    assert any(item.location.path == "example-case.py" for item in implementations.values())

    indexed = WorkflowCodeEvidence(facts)
    refs = indexed.select(("valid",))
    assert refs
    assert {implementations[key].location.path for key in refs} == {"valid.py"}
    assert "example-case.py" not in indexed.local_functions


def test_complete_functions_and_utf8_budget_without_outside_modules(tmp_path):
    for name in ("alpha", "beta", "outside"):
        _write(tmp_path, name + ".py", "\n".join(
            f"def action_{i}(value):\n    return value + '中' * {i}\n" for i in range(8)
        ))
    _write(tmp_path, "huge.py", "def action():\n    return " + repr("中" * 2000) + "\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    refs = WorkflowCodeEvidence(facts).select(("beta", "alpha", "huge"))
    items = _implementations(facts)
    assert refs == WorkflowCodeEvidence(facts).select(("huge", "alpha", "beta", "alpha"))
    assert {items[key].location.path for key in refs} == {"alpha.py", "beta.py"}
    assert len(refs) == 2
    assert sum(_reference_size(items[key]) for key in refs) <= MAX_CODE_CONTEXT_BYTES
    for key in refs:
        item = items[key]
        assert _reference_size(item) <= MAX_IMPLEMENTATION_BYTES
        body = item.description.partition(": ")[2]
        assert len(ast.parse(body).body) == 1
        assert body.endswith("'中' * 0") or body.endswith("'中' * 1")


def test_large_population_cannot_expand_budget_or_truncate_functions(tmp_path):
    for i in range(20):
        _write(tmp_path, f"group_{i}.py", "def transform(value):\n    return value\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    refs = WorkflowCodeEvidence(facts).select(tuple(m.module for m in facts.modules))
    items = _implementations(facts)
    assert 0 < len(refs) < 20
    assert sum(_reference_size(items[key]) for key in refs) <= MAX_CODE_CONTEXT_BYTES
    assert all(ast.parse(items[key].description.partition(": ")[2]).body for key in refs)


@pytest.mark.parametrize("change", ["repository", "path", "line", "revision", "provenance"])
def test_implementation_cannot_be_relocated_or_mixed_with_history(tmp_path, change):
    _write(tmp_path, "sample.py", "def transform(value):\n    return value\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    body = next(iter(_implementations(facts).values()))
    changed = {
        "repository": replace(body, repository_id="other"),
        "path": replace(body, location=SourceLocation("outside.py", line=body.location.line)),
        "line": replace(body, location=SourceLocation("sample.py", line=99)),
        "revision": replace(body, location=SourceLocation("sample.py", commit="past", line=1)),
        "provenance": replace(body, provenance="generated_summary"),
    }[change]
    facts = replace(facts, evidence=tuple(changed if e.evidence_id == body.evidence_id else e
                                          for e in facts.evidence))
    with pytest.raises(ValueError, match="locator mismatch"):
        WorkflowCodeEvidence(facts)


def test_repeated_multiline_definitions_keep_distinct_implementation_locators(tmp_path):
    _write(
        tmp_path,
        "sample.py",
        "@decorate(\n"
        "    option=True,\n"
        ")\n"
        "def transform(\n"
        "    value: int,\n"
        ") -> int:\n"
        "    return value\n"
        "\n"
        "@decorate(\n"
        "    option=True,\n"
        ")\n"
        "def transform(\n"
        "    value: int,\n"
        ") -> int:\n"
        "    return value\n",
    )
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    module = next(module for module in facts.modules if module.path == "sample.py")
    definitions = [
        symbol for symbol in module.symbols if symbol.qualified_name == "sample.transform"
    ]
    assert [symbol.line for symbol in definitions] == [4, 12]

    implementations = _implementations(facts)
    for symbol in definitions:
        body = next(
            implementations[key]
            for key in symbol.evidence_ids
            if key in implementations
        )
        assert body.location.path == symbol.path
        assert body.location.line == symbol.line
        assert body.provenance == "python_ast"

    assert WorkflowCodeEvidence(facts).select(("sample",))


def _repeated_history(root):
    _init_repo(root)
    for index in range(3):
        _write(root, "gateway.py", "from storage import retain\n"
               "def submit(value):\n    return retain(value)\n" + f"REVISION = {index}\n")
        _write(root, "storage.py", "def retain(value):\n    return value.strip()\n"
               + f"REVISION = {index}\n")
        _write(root, "unrelated.py", "def unrelated(value):\n    return value\n")
        _commit(root, "Coordinate public gateway and storage")
    history = GitHistoryMiner().mine(root)
    facts = PythonStaticAnalyzer().analyze(root)
    patterns = ChangePatternMiner().mine(history, facts)
    return history, facts, patterns


def test_history_candidate_preserves_statistics_and_original_code_closure(tmp_path, monkeypatch):
    history, facts, patterns = _repeated_history(tmp_path)
    current = WorkflowCandidateMiner().mine(history, facts, patterns)
    original_select = WorkflowCodeEvidence.select
    monkeypatch.setattr(WorkflowCodeEvidence, "select", lambda *_: ())
    before = WorkflowCandidateMiner().mine(history, facts, patterns)
    monkeypatch.setattr(WorkflowCodeEvidence, "select", original_select)
    bodies = _implementations(facts)
    previous = {c.candidate_id: c for c in before.candidates}
    for candidate in current.candidates:
        old = previous[candidate.candidate_id]
        added = set(candidate.evidence_ids) - set(old.evidence_ids)
        assert added
        assert added <= bodies.keys()
        assert replace(candidate, evidence_ids=old.evidence_ids) == old
        request = build_semantic_workflow_request(history, facts, current, candidate)
        assert {e.evidence_id for e in request.evidence} == set(candidate.evidence_ids)
        assert {bodies[key].location.path for key in added} == {"gateway.py", "storage.py"}
        assert all(bodies[key].location.commit is None for key in added)
        assert not added & {e.evidence_id for e in history.evidence}
        assert all(e == next(item for item in facts.evidence if item.evidence_id == e.evidence_id)
                   for e in current.evidence if e.evidence_id in added)


def test_code_examples_do_not_force_acceptance_or_weaken_citation_gate(tmp_path):
    history, facts, patterns = _repeated_history(tmp_path)
    candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
    candidate = next(c for c in candidates.candidates if c.kind == ChangePatternKind.FILE_CO_CHANGE)

    def reject(request):
        return SemanticDecisionResponse(
            request.repository_id, request.candidate_id,
            SemanticDecisionKind.INSUFFICIENT_EVIDENCE, "Current code is not a historical rule",
            (request.evidence[0].evidence_id,), 0.8, request.support, None,
        )

    result = SemanticWorkflowAbstracter().abstract(
        history, facts, candidates, FakeModel(reject), selected_candidates=(candidate,),
    )
    assert not result.workflows and len(result.rejections) == 1
    request = build_semantic_workflow_request(history, facts, candidates, candidate)
    assert "not historical code" in build_semantic_workflow_prompt(request)
    bad = FakeModel(lambda request: replace(reject(request), decision_evidence_ids=("outside",)))
    with pytest.raises(ValueError, match="unknown Candidate Evidence"):
        SemanticWorkflowAbstracter().abstract(
            history, facts, candidates, bad, selected_candidates=(candidate,),
        )
