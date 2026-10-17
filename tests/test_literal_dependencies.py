"""Bounded, static literal context in analyzer-produced implementation Evidence."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from skillforge.repository.literal_dependencies import MAX_LITERAL_CONTEXT_BYTES
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_code_evidence import (
    MAX_CODE_CONTEXT_BYTES,
    MAX_IMPLEMENTATION_BYTES,
    WorkflowCodeEvidence,
    _reference_size,
)


def _analyze(root: Path):
    facts = PythonStaticAnalyzer().analyze(root)
    implementations = {
        item.evidence_id: item
        for item in facts.evidence
        if item.description.startswith("Python implementation ")
    }
    return facts, implementations


def _selected_body(root: Path, module: str, function_name: str):
    facts, implementations = _analyze(root)
    selected = WorkflowCodeEvidence(facts).select((module,))
    symbol = next(
        symbol
        for analysis in facts.modules
        for symbol in analysis.symbols
        if symbol.qualified_name == f"{module}.{function_name}"
    )
    body_id = next(key for key in selected if key in symbol.evidence_ids)
    return symbol, implementations[body_id], selected, implementations


def _literal_context(body: str) -> str:
    marker = "# Bounded literal observations; dynamic changes unknown."
    if marker not in body:
        return ""
    return body[body.index(marker) :]


def _literal_assignments(context: str):
    if not context:
        return {}
    result = {}
    for node in ast.parse(context).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = ast.literal_eval(node.value)
    return result


def test_static_evidence_carries_complete_referenced_tuples_and_omits_unrelated_values(tmp_path):
    source = '''
POLICIES = ("review", "hold", "release")
LIMITS = (2, 5, 8)
UNRELATED = ("do not include", "unreferenced")

def choose(value: str) -> str:
    if value in POLICIES:
        return str(LIMITS[0])
    return "none"
'''
    (tmp_path / "policy.py").write_text(source, encoding="utf-8")

    symbol, item, selected, implementations = _selected_body(tmp_path, "policy", "choose")
    body = item.description.partition(": ")[2]
    literal_context = _literal_context(body)
    definitions = _literal_assignments(literal_context)

    assert selected == (item.evidence_id,)
    assert definitions["POLICIES"] == ("review", "hold", "release")
    assert definitions["LIMITS"] == (2, 5, 8)
    assert set(definitions) == {"POLICIES", "LIMITS"}
    assert "UNRELATED" not in literal_context
    assert "do not include" not in item.description
    assert len(literal_context.encode("utf-8")) <= MAX_LITERAL_CONTEXT_BYTES
    assert _reference_size(item) <= MAX_IMPLEMENTATION_BYTES
    assert sum(_reference_size(implementations[key]) for key in selected) <= MAX_CODE_CONTEXT_BYTES
    assert ast.parse(body).body[0].name == "choose"
    assert symbol.qualified_name == "policy.choose"


def test_literal_context_keeps_whole_tuples_and_total_selected_context_within_budget(tmp_path):
    for index in range(8):
        module = tmp_path / f"policy_{index}.py"
        words = ", ".join(repr(f"value-{index}-{part}-" + "x" * 40) for part in range(5))
        module.write_text(
            f"RULES = ({words},)\n\n"
            "def choose():\n"
            "    return RULES\n",
            encoding="utf-8",
        )

    facts, implementations = _analyze(tmp_path)
    code = WorkflowCodeEvidence(facts)
    selected = code.select(tuple(module.module for module in facts.modules))

    assert selected
    assert len(selected) < 8  # The global context limit still constrains coverage.
    assert sum(_reference_size(implementations[key]) for key in selected) <= MAX_CODE_CONTEXT_BYTES
    for key in selected:
        item = implementations[key]
        body = item.description.partition(": ")[2]
        context = _literal_context(body)
        # A tuple is either quoted in full or omitted; its values are never clipped.
        assert context
        definitions = _literal_assignments(context)
        index = Path(item.location.path).stem.removeprefix("policy_")
        assert definitions["RULES"] == tuple(
            f"value-{index}-{part}-" + "x" * 40 for part in range(5)
        )
        assert len(context.encode("utf-8")) <= MAX_LITERAL_CONTEXT_BYTES
        assert _reference_size(item) <= MAX_IMPLEMENTATION_BYTES


@pytest.mark.parametrize(
    "source",
    [
        "RULES = (\"module\",)\ndef choose(RULES):\n    return RULES\n",
        "RULES = (\"module\",)\ndef choose():\n    RULES = (\"local\",)\n    return RULES\n",
        "RULES = (\"module\",)\ndef choose():\n    import package as RULES\n    return RULES\n",
        (
            "RULES = (\"module\",)\ndef choose():\n    try:\n"
            "        raise ValueError()\n    except ValueError as RULES:\n"
            "        return RULES\n"
        ),
        "RULES = (\"module\",)\ndef choose():\n    global RULES\n    return RULES\n",
        "RULES = (\"first\",)\nRULES += (\"later\",)\ndef choose():\n    return RULES\n",
        "RULES = make_rules()\ndef choose():\n    return RULES\n",
        "RULES = [\"mutable\"]\ndef choose():\n    return RULES\n",
        "API_TOKEN = (\"sk-example-secret\",)\ndef choose():\n    return API_TOKEN\n",
        "RULES = (\"first\",)\nexec('RULES = ()')\ndef choose():\n    return RULES\n",
        "RULES = (\"first\",)\nglobals()['RULES'] = ()\ndef choose():\n    return RULES\n",
    ],
)
def test_ambiguous_dynamic_mutable_or_sensitive_bindings_are_omitted(tmp_path, source):
    (tmp_path / "policy.py").write_text(source, encoding="utf-8")
    facts, implementations = _analyze(tmp_path)
    selected = WorkflowCodeEvidence(facts).select(("policy",))
    item = implementations[selected[0]]
    implementation = item.description.partition(": ")[2]
    context = _literal_context(implementation)

    assert context == ""


def test_referenced_literal_value_changes_body_evidence_but_not_symbol_identity(tmp_path):
    path = tmp_path / "policy.py"
    path.write_text(
        'RULES = ("alpha", "bravo")\n\ndef choose():\n    return RULES\n',
        encoding="utf-8",
    )
    before_symbol, before_body, _, _ = _selected_body(tmp_path, "policy", "choose")

    path.write_text(
        'RULES = ("other", "bravo")\n\ndef choose():\n    return RULES\n',
        encoding="utf-8",
    )
    after_symbol, after_body, _, _ = _selected_body(tmp_path, "policy", "choose")

    assert before_symbol.symbol_id == after_symbol.symbol_id
    assert before_body.evidence_id != after_body.evidence_id
    before_context = _literal_assignments(
        _literal_context(before_body.description.partition(": ")[2])
    )
    after_context = _literal_assignments(
        _literal_context(after_body.description.partition(": ")[2])
    )
    assert before_context["RULES"] == ("alpha", "bravo")
    assert after_context["RULES"] == ("other", "bravo")
