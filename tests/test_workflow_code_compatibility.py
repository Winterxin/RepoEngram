"""Nonimportable example files must not prevent valid module knowledge extraction."""

from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_code_evidence import WorkflowCodeEvidence


def test_example_filename_preserves_source_facts_and_valid_module_evidence(tmp_path):
    (tmp_path / "example-case.py").write_text(
        "def example(value):\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "valid.py").write_text(
        "def transform(value):\n    return value\n", encoding="utf-8"
    )
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    example = next(module for module in facts.modules if module.path == "example-case.py")
    assert example.module is None
    assert example.symbols
    bodies = {
        item.evidence_id: item for item in facts.evidence
        if item.description.startswith("Python implementation ")
    }
    assert any(item.location.path == "example-case.py" for item in bodies.values())

    indexed = WorkflowCodeEvidence(facts)
    selected = indexed.select(("valid",))
    assert selected
    assert {bodies[key].location.path for key in selected} == {"valid.py"}
    assert "example-case.py" not in indexed.local_functions
