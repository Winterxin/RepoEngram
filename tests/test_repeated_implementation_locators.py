from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_code_evidence import WorkflowCodeEvidence


def _write(root, name, content):
    (root / name).write_text(content, encoding="utf-8")


def _implementations(facts):
    return {item.evidence_id: item for item in facts.evidence
            if item.description.startswith("Python implementation ")}


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
