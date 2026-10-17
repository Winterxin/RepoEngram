"""Historical call sites may cite bounded, whole current helper definitions."""

from dataclasses import replace

from test_git_history import _write

from skillforge.domain.evidence import EvidenceType, SourceLocation
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_code_evidence import (
    MAX_CODE_CONTEXT_BYTES,
    MAX_IMPLEMENTATION_BYTES,
    MAX_PATCH_HELPERS,
    MAX_PATCH_IMPLEMENTATION_BYTES,
    WorkflowCodeEvidence,
    _reference_size,
)


def _patch(facts, body, path="policy.py"):
    return replace(facts.evidence[0], kind=EvidenceType.GIT,
                   location=SourceLocation(path, "a" * 40, 1), provenance="git_patch",
                   description='{"path":"' + path + '"}\n@@ -1,1 +1,1 @@\n' + body)


def _items(facts, selected):
    return [item for item in facts.evidence if item.evidence_id in selected]


def test_condition_and_exception_helpers_keep_original_whole_source(tmp_path):
    _write(tmp_path, "policy.py", """def eligible(mapping, message):
    if available(mapping):
        return True
    return exception(message)

def available(mapping):
    if mapping is None:
        return True
    return 'enabled' in mapping

def exception(message):
    return message.startswith('unavailable')
""")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    selected = WorkflowCodeEvidence(facts).select_for_patches([
        _patch(facts, "+    if eligible(mapping, message):\n+        return message\n")])
    items = _items(facts, selected)
    assert len(items) == 3
    assert all(item.provenance == "python_ast" and item.location.commit is None for item in items)
    assert all(_reference_size(item) <= MAX_IMPLEMENTATION_BYTES for item in items)
    assert sum(_reference_size(item) for item in items) <= MAX_CODE_CONTEXT_BYTES
    assert any("if mapping is None:" in item.description for item in items)
    assert any("message.startswith('unavailable')" in item.description for item in items)
    assert selected == WorkflowCodeEvidence(facts).select_for_patches([
        _patch(facts, "+    if eligible(mapping, message):\n+        return message\n")])


def test_removed_attribute_quoted_comment_and_other_file_calls_are_excluded(tmp_path):
    _write(tmp_path, "policy.py", "def eligible():\n    return True\n")
    _write(tmp_path, "foreign.py", "def outside():\n    return True\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    selector = WorkflowCodeEvidence(facts)
    assert not selector.select_for_patches([_patch(facts, """-    eligible()
+    obj.eligible()
+    note = 'eligible()'
+    # eligible()
+    outside()
""")])
    assert not selector.select_for_patches([_patch(facts, "+    eligible()", "foreign.py")])
    assert selector.select_for_patches([_patch(facts, "     if eligible():")])


def test_oversize_or_deep_dependency_closure_is_not_partially_selected(tmp_path):
    _write(tmp_path, "policy.py", "def root():\n    return large()\n"
           "def large():\n    return " + repr("中" * 2000) + "\n"
           "def one():\n    return two()\n"
           "def two():\n    return three()\n"
           "def three():\n    return four()\n"
           "def four():\n    return True\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    assert not WorkflowCodeEvidence(facts).select_for_patches([
        _patch(facts, "+    root()\n+    one()")])


def test_cycles_duplicates_and_large_patch_population_are_bounded(tmp_path):
    _write(tmp_path, "policy.py", "def first():\n    return second()\n"
           "def second():\n    return first()\n" + "\n".join(
               f"def action{i}():\n    return {i}\n" for i in range(20)))
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    patch = _patch(facts, "+    first()\n" + "\n".join(f"+    action{i}()" for i in range(20)))
    selected = WorkflowCodeEvidence(facts).select_for_patches([patch, patch])
    assert len(selected) <= MAX_PATCH_HELPERS
    assert sum(_reference_size(item) for item in _items(facts, selected)) <= MAX_CODE_CONTEXT_BYTES
    names = {item.description.partition(": ")[0] for item in _items(facts, selected)}
    assert {"Python implementation policy.first", "Python implementation policy.second"} <= names


def test_added_call_has_priority_over_branch_rich_unchanged_context(tmp_path):
    _write(tmp_path, "policy.py", "def added():\n    return True\n"
           "def old(value):\n    if check(value):\n        return fallback()\n    return False\n"
           "def check(value):\n    return value is not None\n"
           "def fallback():\n    return False\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    selected = WorkflowCodeEvidence(facts).select_for_patches([
        _patch(facts, "     old(value)\n+    added()")])
    names = {item.description.partition(": ")[0] for item in _items(facts, selected)}
    assert names == {"Python implementation policy.added"}


def test_added_predicate_gets_whole_evidence_before_large_cleaner(tmp_path):
    # Both fit the patch allowance individually, but cannot fit together.
    padding = "x" * 1400
    _write(tmp_path, "policy.py", f'''def cleaner(value) -> str:
    """{padding}"""
    if value:
        value = value.strip()
    if value:
        return value
    return 'empty'

def eligible(value) -> bool:
    """{padding}"""
    if value is None:
        return True
    return value.startswith('available')
''')
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    selector = WorkflowCodeEvidence(facts)
    functions = selector.local_functions["policy.py"]
    assert all(MAX_IMPLEMENTATION_BYTES < row[1] <= MAX_PATCH_IMPLEMENTATION_BYTES
               for row in functions.values())
    assert sum(row[1] for row in functions.values()) > MAX_CODE_CONTEXT_BYTES
    assert selector.select(["policy"]) == ()  # Ordinary sampling stays unchanged.
    selected = selector.select_for_patches([_patch(
        facts, "+    value = cleaner(value)\n+    if other({'key': value}) and eligible(value):")])
    assert selected == (functions["eligible"][0],)
    item, = _items(facts, selected)
    assert "if value is None:" in item.description
    assert item.description.endswith("return value.startswith('available')")
    assert selector.select_for_patches([_patch(
        facts, "+    cleaner(value)\n     if eligible(value):")]) == (functions["cleaner"][0],)
    in_body = _patch(facts, "+    cleaner(value)\n+    if value: eligible(value)")
    assert selector.select_for_patches([in_body]) == (functions["cleaner"][0],)


def test_added_branch_without_local_call_gets_one_current_same_file_predicate(tmp_path):
    _write(tmp_path, "policy.py", "def known_exception(value) -> bool:\n"
           "    if value is None:\n        return True\n    return False\n"
           "def plain() -> bool:\n    return True\n")
    _write(tmp_path, "foreign.py", "def unrelated(value) -> bool:\n"
           "    if value:\n        return True\n    return False\n")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    selector = WorkflowCodeEvidence(facts)
    selected = selector.select_for_patches([_patch(facts, "+    if obj.reject(value):\n")])
    item, = _items(facts, selected)
    assert item.description.startswith("Python implementation policy.known_exception: ")
    assert item.location.path == "policy.py" and item.location.commit is None
    assert not selector.select_for_patches([_patch(facts, "+    note = 'known_exception()'")])
    assert not selector.select_for_patches([_patch(facts, "     if obj.reject(value):")])
    # A usable direct call wins; unrelated predicates are not added as extra context.
    assert selector.select_for_patches([_patch(facts, "+    if plain():")]) == (
        selector.local_functions["policy.py"]["plain"][0],)
