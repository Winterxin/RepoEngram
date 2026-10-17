import ast

from skillforge.integration.source_task_focus import source_task_focus


def _functions(path: str, source: str):
    tree = ast.parse(source)
    result = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append((path, node.name, node, source))
        elif isinstance(node, ast.ClassDef):
            result.extend(
                (path, f"{node.name}.{child.name}", child, source)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return result


def test_explicit_identifiers_and_qualified_suffixes_match():
    functions = _functions(
        "src/api.py",
        "class Response:\n"
        "    def links(self): return []\n"
        "def update_record(): return None\n",
    )
    focus = source_task_focus(functions, "Update `Response.links` and update_record")
    assert focus.anchors == {
        ("src/api.py", "Response.links"), ("src/api.py", "update_record"),
    }


def test_plain_prose_words_and_partial_identifiers_do_not_match():
    functions = _functions(
        "src/utils.py",
        "def get(): return None\n"
        "def save_record(): return None\n"
        "def save(): return None\n",
    )
    focus = source_task_focus(functions, "get values and save_record_extra later")
    assert focus.anchors == frozenset()


def test_plain_lowercase_name_requires_code_shaped_reference():
    functions = _functions("src/utils.py", "def get(): return None\n")
    assert source_task_focus(functions, "get the result").anchors == frozenset()
    assert source_task_focus(functions, "call get() now").anchors == {("src/utils.py", "get")}
    assert source_task_focus(functions, "update `get`").anchors == {("src/utils.py", "get")}


def test_unmatched_qualified_reference_does_not_match_its_bare_leaf():
    functions = _functions("src/api.py", "class Actual:\n    def update_record(self): pass\n")
    assert not source_task_focus(functions, "Fix Different.update_record").anchors
    assert not source_task_focus(functions, "Fix update_record.extra").anchors
    assert source_task_focus(functions, "Fix Actual.update_record.").anchors


def test_ambiguous_basename_is_ignored_but_unique_qualified_symbol_matches():
    functions = _functions(
        "src/api.py",
        "class Request:\n    def links(self): return []\n"
        "class Response:\n    def links(self): return []\n",
    )
    assert source_task_focus(functions, "links() please").anchors == frozenset()
    assert source_task_focus(functions, "Response.links").anchors == {
        ("src/api.py", "Response.links"),
    }


def test_direct_call_neighbors_are_same_file_unambiguous_and_one_hop():
    functions = _functions(
        "src/flow.py",
        "def prepare(): return None\n"
        "def target():\n    prepare()\n    finish()\n"
        "def finish(): return None\n"
        "def beyond():\n    finish()\n",
    )
    focus = source_task_focus(functions, "target()")
    assert focus.anchors == {("src/flow.py", "target")}
    assert focus.related == {("src/flow.py", "prepare"), ("src/flow.py", "finish")}
    assert focus.selected == focus.anchors | focus.related
    assert ("src/flow.py", "beyond") not in focus.selected


def test_ambiguous_bare_call_resolution_falls_back_to_anchors_only():
    functions = _functions(
        "src/flow.py",
        "def helper(): return None\n"
        "def helper(): return None\n"
        "def target(): helper()\n",
    )
    focus = source_task_focus(functions, "target()")
    assert focus.anchors == {("src/flow.py", "target")}
    assert focus.related == frozenset()


def test_parameters_and_locals_shadow_bare_call_targets():
    functions = _functions(
        "src/flow.py",
        "def helper(): return None\n"
        "def parameter_shadow(helper): helper()\n"
        "def local_shadow():\n    helper = object()\n    helper()\n",
    )
    for anchor in ("parameter_shadow", "local_shadow"):
        focus = source_task_focus(functions, f"{anchor}()")
        assert focus.anchors == {("src/flow.py", anchor)}
        assert focus.related == frozenset()


def test_self_calls_resolve_only_within_the_current_class():
    functions = _functions(
        "src/classes.py",
        "class A:\n"
        "    def entry(self): self.work()\n"
        "    def work(self): return None\n"
        "class B:\n"
        "    def work(self): return None\n"
        "    def other(self, obj): obj.work()\n",
    )
    focus = source_task_focus(functions, "A.entry")
    assert focus.anchors == {("src/classes.py", "A.entry")}
    assert focus.related == {("src/classes.py", "A.work")}


def test_task_is_bounded_to_8_kibibytes():
    functions = _functions("src/flow.py", "def target(): return None\n")
    assert source_task_focus(functions, ("x" * 8192) + " target()").anchors == frozenset()
    assert source_task_focus(functions, "target()" + ("x" * 8192)).anchors == {
        ("src/flow.py", "target"),
    }


def test_more_than_eight_anchors_falls_back_to_empty_focus():
    source = "\n".join(f"def action_{i}(): return None" for i in range(9))
    functions = _functions("src/actions.py", source)
    task = " ".join(f"action_{i}" for i in range(9))
    focus = source_task_focus(functions, task)
    assert focus.anchors == frozenset()
    assert focus.related == frozenset()


def test_unrelated_domain_words_do_not_trigger_semantic_selection():
    functions = _functions(
        "src/operations.py",
        "def apply_reservation_status(value): return value\n"
        "def calculate_invoice_tax(value): return value\n",
    )
    focus = source_task_focus(
        functions,
        "Preserve booking status when a reservation is cancelled or confirmed",
    )
    assert focus.selected == frozenset()
