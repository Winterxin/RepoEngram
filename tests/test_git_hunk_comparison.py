from types import SimpleNamespace

from skillforge.repository.git_hunk_comparison import (
    HunkCandidate,
    compact_context_requests,
    descendants_first_order,
    function_anchor,
    select_compared_hunks,
)


def _cs(commit_id: str, parent: str, authored_at: int):
    return SimpleNamespace(
        commit_id=commit_id, parent_ids=(parent,), authored_at=authored_at,
    )


def _hunk(anchor: str | None, *, size: int = 100, priority: int = 1, order: int = 0):
    description = "{}\n@@ -10,5 +10,5 @@\n"
    return HunkCandidate(SimpleNamespace(description=description), size, priority, order, anchor)


def _range_hunk(
    anchor: str | None, *, size: int, old_start: int, old_count: int,
    new_start: int, new_count: int, priority: int = 1, order: int = 0,
):
    description = (
        "{}\n"
        f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n"
    )
    return HunkCandidate(SimpleNamespace(description=description), size, priority, order, anchor)


def test_function_anchor_requires_a_unique_function_context():
    assert function_anchor("@@ -8,3 +8,3 @@ def transform(value):\n x\n") == "def:transform"
    assert function_anchor("@@ -8,3 +8,3 @@\n def first():\n def second():\n") is None
    assert function_anchor('@@ -8 +8 @@\n+    note = "def fake("\n') is None


def test_same_function_is_aligned_after_an_unrelated_hunk_without_using_timestamps():
    old, new = "a" * 40, "b" * 40
    changesets = {
        old: _cs(old, "0" * 40, authored_at=900),
        new: _cs(new, old, authored_at=10),
    }
    requests = [(old, "module.py"), (new, "module.py")]
    assert descendants_first_order(requests, changesets) == [requests[1], requests[0]]

    selected, blocked = select_compared_hunks(
        requests, changesets,
        {
            requests[0]: (_hunk("def:convert", priority=0),),
            requests[1]: (
                _hunk("def:helper", priority=0, order=0),
                _hunk("def:convert", priority=1, order=1),
            ),
        },
        {}, 1536,
    )
    assert not blocked
    assert selected[requests[0]].anchor == "def:convert"
    assert selected[requests[1]].anchor == "def:convert"


def test_oversized_later_same_function_blocks_the_old_observation():
    old, new = "a" * 40, "b" * 40
    changesets = {old: _cs(old, "0" * 40, 1), new: _cs(new, old, 2)}
    old_key, new_key = (old, "module.py"), (new, "module.py")
    selected, blocked = select_compared_hunks(
        [old_key, new_key], changesets,
        {old_key: (_hunk("def:convert"),),
         new_key: (_hunk("def:convert", size=1700),)},
        {}, 1536,
    )
    assert old_key not in selected
    assert old_key in blocked


def test_compact_context_retry_requires_a_unique_oversized_descendant():
    old, new = "a" * 40, "b" * 40
    changesets = {old: _cs(old, "0" * 40, 1), new: _cs(new, old, 2)}
    old_key, new_key = (old, "module.py"), (new, "module.py")
    assert compact_context_requests(
        [old_key, new_key], changesets,
        {old_key: (_hunk("def:convert"),),
         new_key: (_hunk("def:convert", size=1700),)},
        1536,
    ) == [new_key]
    selected, blocked = select_compared_hunks(
        [old_key, new_key], changesets,
        {old_key: (_hunk("def:convert"),),
         new_key: (_hunk("def:convert", size=1700),)},
        {new_key: "compact_context_read_budget"}, 1536,
    )
    assert old_key not in selected and old_key in blocked

    # Sibling branches have no unique descendant comparison target.
    sibling = "c" * 40
    sibling_changesets = {
        old: _cs(old, "0" * 40, 1),
        new: _cs(new, old, 2),
        sibling: _cs(sibling, old, 3),
    }
    sibling_key = (sibling, "module.py")
    assert compact_context_requests(
        [old_key, new_key, sibling_key], sibling_changesets,
        {old_key: (_hunk("def:convert"),),
         new_key: (_hunk("def:convert", size=1700),),
         sibling_key: (_hunk("def:convert", size=1700),)},
        1536,
    ) == []


def test_same_function_hunks_align_by_overlapping_parent_line_ranges():
    old, new = "a" * 40, "b" * 40
    changesets = {old: _cs(old, "0" * 40, 1), new: _cs(new, old, 2)}
    old_key, new_key = (old, "module.py"), (new, "module.py")
    old_hunk = _range_hunk(
        "def:transform", size=200, old_start=90, old_count=10,
        new_start=90, new_count=10,
    )
    overlapping_oversize = _range_hunk(
        "def:transform", size=1700, old_start=94, old_count=5,
        new_start=94, new_count=5, priority=0,
    )
    later_fitting = _range_hunk(
        "def:transform", size=800, old_start=150, old_count=5,
        new_start=150, new_count=5,
    )
    candidates = {old_key: (old_hunk,), new_key: (overlapping_oversize, later_fitting)}
    assert compact_context_requests([old_key, new_key], changesets, candidates, 1536) == [new_key]
    selected, blocked = select_compared_hunks(
        [old_key, new_key], changesets, candidates, {}, 1536,
    )
    assert old_key in blocked and old_key not in selected
    assert selected[new_key] is later_fitting

    nonoverlap = _range_hunk(
        "def:transform", size=1700, old_start=150, old_count=5,
        new_start=150, new_count=5,
    )
    nonmatching = {old_key: (old_hunk,), new_key: (nonoverlap, later_fitting)}
    assert compact_context_requests([old_key, new_key], changesets, nonmatching, 1536) == []
    selected, blocked = select_compared_hunks(
        [old_key, new_key], changesets, nonmatching, {}, 1536,
    )
    assert old_key in selected and not blocked


def test_sibling_branches_are_not_compared_or_suppressed():
    root, left, right = "0" * 40, "a" * 40, "b" * 40
    changesets = {
        left: _cs(left, root, 2),
        right: _cs(right, root, 1),
    }
    left_key, right_key = (left, "module.py"), (right, "module.py")
    selected, blocked = select_compared_hunks(
        [left_key, right_key], changesets,
        {left_key: (_hunk("def:convert"),),
         right_key: (_hunk("def:convert", size=1700),)},
        {}, 1536,
    )
    assert left_key in selected
    assert not blocked


def test_unread_newer_same_path_blocks_an_old_only_observation():
    old, new = "a" * 40, "b" * 40
    changesets = {old: _cs(old, "0" * 40, 1), new: _cs(new, old, 2)}
    old_key, new_key = (old, "module.py"), (new, "module.py")
    selected, blocked = select_compared_hunks(
        [old_key, new_key], changesets,
        {old_key: (_hunk("def:convert"),)},
        {new_key: "git_read_budget"}, 1536,
    )
    assert old_key not in selected
    assert old_key in blocked


def test_unread_latest_descendant_blocks_even_when_an_intermediate_matches():
    old, middle, new = "a" * 40, "b" * 40, "c" * 40
    changesets = {
        old: _cs(old, "0" * 40, 1),
        middle: _cs(middle, old, 2),
        new: _cs(new, middle, 3),
    }
    old_key = (old, "module.py")
    middle_key = (middle, "module.py")
    new_key = (new, "module.py")
    selected, blocked = select_compared_hunks(
        [old_key, middle_key, new_key], changesets,
        {old_key: (_hunk("def:convert"),),
         middle_key: (_hunk("def:convert"),)},
        {new_key: "git_read_budget"}, 1536,
    )
    assert old_key not in selected
    assert middle_key not in selected
    assert blocked == {old_key, middle_key}
