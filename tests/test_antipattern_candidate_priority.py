"""Bounded negative-evidence selection must not be dominated by path ordering."""

from dataclasses import replace

import pytest
import test_semantic_antipatterns as fixtures

from skillforge.repository.semantic_antipattern_evaluation import (
    select_semantic_antipattern_candidates,
)


def candidates(*specs):
    result = fixtures._candidate_result(fixtures._history(*specs))
    # Existing Git patch observations; no source scan or semantic decision.
    evidence = tuple(replace(item, provenance="git_patch", description=(
        "@@ -1,2 +1,4 @@\n+    if unavailable:\n+        raise ValueError('closed')\n"
    )) if item.provenance == "git_diff" else item for item in result.evidence)
    return replace(result, evidence=evidence)


@pytest.mark.parametrize("path", ["z_package/core.py", "a_package/core.py"])
def test_python_correction_not_starved_by_alphabetically_earlier_documents(path):
    result = candidates(
        {"paths": ("000_notes.rst",), "message": "Fix documentation example"},
        {"paths": (path,), "message": "Fix guard behavior"},
    )
    selected = select_semantic_antipattern_candidates(result, limit=1)
    assert selected[0].affected_paths == (path,)


def test_verified_revert_in_documentation_remains_stronger_than_code_hint():
    result = candidates(
        {"paths": ("z_notes.rst",), "message": "Update notes"},
        {"paths": ("a_module.py",), "message": "Fix guard behavior"},
        {"paths": ("z_notes.rst",),
         "message": f'Revert "Update notes"\n\nThis reverts commit {fixtures._commit_id(1)}.'},
    )
    assert select_semantic_antipattern_candidates(result, limit=1)[0].affected_paths == (
        "z_notes.rst",
    )


def test_distinct_correction_commits_fill_quota_before_repeated_same_event():
    result = candidates(
        {"paths": ("a.py", "b.py"), "message": "Fix one shared event"},
        {"paths": ("c.py",), "message": "Fix another event"},
    )
    before = result.to_dict()
    selected = select_semantic_antipattern_candidates(result, limit=2)
    assert tuple(item.affected_paths for item in selected) == (("a.py",), ("c.py",))
    assert select_semantic_antipattern_candidates(result, limit=2) == selected
    assert result.to_dict() == before


def test_no_patch_does_not_invent_a_code_quality_signal():
    result = fixtures._candidate_result(fixtures._history(
        {"paths": ("a_notes.rst", "z_module.py"), "message": "Fix examples"},
    ))
    assert select_semantic_antipattern_candidates(result, limit=1)[0].affected_paths == (
        "a_notes.rst",
    )
