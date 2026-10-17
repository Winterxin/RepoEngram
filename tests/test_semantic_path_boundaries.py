"""Prose punctuation does not invent paths or truncate filename extensions."""

import pytest

from skillforge.repository.semantic_workflows import _validate_text_claims


@pytest.mark.parametrize("template", [
    "Observed {}.", "Observed {}, then continued.", "Observed {}; then continued.",
    "Observed {}: local scope.", "Observed ({}).", 'Observed "{}".',
    "Observed '{}'.", "Observed `{}`.",
])
@pytest.mark.parametrize("path", ["foo/package", "foo/bar.py", "foo/.settings"])
def test_prose_boundaries_preserve_known_path(template, path):
    _validate_text_claims((template.format(path),), set(), set(), {path})


@pytest.mark.parametrize("path", ["foo/bar.py", "foo/bar.py.extra", "foo/name."])
def test_exact_dotted_path_is_preserved(path):
    _validate_text_claims((f"Observed {path}",), set(), set(), {path})


@pytest.mark.parametrize("claim", [
    "Observed unknown/module.", "Observed foo/bar.py.extra.", "Observed foo/bar.py.",
])
def test_unknown_paths_and_extensions_remain_rejected(claim):
    with pytest.raises(ValueError, match="unknown path"):
        _validate_text_claims((claim,), set(), set(), {"foo/bar"})


def test_adjacent_paths_and_punctuation_are_separately_validated():
    _validate_text_claims(("Observed foo/a.py., foo/b.py; foo/c: local.",),
                          set(), set(), {"foo/a.py", "foo/b.py", "foo/c"})
