"""Repeated syntax remains separate source observations with exact locations."""

import pytest
from test_static_testing import _testing, fixture

from skillforge.repository.static_analysis import PythonStaticAnalyzer


@pytest.mark.parametrize("raises", [False, True])
def test_repeated_identical_test_calls_preserve_every_location(tmp_path, raises):
    fixture(tmp_path)
    path = tmp_path / "tests/test_behavior.py"
    text = path.read_text()
    if raises:
        text = "import pytest\n" + text
        for index in range(2):
            assertion = f"  self.assertEqual(self.subject.change_{index}('x'), 'x')\n"
            replacement = (
                f"  with pytest.raises(ValueError):\n   self.subject.change_{index}('x')\n"
            )
            text = text.replace(assertion, replacement * 2)
    else:
        for index in range(2):
            assertion = f"  self.assertEqual(self.subject.change_{index}('x'), 'x')\n"
            text = text.replace(assertion, assertion * 2)
    path.write_text(text)
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    (candidate,) = _testing(facts)
    assert len(candidate.test_linkages) == 4
    evidence = {item.evidence_id: item for item in facts.evidence}
    for link in candidate.test_linkages:
        for prefix, line in (
            ("Python public test call", link.call_line),
            ("Python behavior assertion", link.assertion_line),
        ):
            assert any(
                evidence[key].description.startswith(prefix) and evidence[key].location.line == line
                for key in link.evidence_ids
            )
    assert facts == PythonStaticAnalyzer().analyze(tmp_path)
