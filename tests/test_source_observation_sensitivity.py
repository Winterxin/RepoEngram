"""Sensitive statements cannot suppress safe siblings or escape via context."""

from test_source_constraints import _export, _files

from skillforge.integration.source_observation_selection import COLD_PACKAGE_BYTES, package_sizes


def _source(tmp_path, text):
    path = tmp_path / "src/example.py"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")


def test_tokenization_in_same_module_does_not_hide_business_observations(tmp_path):
    _source(tmp_path,
            "def split_words(text):\n"
            "    tokens = text.split()\n"
            "    return tokens\n"
            "def normalize(value):\n"
            "    if value is None:\n"
            "        return 0\n"
            "    return value\n")
    plan = _export(tmp_path)
    text = _files(plan)["SKILL.md"]
    assert "if value is None" in text and "return 0" in text
    # Ambiguous vocabulary remains conservatively excluded, not whitelisted.
    assert "tokens" not in text
    assert all(size <= COLD_PACKAGE_BYTES for size in package_sizes(plan))


def test_sensitive_effects_contexts_and_comments_are_not_exported(tmp_path):
    _source(tmp_path,
            "password = 'effect-canary'\n"
            "def get_secret():\n"
            "    return 'function-canary'\n"
            "def execute(items, connection, api_key):\n"
            "    if api_key == 'predicate-canary':\n"
            "        return 'branch-canary'\n"
            "    for token in items:\n"
            "        publish('loop-canary')\n"
            "    with connection.secret('manager-canary'):\n"
            "        publish('body-canary')\n"
            "    try:\n"
            "        run()\n"
            "    except ValueError as credential:\n"
            "        publish('handler-canary')\n"
            "    value = dict(\n"
            "        password='multiline-canary',\n"
            "    )\n"
            "    result = 1  # password: comment-canary\n"
            "    return 0\n")
    text = _files(_export(tmp_path))["SKILL.md"]
    assert "return 0" in text and "run()" in text
    assert "canary" not in text
    for word in ("password", "secret", "credential", "api_key", "token"):
        assert word not in text


def test_module_with_only_sensitive_observations_still_has_no_export(tmp_path):
    _source(tmp_path, "api_key = 'do-not-export'\ndef get_secret():\n    return 'hidden'\n")
    assert _export(tmp_path) is None


def test_newly_eligible_file_cannot_create_a_larger_legacy_allowance(tmp_path):
    _source(tmp_path, "password = 'hidden'\n" + "".join(
        f"def normalize_{i}(value):\n    if value == {i}:\n        return {i}\n"
        for i in range(100)
    ))
    plan = _export(tmp_path)
    assert plan is not None
    assert all(size <= COLD_PACKAGE_BYTES for size in package_sizes(plan))
    assert "hidden" not in _files(plan)["SKILL.md"]
