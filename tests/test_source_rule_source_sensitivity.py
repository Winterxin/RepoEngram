from __future__ import annotations

import ast
import json

from skillforge.domain.project_dna import ProjectScope, ProjectScopeKind
from skillforge.integration.source_rule_sources import _file_units


def test_tokenization_functions_remain_eligible_and_credential_values_do_not_escape(tmp_path):
    source = '''
def _canonical_token(value):
    token = str(value).strip().lower()
    return token

def _metric_tokens(value):
    tokens = {str(value), f"{value:.1%}"}
    return tokens

def build_scoped_evidence_rows(rows):
    metric_tokens = set()
    return [{"metric_tokens": metric_tokens}]

def violates_scoped_evidence(text, rows):
    tokens = {"0", "0.0"}
    return any(token in text for token in tokens)

def assigned_token():
    access_token = "assigned-canary-71"
    return access_token

def annotated_token():
    token_value: str = "annotated-canary-82"
    return token_value

def camel_case_token():
    authToken = "camel-canary-83"
    return authToken

def subscript_token():
    config["token"] = "subscript-canary-84"
    return config

def default_token(token="default-canary-93"):
    return token

def keyword_token():
    return client(access_token="keyword-canary-14")

def mapping_token():
    return {"refresh_token": "mapping-canary-25"}

def get_token():
    return "getter-canary-36"

def comment_token():
    # token: comment-canary-47
    return "ordinary result"

def password_guard():
    return bool(password_value)
'''
    path = tmp_path / "guard.py"
    path.write_text(source, encoding="utf-8")
    units = _file_units("guard.py", path.read_bytes(), ".py")
    safe_symbols = {symbol for symbol, _node, _text in units.functions}

    assert {
        "_canonical_token",
        "_metric_tokens",
        "build_scoped_evidence_rows",
        "violates_scoped_evidence",
    } <= safe_symbols
    assert not {
        "assigned_token", "annotated_token", "camel_case_token", "subscript_token",
        "default_token", "keyword_token",
        "mapping_token", "get_token", "comment_token", "password_guard",
    } & safe_symbols

    # Exercise the serialized source channel used to build the model request.
    from skillforge.integration.source_rule_distillation import _chunks

    selected, _hashes = _chunks(
        tmp_path, ProjectScope(ProjectScopeKind.FILE, ("guard.py",))
    )
    payload = json.dumps(selected, ensure_ascii=False)
    for canary in (
        "assigned-canary-71", "annotated-canary-82", "camel-canary-83",
        "subscript-canary-84", "default-canary-93",
        "keyword-canary-14", "mapping-canary-25", "getter-canary-36",
        "comment-canary-47",
    ):
        assert canary not in payload


def test_legacy_sensitive_predicate_matches_prechange_lexical_rule():
    from skillforge.integration.source_rule_sources import _legacy_sensitive

    def prechange_predicate(node: ast.AST) -> bool:
        import re

        sensitive = re.compile(
            r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)",
            re.I,
        )
        return any(
            (isinstance(item, ast.Name) and sensitive.search(item.id))
            or (isinstance(item, ast.Attribute) and sensitive.search(item.attr))
            or (isinstance(item, ast.Constant) and isinstance(item.value, str)
                and sensitive.search(item.value))
            for item in ast.walk(node)
        )

    samples = (
        "def f(token): return token",
        "def f(value): return tokenize(value)",
        "def f(): return 'ordinary text'",
        "def f(): return 'hardcoded token value'",
        "def f(): return password_value",
    )
    for source in samples:
        node = ast.parse(source).body[0]
        assert _legacy_sensitive(node) == prechange_predicate(node)
