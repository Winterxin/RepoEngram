"""Evidence-bound prose references, distinct from structural path authority."""

import re

_CLASS_MEMBER = re.compile(r"\b[A-Z][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*\b")
_CLASS_MEMBER_GROUP = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*)\.([a-z_][A-Za-z0-9_]*(?:/[a-z_][A-Za-z0-9_]*)+)\b"
)
_IMPLEMENTATION_HEADING = re.compile(
    r"^Python implementation "
    r"(?P<symbol>(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*):"
)
_COMMIT_MESSAGE_PROVENANCE = frozenset({"git_log", "git_diff", "git_patch"})


def _class_member_alias(symbol):
    """Return a short class.member spelling only for an exact qualified symbol."""
    parts = symbol.rsplit(".", 2)
    if len(parts) != 3:
        return None
    class_name, member = parts[-2:]
    if (re.fullmatch(r"[A-Z][A-Za-z0-9_]*", class_name)
            and re.fullmatch(r"[a-z_][A-Za-z0-9_]*", member)):
        return f"{class_name}.{member}"
    return None

def cited_prose_filenames(evidence, commits, evidence_ids, file_token):
    """Recognize mentioned filenames without authorizing paths/scopes to use them."""
    cited = set(evidence_ids)
    commit_ids = {item.commit_id for item in evidence
                  if item.evidence_id in cited and item.provenance == 'git_log'}
    texts = [item.description for item in evidence if item.evidence_id in cited]
    texts.extend(commit.message for commit in commits if commit.commit_id in commit_ids)
    return frozenset(token.rstrip('.') for text in texts
                     for token in file_token.findall(text or '')
                     if '/' not in token and '\\' not in token and '..' not in token)


def cited_rejection_symbols(evidence, commits, evidence_ids, file_token):
    """Permit exact method names grounded in cited history or implementation Evidence.

    A cited diff/patch/log can ground only its own commit message. A cited Python
    implementation can ground only the qualified symbol in its Evidence heading,
    never names appearing in the code body. These prose references do not grant
    module, path, or scope-selector authority.
    """
    cited = set(evidence_ids)
    commit_ids = {
        item.commit_id for item in evidence
        if item.evidence_id in cited
        and item.provenance in _COMMIT_MESSAGE_PROVENANCE
        and item.commit_id is not None
    }
    symbols = set()
    for commit in commits:
        if commit.commit_id not in commit_ids:
            continue
        symbols.update(
            symbol for symbol in _CLASS_MEMBER.findall(commit.message or "")
            if not file_token.fullmatch(symbol)
        )
        for class_name, members in _CLASS_MEMBER_GROUP.findall(commit.message or ""):
            symbols.update(f"{class_name}.{member}" for member in members.split('/'))

    for item in evidence:
        if (item.evidence_id not in cited or item.kind != "source"
                or item.provenance != "python_ast" or item.commit_id is not None):
            continue
        match = _IMPLEMENTATION_HEADING.match(item.description or "")
        if match is None:
            continue
        symbol = match.group("symbol")
        if not file_token.fullmatch(symbol):
            symbols.add(symbol)
        alias = _class_member_alias(symbol)
        if alias is not None and not file_token.fullmatch(alias):
            symbols.add(alias)
    return frozenset(symbols)
