"""Evidence-bound symbol names used only to explain rejected candidates."""

import re


def cited_rejection_symbols(evidence, commits, evidence_ids, file_token):
    """Permit exact class members from cited logs, never file names or drafts."""
    cited = set(evidence_ids)
    commit_ids = {
        item.commit_id for item in evidence
        if item.evidence_id in cited and item.provenance == "git_log"
    }
    return frozenset(
        symbol
        for commit in commits if commit.commit_id in commit_ids
        for symbol in re.findall(
            r"\b[A-Z][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*\b", commit.message or ""
        )
        if not file_token.fullmatch(symbol)
    )
