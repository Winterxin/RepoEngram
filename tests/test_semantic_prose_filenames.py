from types import SimpleNamespace as Item

import pytest

from skillforge.repository.semantic_references import cited_prose_filenames
from skillforge.repository.semantic_workflows import _PATH_TOKEN, _validate_text_claims


def test_only_cited_bare_filenames_can_be_mentioned_without_path_authority():
    evidence = [Item(evidence_id='a', commit_id='c', provenance='git_log', description='Commit'),
                Item(evidence_id='b', commit_id=None, provenance='python_ast',
                     description='Read "settings.toml" before parsing'),
                Item(evidence_id='other', commit_id=None, provenance='python_ast',
                     description='unrelated.ini')]
    commits = [Item(commit_id='c', message='Respect project.ini; not arbitrary/other.cfg')]
    names = cited_prose_filenames(evidence, commits, ('a', 'b'), _PATH_TOKEN)
    assert names == {'settings.toml', 'project.ini'}
    _validate_text_claims(('Respect settings.toml',), set(), set(), set(), mentioned_paths=names)
    with pytest.raises(ValueError, match='unknown path'):
        _validate_text_claims(('Read unrelated.ini',), set(), set(), set(), mentioned_paths=names)
    with pytest.raises(ValueError, match='unknown path'):
        _validate_text_claims(
            ('Read arbitrary/other.cfg',), set(), set(), set(), mentioned_paths=names,
        )


def test_uncited_commit_messages_do_not_authorize_filename_mentions():
    names = cited_prose_filenames(
        [], [Item(commit_id='c', message='settings.toml')], (), _PATH_TOKEN,
    )
    assert not names


def test_exact_cited_evidence_hash_is_not_mistaken_for_a_git_commit():
    evidence_id = '0123456789abcdef012345'
    _validate_text_claims(
        (f'Observed constraint [{evidence_id}]',), set(), set(), set(),
        cited_evidence_ids=frozenset({evidence_id}),
    )
    for unknown in ('abcdef0123456789abcdef', evidence_id[:8]):
        with pytest.raises(ValueError, match='unknown commit'):
            _validate_text_claims(
                (f'Unknown reference {unknown}',), set(), set(), set(),
                cited_evidence_ids=frozenset({evidence_id}),
            )
