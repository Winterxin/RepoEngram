"""Contracts for deterministic negative-signal candidate mining."""

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from skillforge.cli import main
from skillforge.domain import NegativeSignalKind
from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind, GitHistoryResult
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.repository import GitHistoryMiner, NegativePatternMiner

REPOSITORY_ID = "repo-negative-patterns-test"
BASE_TIME = datetime(2026, 9, 1, tzinfo=UTC)


def _commit_id(index):
    return f"{index:040x}"


def _history(*specs, repository_id=REPOSITORY_ID):
    changesets = []
    evidence = []
    for index, spec in enumerate(specs, start=1):
        commit_id = _commit_id(index)
        timestamp = BASE_TIME + timedelta(days=spec.get("day", index - 1))
        paths = spec.get("changes", spec.get("paths", ()))
        files = []
        evidence_ids = []
        log_id = hashlib.sha256(f"{repository_id}:{commit_id}:git_log".encode()).hexdigest()[:24]
        evidence_ids.append(log_id)
        evidence.append(Evidence(
            log_id,
            repository_id,
            EvidenceType.GIT,
            SourceLocation(".", commit_id),
            f"Observed Git commit {commit_id}",
            "git_log",
        ))
        for entry in paths:
            if isinstance(entry, str):
                path = entry
                kind = FileChangeKind.MODIFIED
                previous_path = None
            else:
                path = entry["path"]
                kind = FileChangeKind(entry.get("kind", "modified"))
                previous_path = entry.get("previous_path")
            file_evidence_ids = []
            for provenance in ("git_diff", "git_numstat"):
                evidence_id = hashlib.sha256(
                    f"{repository_id}:{commit_id}:{provenance}:{path}".encode()
                ).hexdigest()[:24]
                file_evidence_ids.append(evidence_id)
                evidence.append(Evidence(
                    evidence_id,
                    repository_id,
                    EvidenceType.GIT,
                    SourceLocation(path, commit_id),
                    f"Observed {provenance} for {path}",
                    provenance,
                ))
            files.append(FileChange(
                path=path,
                kind=kind,
                previous_path=previous_path,
                additions=1,
                deletions=1,
                evidence_ids=tuple(file_evidence_ids),
            ))
            evidence_ids.extend(file_evidence_ids)
        is_merge = spec.get("is_merge", False)
        if is_merge:
            parent_ids = (_commit_id(index + 20), _commit_id(index + 30))
        elif index > 1:
            parent_ids = (_commit_id(index - 1),)
        else:
            parent_ids = ()
        changesets.append(ChangeSet(
            repository_id=repository_id,
            commit_id=commit_id,
            committed_at=timestamp.isoformat(),
            files=tuple(files),
            evidence_ids=tuple(evidence_ids),
            parent_ids=parent_ids,
            message=spec.get("message", "Update files"),
            is_merge=is_merge,
        ))
    return GitHistoryResult(
        repository_id,
        tuple(changesets),
        tuple(sorted(evidence, key=lambda item: item.evidence_id)),
    )


def _signals(result, kind):
    return tuple(signal for signal in result.signals if signal.kind == kind)


def test_explicit_revert_message_is_only_a_message_based_hint():
    history = _history({
        "paths": ("src/parser.py",),
        "message": 'Revert "Add fast parser"',
    })

    result = NegativePatternMiner().mine(history)

    signal, = _signals(result, NegativeSignalKind.EXPLICIT_REVERT_HINT)
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    assert signal.heuristic is True
    assert signal.method.endswith("patch reversal not verified")
    assert any(
        evidence_by_id[item].provenance == "git_log" for item in signal.evidence_ids
    )
    assert not _signals(result, NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT)


def test_correction_keyword_is_case_insensitive_and_normal_message_is_not_a_hint():
    history = _history(
        {"paths": ("src/a.py",), "message": "hOtFiX: repair request parsing"},
        {"paths": ("docs/guide.md",), "message": "Update fixture documentation"},
    )

    result = NegativePatternMiner().mine(history)

    correction, = _signals(result, NegativeSignalKind.CORRECTION_MESSAGE_HINT)
    assert correction.heuristic is True
    assert correction.matched_terms == ("hotfix",)
    assert correction.commit_ids == (_commit_id(1),)
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    assert any(
        evidence_by_id[item].provenance == "git_log" for item in correction.evidence_ids
    )


@pytest.mark.parametrize(
    "message,term",
    [
        ("Fix parser", "fix"),
        ("bugfix: repair parser", "bugfix"),
        ("REGRESSION in cache", "regression"),
        ("add workaround for remote", "workaround"),
    ],
)
def test_supported_correction_words_are_reported_as_hints(message, term):
    result = NegativePatternMiner().mine(_history({"message": message}))

    signal, = _signals(result, NegativeSignalKind.CORRECTION_MESSAGE_HINT)
    assert term in signal.matched_terms
    assert signal.heuristic is True


def test_short_window_repeat_edit_uses_only_related_file_diff_and_numstat_evidence():
    history = _history(
        {"paths": ("src/repeated.py",)},
        {"paths": ("docs/unrelated.md",)},
        {"paths": ("src/repeated.py",)},
    )

    result = NegativePatternMiner(window_commits=3, window_days=4).mine(history)

    signal, = _signals(result, NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT)
    assert signal.heuristic is False
    assert signal.commit_ids == (_commit_id(1), _commit_id(3))
    assert signal.affected_paths == ("src/repeated.py",)
    by_id = {item.evidence_id: item for item in history.evidence}
    linked = [by_id[item] for item in signal.evidence_ids]
    assert {item.provenance for item in linked} == {"git_diff", "git_numstat"}
    assert {item.location.commit for item in linked} == set(signal.commit_ids)
    assert {item.location.path for item in linked} == {"src/repeated.py"}


@pytest.mark.parametrize(
    "window_commits,window_days,specs",
    [
        (2, 10, (
            {"paths": ("a.py",)},
            {"paths": ("b.py",)},
            {"paths": ("a.py",)},
        )),
        (5, 1, (
            {"paths": ("a.py",), "day": 0},
            {"paths": ("a.py",), "day": 2},
        )),
    ],
)
def test_repeated_edit_outside_either_window_is_not_reported(
    window_commits, window_days, specs
):
    assert not _signals(
        NegativePatternMiner(window_commits, window_days).mine(_history(*specs)),
        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
    )


def test_merge_changes_are_excluded_from_repeat_participants_and_window_positions():
    history = _history(
        {"paths": ("src/a.py",)},
        {"paths": ("src/a.py",), "is_merge": True},
        {"paths": ("src/a.py",)},
    )

    result = NegativePatternMiner(window_commits=2, window_days=5).mine(history)

    signal, = _signals(result, NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT)
    assert signal.commit_ids == (_commit_id(1), _commit_id(3))
    assert _commit_id(2) not in signal.commit_ids


def test_multiple_files_get_path_specific_candidates_and_combine_shared_signals():
    history = _history(
        {"paths": ("src/a.py", "src/b.py")},
        {"paths": ("src/a.py", "src/c.py"), "message": "Fix parser input"},
    )

    result = NegativePatternMiner(window_commits=3, window_days=5).mine(history)

    by_path = {candidate.affected_paths: candidate for candidate in result.candidates}
    candidate_a = by_path[("src/a.py",)]
    candidate_c = by_path[("src/c.py",)]
    assert len(candidate_a.signal_ids) == 2
    assert len(candidate_c.signal_ids) == 1
    assert ("src/b.py",) not in by_path
    assert {signal.kind for signal in result.signals} == {
        NegativeSignalKind.CORRECTION_MESSAGE_HINT,
        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
    }


def test_rename_does_not_join_old_and_new_paths_into_a_repeat():
    history = _history(
        {"paths": ("src/old.py",)},
        {"changes": ({
            "path": "src/new.py",
            "kind": "renamed",
            "previous_path": "src/old.py",
        },)},
        {"paths": ("src/new.py",)},
    )

    result = NegativePatternMiner(window_commits=3, window_days=5).mine(history)

    assert not _signals(result, NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT)


def test_revert_sha_reference_is_observed_only_when_target_is_in_history():
    message = f'Revert "Initial parser"\n\nThis reverts commit {_commit_id(1)}.\n'
    history = _history(
        {"paths": ("src/parser.py",), "message": "Initial parser"},
        {"paths": ("src/parser.py",), "message": message},
    )

    result = NegativePatternMiner().mine(history)

    reference, = _signals(result, NegativeSignalKind.REVERT_SHA_REFERENCE)
    assert reference.heuristic is False
    assert reference.referenced_commit_id == _commit_id(1)
    assert set(reference.commit_ids) == {_commit_id(1), _commit_id(2)}
    missing_target = NegativePatternMiner().mine(_history({
        "paths": ("src/parser.py",),
        "message": f'This reverts commit {_commit_id(99)}.',
    }))
    assert not _signals(missing_target, NegativeSignalKind.REVERT_SHA_REFERENCE)
    assert len(_signals(missing_target, NegativeSignalKind.EXPLICIT_REVERT_HINT)) == 1


def test_revert_reference_counts_involved_commits_without_claiming_pattern_support():
    history = _history(
        {"paths": ("src/parser.py",), "message": "Initial parser"},
        {
            "paths": ("src/parser.py",),
            "message": f"This reverts commit {_commit_id(1)}.",
        },
    )
    result = NegativePatternMiner().mine(history)
    reference, = _signals(result, NegativeSignalKind.REVERT_SHA_REFERENCE)
    candidate, = result.candidates

    assert len(reference.commit_ids) == 2
    assert candidate.signal_count == len(candidate.signal_ids) == 3
    assert candidate.involved_commit_count == len(candidate.commit_ids) == 2
    assert candidate.examined_commits == result.examined_commits == 2
    serialized = result.to_dict()
    assert "support" not in serialized["candidates"][0]
    assert serialized["candidates"][0]["involved_commit_count"] == 2


def test_repeated_edit_counts_two_involved_commits_as_one_signal():
    result = NegativePatternMiner().mine(_history(
        {"paths": ("src/parser.py",)},
        {"paths": ("src/parser.py",)},
    ))
    candidate, = result.candidates

    assert candidate.signal_count == 1
    assert candidate.involved_commit_count == 2
    assert candidate.examined_commits == result.examined_commits == 2
    assert "support" not in candidate.__dataclass_fields__


def test_raw_candidate_counts_match_signal_closure_and_result_history_size():
    result = NegativePatternMiner().mine(_history(
        {"paths": ("src/a.py",)},
        {"paths": ("src/a.py",), "message": "Fix A"},
        {"paths": ("src/b.py",), "message": "Fix B"},
    ))
    by_id = {signal.signal_id: signal for signal in result.signals}

    for candidate in result.candidates:
        selected = [by_id[signal_id] for signal_id in candidate.signal_ids]
        assert candidate.signal_count == len(selected)
        assert candidate.involved_commit_count == len({
            commit_id for signal in selected for commit_id in signal.commit_ids
        })
        assert candidate.examined_commits == result.examined_commits == 3

    candidate = result.candidates[0]
    with pytest.raises(ValueError, match="signal_count"):
        replace(candidate, signal_count=candidate.signal_count + 1)
    with pytest.raises(ValueError, match="involved_commit_count"):
        replace(candidate, involved_commit_count=candidate.involved_commit_count + 1)
    with pytest.raises(ValueError, match="examined_commits must match"):
        replace(result, examined_commits=result.examined_commits + 1)


def test_result_evidence_is_exact_signal_and_candidate_closure():
    result = NegativePatternMiner().mine(_history(
        {"paths": ("src/a.py",), "message": "Add parser"},
        {"paths": ("src/a.py",), "message": "Fix parser"},
        {"paths": ("unrelated.py",), "message": "Update file"},
    ))
    signals_by_id = {signal.signal_id: signal for signal in result.signals}

    for candidate in result.candidates:
        expected = {
            evidence_id
            for signal_id in candidate.signal_ids
            for evidence_id in signals_by_id[signal_id].evidence_ids
        }
        assert candidate.evidence_ids == tuple(sorted(expected))
    expected_result = {
        evidence_id for signal in result.signals for evidence_id in signal.evidence_ids
    } | {
        evidence_id for candidate in result.candidates for evidence_id in candidate.evidence_ids
    }
    assert tuple(item.evidence_id for item in result.evidence) == tuple(sorted(expected_result))
    assert all(item.repository_id == result.repository_id for item in result.evidence)


def test_candidate_ids_order_and_json_are_deterministic():
    history = _history(
        {"paths": ("src/b.py",), "message": "fix B"},
        {"paths": ("src/a.py",), "message": "Revert \"Add A\""},
        {"paths": ("src/b.py",)},
    )
    first = NegativePatternMiner().mine(history)
    reversed_history = replace(
        history,
        changesets=tuple(reversed(history.changesets)),
        evidence=tuple(reversed(history.evidence)),
    )
    second = NegativePatternMiner().mine(reversed_history)

    assert first == second
    assert [signal.signal_id for signal in first.signals] == [
        signal.signal_id for signal in second.signals
    ]
    assert [candidate.candidate_id for candidate in first.candidates] == [
        candidate.candidate_id for candidate in second.candidates
    ]
    encoded = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert encoded == json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=True)
    assert decoded["examined_commits"] == 3
    assert all("support" not in candidate for candidate in decoded["candidates"])
    assert decoded["signals"][0]["kind"] in {
        item.value for item in NegativeSignalKind
    }


def test_empty_history_returns_empty_result():
    result = NegativePatternMiner().mine(GitHistoryResult(REPOSITORY_ID, (), ()))

    assert result.signals == ()
    assert result.candidates == ()
    assert result.evidence == ()
    assert result.examined_commits == 0
    assert result.to_dict()["repository_id"] == REPOSITORY_ID


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda history: replace(history, repository_id=""), "repository_id is required"),
        (
            lambda history: replace(
                history,
                changesets=(replace(history.changesets[0], repository_id="other"),),
            ),
            "changeset repository_id mismatch",
        ),
        (
            lambda history: replace(
                history,
                evidence=(
                    replace(history.evidence[0], repository_id="other"),
                    *history.evidence[1:],
                ),
            ),
            "Evidence repository_id mismatch",
        ),
        (
            lambda history: replace(
                history,
                changesets=(replace(history.changesets[0], committed_at="not-a-date"),),
            ),
            "Invalid commit timestamp",
        ),
        (
            lambda history: replace(
                history,
                changesets=(replace(history.changesets[0], commit_id="short"),),
            ),
            "Malformed Git commit id",
        ),
        (
            lambda history: replace(
                history,
                changesets=(replace(
                    history.changesets[0],
                    files=(replace(history.changesets[0].files[0], path="../escape.py"),),
                ),),
            ),
            "Invalid repository-relative POSIX path",
        ),
    ],
)
def test_repository_mismatch_and_malformed_history_are_rejected(mutate, match):
    history = _history({"paths": ("src/a.py",)})

    with pytest.raises(ValueError, match=match):
        NegativePatternMiner().mine(mutate(history))


def test_duplicate_missing_conflicting_and_unreferenced_input_evidence_are_rejected():
    history = _history({"paths": ("src/a.py",)})
    with pytest.raises(ValueError, match="Duplicate Evidence ID"):
        NegativePatternMiner().mine(replace(
            history, evidence=history.evidence + (history.evidence[0],)
        ))
    conflicting = replace(history.evidence[0], description="conflicting record")
    with pytest.raises(ValueError, match="Conflicting Evidence ID"):
        NegativePatternMiner().mine(replace(
            history, evidence=history.evidence + (conflicting,)
        ))
    with pytest.raises(ValueError, match="Missing Git history Evidence"):
        NegativePatternMiner().mine(replace(history, evidence=history.evidence[1:]))
    orphan = replace(history.evidence[0], evidence_id="orphan-evidence")
    with pytest.raises(ValueError, match="Unreferenced Git history Evidence"):
        NegativePatternMiner().mine(replace(history, evidence=history.evidence + (orphan,)))


def test_duplicate_commit_ids_and_file_evidence_mismatches_are_rejected():
    history = _history({"paths": ("src/a.py",)})
    with pytest.raises(ValueError, match="Duplicate commit_id"):
        NegativePatternMiner().mine(replace(
            history, changesets=history.changesets + history.changesets
        ))
    changeset = history.changesets[0]
    broken_file = replace(changeset.files[0], evidence_ids=changeset.files[0].evidence_ids[:1])
    with pytest.raises(ValueError, match="Missing or duplicate git_diff/git_numstat Evidence"):
        NegativePatternMiner().mine(replace(
            history, changesets=(replace(changeset, files=(broken_file,)),)
        ))


def test_result_contract_rejects_repository_mismatch_and_broken_evidence_closure():
    result = NegativePatternMiner().mine(_history({
        "paths": ("src/a.py",), "message": "fix parser",
    }))
    with pytest.raises(ValueError, match="NegativeSignal repository_id mismatch"):
        replace(result, repository_id="other")
    with pytest.raises(ValueError, match="Missing result Evidence"):
        replace(result, evidence=result.evidence[1:])
    with pytest.raises(ValueError, match="duplicate Evidence ID"):
        replace(result, evidence=result.evidence + (result.evidence[0],))
    conflict = replace(result.evidence[0], description="conflicting output Evidence")
    with pytest.raises(ValueError, match="Conflicting Evidence ID"):
        replace(result, evidence=result.evidence + (conflict,))
    with pytest.raises(ValueError, match="duplicate NegativeSignal ID"):
        replace(result, signals=result.signals + (result.signals[0],))
    with pytest.raises(ValueError, match="duplicate NegativePatternCandidate ID"):
        replace(result, candidates=result.candidates + (result.candidates[0],))
    mismatched_evidence = replace(result.evidence[0], repository_id="other")
    with pytest.raises(ValueError, match="Evidence repository_id mismatch"):
        replace(result, evidence=(mismatched_evidence, *result.evidence[1:]))


def test_cli_supports_human_json_and_parameter_validation(tmp_path, monkeypatch, capsys):
    history = _history({"paths": ("src/a.py",), "message": "fix parser"})

    def mine(_self, _path, limit=20):
        assert limit == 20
        return history

    monkeypatch.setattr(GitHistoryMiner, "mine", mine)
    assert main(["negative-patterns", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Negative signals: 1" in output
    assert "heuristic hint" in output
    assert "involved commits: 1; examined commits: 1" in output

    assert main(["negative-patterns", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signals"][0]["kind"] == "correction_message_hint"
    assert payload["candidates"]
    assert payload["candidates"][0]["signal_count"] == 1
    assert "support" not in payload["candidates"][0]

    with pytest.raises(SystemExit) as invalid_window:
        main(["negative-patterns", str(tmp_path), "--window-commits", "1"])
    assert invalid_window.value.code == 2
    with pytest.raises(SystemExit) as invalid_days:
        main(["negative-patterns", str(tmp_path), "--window-days", "inf"])
    assert invalid_days.value.code == 2


def test_cli_handles_a_real_empty_git_history(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)

    assert main(["negative-patterns", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["signals"] == []
    assert payload["candidates"] == []
    assert payload["evidence"] == []
