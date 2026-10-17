"""Evidence-closure and decision-gate contracts for semantic Anti-patterns."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from skillforge.domain.change import ChangeSet, FileChange, FileChangeKind, GitHistoryResult
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.domain.negative_patterns import (
    NegativePatternCandidateResult,
    NegativeSignalKind,
)
from skillforge.domain.semantic_antipattern import (
    SemanticAntiPatternDecisionKind,
    SemanticAntiPatternDecisionResponse,
    SemanticAntiPatternResponse,
    SemanticAntiPatternScope,
    SemanticAntiPatternScopeKind,
)
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_antipattern_prompt import (
    build_semantic_antipattern_prompt,
    semantic_antipattern_response_schema,
)
from skillforge.repository.semantic_antipatterns import (
    CONFIDENCE_METHOD,
    SemanticAntiPatternAbstracter,
    build_semantic_antipattern_request,
)

REPOSITORY_ID = "repo-semantic-antipattern-test"
BASE_TIME = datetime(2026, 9, 1, tzinfo=UTC)


def _commit_id(index):
    return f"{index:040x}"


def _history(*specs):
    changesets = []
    evidence = []
    for index, spec in enumerate(specs, start=1):
        commit_id = _commit_id(index)
        message = spec.get("message", "Update files")
        log_id = f"{commit_id}:git_log"
        evidence.append(Evidence(
            log_id, REPOSITORY_ID, EvidenceType.GIT, SourceLocation(".", commit_id),
            f"Observed Git commit {commit_id}", "git_log",
        ))
        evidence_ids = [log_id]
        files = []
        for entry in spec.get("changes", spec.get("paths", ())):
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
                evidence_id = f"{commit_id}:{path}:{provenance}"
                file_evidence_ids.append(evidence_id)
                evidence.append(Evidence(
                    evidence_id, REPOSITORY_ID, EvidenceType.GIT,
                    SourceLocation(path, commit_id),
                    f"Observed {provenance} for {path}", provenance,
                ))
            evidence_ids.extend(file_evidence_ids)
            files.append(FileChange(
                path=path,
                kind=kind,
                previous_path=previous_path,
                additions=2,
                deletions=1,
                evidence_ids=tuple(file_evidence_ids),
            ))
        changesets.append(ChangeSet(
            repository_id=REPOSITORY_ID,
            commit_id=commit_id,
            committed_at=(
                BASE_TIME + timedelta(days=spec.get("day", index))
            ).isoformat(),
            files=tuple(files),
            evidence_ids=tuple(evidence_ids),
            parent_ids=() if index == 1 else (_commit_id(index - 1),),
            message=message,
        ))
    return GitHistoryResult(
        REPOSITORY_ID,
        tuple(changesets),
        tuple(sorted(evidence, key=lambda item: item.evidence_id)),
    )


def _candidate_result(history):
    return NegativePatternMiner().mine(history)


def _candidate_for_path(result, path):
    return next(item for item in result.candidates if item.affected_paths == (path,))


def _without_provenance(result, history, provenance):
    evidence_by_id = {item.evidence_id: item for item in history.evidence}
    signals = tuple(
        replace(
            signal,
            evidence_ids=tuple(
                item for item in signal.evidence_ids
                if evidence_by_id[item].provenance != provenance
            ),
        )
        for signal in result.signals
    )
    signals_by_id = {item.signal_id: item for item in signals}
    candidates = tuple(
        replace(
            candidate,
            evidence_ids=tuple(sorted({
                evidence_id
                for signal_id in candidate.signal_ids
                for evidence_id in signals_by_id[signal_id].evidence_ids
            })),
        )
        for candidate in result.candidates
    )
    evidence_ids = {
        evidence_id for signal in signals for evidence_id in signal.evidence_ids
    }
    evidence = tuple(
        item for item in result.evidence if item.evidence_id in evidence_ids
    )
    return replace(result, signals=signals, candidates=candidates, evidence=evidence)


def _valid_anti_pattern(request):
    evidence_ids = tuple(item.evidence_id for item in request.evidence)
    linked = tuple(
        item for item in request.signals
        if item.kind == NegativeSignalKind.REVERT_SHA_REFERENCE
    )
    if linked:
        signal = linked[0]
        target_commit = signal.referenced_commit_id
        revert_commit = next(
            commit_id for commit_id in signal.commit_ids if commit_id != target_commit
        )
        path = request.affected_paths[0]
        summary = f"Commit {revert_commit} explicitly reverts commit {target_commit}."
        avoid = f"Do not reintroduce the legacy parser behavior at {path}."
    else:
        path = request.affected_paths[0] if request.affected_paths else None
        summary = "The supplied Git Evidence records the observed negative history signal."
        avoid = (
            f"Do not repeat the specific behavior recorded for {path}."
            if path else "Do not repeat the behavior recorded in this repository history."
        )

    scope_kind = (
        SemanticAntiPatternScopeKind.FILE
        if request.affected_paths else SemanticAntiPatternScopeKind.UNKNOWN
    )
    scope = SemanticAntiPatternScope(
        kind=scope_kind,
        selectors=request.affected_paths,
        description="The scope follows the exact candidate path."
        if path else "The supplied Evidence does not establish a narrower scope.",
        reason="The cited Git Evidence is tied to this exact path."
        if path else "No path is present in the candidate Evidence.",
        evidence_ids=evidence_ids,
    )
    return SemanticAntiPatternResponse(
        name="Repeating an explicitly reverted behavior",
        name_evidence_ids=evidence_ids,
        description="A prior behavior was followed by an explicit revert relationship.",
        description_evidence_ids=evidence_ids,
        avoid_behavior=avoid,
        avoid_behavior_evidence_ids=evidence_ids,
        negative_evidence_summary=summary,
        negative_evidence_ids=evidence_ids,
        scope=scope,
        exceptions=(),
        evidence_ids=evidence_ids,
    )


def _valid_response(request, decision=SemanticAntiPatternDecisionKind.ANTI_PATTERN):
    evidence_ids = tuple(item.evidence_id for item in request.evidence)
    if decision == SemanticAntiPatternDecisionKind.ANTI_PATTERN:
        anti_pattern = _valid_anti_pattern(request)
        reason = "The supplied negative history supports this specific avoidable behavior."
    else:
        anti_pattern = None
        reason = "The supplied signals do not establish one reusable avoidable behavior."
    return SemanticAntiPatternDecisionResponse(
        repository_id=request.repository_id,
        candidate_id=request.candidate_id,
        decision=decision,
        reason=reason,
        decision_evidence_ids=evidence_ids,
        confidence_score=0.81,
        anti_pattern=anti_pattern,
    )


class FakeModel:
    def __init__(self, response_factory=_valid_response):
        self.response_factory = response_factory
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.response_factory(request)


def test_request_is_candidate_scoped_and_repeated_edit_cannot_leak_commit_messages():
    history = _history(
        {"paths": ("src/parser.py",), "message": "PRIVATE first parser message"},
        {"paths": ("docs/guide.md",), "message": "Fix the documentation example"},
        {"paths": ("src/parser.py",), "message": "PRIVATE later parser message"},
        {"paths": ("unrelated.py",), "message": "Update an unrelated helper"},
    )
    result = _candidate_result(history)
    candidate = _candidate_for_path(result, "src/parser.py")

    request = build_semantic_antipattern_request(history, result, candidate)

    assert tuple(item.kind for item in request.signals) == (
        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
    )
    assert {item.commit_id for item in request.commits} == {
        _commit_id(1), _commit_id(3),
    }
    assert all(item.message is None for item in request.commits)
    assert all(
        item.provenance != "git_log" for item in request.evidence
    )
    assert all(item.path == "src/parser.py" for item in request.evidence)
    assert "docs/guide.md" not in {item.path for item in request.evidence}
    assert "unrelated.py" not in {item.path for item in request.evidence}
    assert "PRIVATE" not in json.dumps(request.to_dict())
    assert request.signal_count == candidate.signal_count
    assert request.involved_commit_count == candidate.involved_commit_count
    assert request.examined_commits == candidate.examined_commits
    assert "support" not in request.to_dict()


def test_request_does_not_expose_additions_or_deletions_without_candidate_numstat():
    history = _history(
        {"paths": ("src/repeated.py",), "message": "PRIVATE A"},
        {"paths": ("src/repeated.py",), "message": "PRIVATE B"},
    )
    mined = _candidate_result(history)
    result = _without_provenance(mined, history, "git_numstat")
    candidate = _candidate_for_path(result, "src/repeated.py")

    request = build_semantic_antipattern_request(history, result, candidate)

    files = tuple(
        changed_file for commit in request.commits
        for changed_file in commit.relevant_changed_files
    )
    assert len(files) == 2
    assert all(item.additions is None and item.deletions is None for item in files)
    assert all(item.kind == "modified" for item in files)
    assert "git_numstat" not in {item.provenance for item in request.evidence}
    assert all(item.message is None for item in request.commits)


def test_message_heuristic_requires_its_candidate_git_log_evidence():
    history = _history({
        "paths": ("src/parser.py",),
        "message": "Fix parser input",
    })
    mined = _candidate_result(history)
    result = _without_provenance(mined, history, "git_log")

    with pytest.raises(ValueError, match="missing git_log Evidence"):
        build_semantic_antipattern_request(
            history, result, _candidate_for_path(result, "src/parser.py")
        )


def test_revert_reference_exposes_only_closure_messages_and_file_facts():
    first_id = _commit_id(1)
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add unsafe legacy parser path"},
        {
            "paths": ("src/parser.py",),
            "day": 30,
            "message": (
                'Revert "Add unsafe legacy parser path"\n\n'
                f"This reverts commit {first_id}.\n"
            ),
        },
        {"paths": ("src/other.py",), "day": 31, "message": "PRIVATE unrelated message"},
    )
    result = _candidate_result(history)
    candidate = _candidate_for_path(result, "src/parser.py")

    request = build_semantic_antipattern_request(history, result, candidate)

    assert {item.commit_id for item in request.commits} == {_commit_id(1), _commit_id(2)}
    first, revert = request.commits
    assert first.message == "Add unsafe legacy parser path"
    assert first.relevant_changed_files == ()
    assert "Revert" in revert.message
    changed_file, = revert.relevant_changed_files
    assert changed_file.path == "src/parser.py"
    assert changed_file.kind == "modified"
    assert changed_file.additions is None and changed_file.deletions is None
    assert changed_file.evidence_ids
    assert "src/other.py" not in {item.path for item in request.evidence}
    assert "PRIVATE unrelated message" not in json.dumps(request.to_dict())


def test_accepted_anti_pattern_has_method_specific_confidence_and_exact_evidence_closure():
    first_id = _commit_id(1)
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add unsafe legacy parser path"},
        {
            "paths": ("src/parser.py",),
            "message": (
                'Revert "Add unsafe legacy parser path"\n\n'
                f"This reverts commit {first_id}.\n"
            ),
        },
    )
    candidates = _candidate_result(history)
    model = FakeModel()

    result = SemanticAntiPatternAbstracter().abstract(history, candidates, model)

    assert len(model.requests) == 1
    draft, = result.anti_patterns
    assert draft.candidate_id == candidates.candidates[0].candidate_id
    assert draft.draft_id
    assert draft.avoid_behavior
    assert draft.negative_evidence_summary
    assert draft.confidence.method == CONFIDENCE_METHOD
    assert draft.confidence.score == 0.81
    assert draft.confidence.support is None
    assert "support" not in draft.__dataclass_fields__
    assert "support" not in result.to_dict()
    assert draft.evidence_ids == tuple(sorted(draft.evidence_ids))
    assert {item.evidence_id for item in result.evidence} == set(draft.evidence_ids)


@pytest.mark.parametrize(
    "decision",
    [
        SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
        SemanticAntiPatternDecisionKind.INCOHERENT,
    ],
)
def test_semantic_rejections_are_valid_results_with_no_anti_pattern(decision):
    history = _history(
        {"paths": ("src/a.py",), "message": "Fix parser input"},
        {"paths": ("src/a.py",), "message": 'Revert "Add separate parser option"'},
    )
    candidates = _candidate_result(history)
    model = FakeModel(lambda request: _valid_response(request, decision))

    result = SemanticAntiPatternAbstracter().abstract(history, candidates, model)

    assert result.anti_patterns == ()
    rejection, = result.rejections
    assert rejection.decision == decision
    assert rejection.evidence_ids
    assert rejection.confidence.support is None
    assert rejection.confidence.method == CONFIDENCE_METHOD
    assert {item.evidence_id for item in result.evidence} == set(rejection.evidence_ids)


def test_short_window_repeat_signal_alone_can_be_legally_rejected_as_insufficient():
    history = _history(
        {"paths": ("src/a.py",), "message": "First implementation"},
        {"paths": ("src/a.py",), "message": "Follow-up implementation"},
    )
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    assert tuple(item.kind for item in candidates.signals) == (
        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
    )
    model = FakeModel(lambda request: _valid_response(
        request, SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE
    ))

    result = SemanticAntiPatternAbstracter().abstract(history, candidates, model)

    assert result.anti_patterns == ()
    rejection, = result.rejections
    assert rejection.candidate_id == candidate.candidate_id
    assert rejection.decision == SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE


def test_rejection_requires_a_complete_repeat_edit_signal_closure():
    history = _history(
        {"paths": ("src/a.py",), "message": "First implementation"},
        {"paths": ("src/a.py",), "message": "Follow-up implementation"},
    )
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    request = build_semantic_antipattern_request(history, candidates, candidate)
    signal, = request.signals
    assert signal.kind == NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT
    assert len(signal.evidence_ids) > 1
    one_diff = next(
        item.evidence_id for item in request.evidence if item.provenance == "git_diff"
    )
    assert one_diff in signal.evidence_ids
    decision = SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE

    with pytest.raises(ValueError, match="complete NegativeSignal Evidence closure"):
        SemanticAntiPatternAbstracter().abstract(
            history, candidates,
            FakeModel(lambda req: replace(
                _valid_response(req, decision), decision_evidence_ids=(one_diff,)
            )),
        )

    result = SemanticAntiPatternAbstracter().abstract(
        history, candidates,
        FakeModel(lambda req: replace(
            _valid_response(req, decision), decision_evidence_ids=signal.evidence_ids
        )),
    )
    rejection, = result.rejections
    assert rejection.evidence_ids == signal.evidence_ids
    assert {item.evidence_id for item in result.evidence} == set(signal.evidence_ids)


def test_rejection_needs_only_one_complete_signal_among_multiple():
    history = _history(
        {"paths": ("src/a.py",), "message": "Fix parser input"},
        {"paths": ("src/a.py",), "message": "Follow-up implementation"},
    )
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    request = build_semantic_antipattern_request(history, candidates, candidate)
    assert {signal.kind for signal in request.signals} == {
        NegativeSignalKind.CORRECTION_MESSAGE_HINT,
        NegativeSignalKind.SHORT_WINDOW_REPEAT_EDIT,
    }
    message_signal = next(
        signal for signal in request.signals
        if signal.kind == NegativeSignalKind.CORRECTION_MESSAGE_HINT
    )
    assert set(message_signal.evidence_ids) < {
        item.evidence_id for item in request.evidence
    }
    assert any(
        item.provenance == "git_log" and item.evidence_id in message_signal.evidence_ids
        for item in request.evidence
    )
    result = SemanticAntiPatternAbstracter().abstract(
        history, candidates,
        FakeModel(lambda req: replace(
            _valid_response(req, SemanticAntiPatternDecisionKind.INCOHERENT),
            decision_evidence_ids=message_signal.evidence_ids,
        )),
    )
    rejection, = result.rejections
    assert rejection.decision == SemanticAntiPatternDecisionKind.INCOHERENT
    assert rejection.evidence_ids == message_signal.evidence_ids
    assert {item.evidence_id for item in result.evidence} == set(message_signal.evidence_ids)


def test_accepted_draft_keeps_its_claim_level_contract_without_rejection_gate():
    first_id = _commit_id(1)
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add unsafe legacy parser path"},
        {
            "paths": ("src/parser.py",),
            "message": f"This reverts commit {first_id}.",
        },
    )
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    request = build_semantic_antipattern_request(history, candidates, candidate)
    one_id = (request.evidence[0].evidence_id,)
    assert not any(set(signal.evidence_ids) <= set(one_id) for signal in request.signals)
    result = SemanticAntiPatternAbstracter().abstract(
        history, candidates,
        FakeModel(lambda req: replace(
            _valid_response(req), decision_evidence_ids=one_id
        )),
    )
    draft, = result.anti_patterns
    assert draft.decision_evidence_ids == one_id
    assert draft.evidence_ids == tuple(item.evidence_id for item in request.evidence)
    assert draft.confidence.support is None


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_evidence", "unknown Candidate Evidence"),
        ("unknown_commit", "unknown commit"),
        ("unknown_path", "unknown path"),
        ("unknown_path_suffix", "unknown path"),
        ("fabricated_scope", "exactly equal the candidate path"),
        ("repository_mismatch", "repository_id mismatch"),
        ("candidate_mismatch", "candidate_id mismatch"),
        ("accepted_without_object", "requires an Anti-pattern object"),
        ("rejection_with_object", "must not include an Anti-pattern"),
        ("empty_evidence", "must cite at least one Evidence ID"),
        ("bad_confidence", "between 0 and 1"),
        ("duplicate_evidence", "duplicate Evidence IDs"),
        ("closure_mismatch", "claim reference closure"),
    ],
)
def test_strict_validator_rejects_invalid_model_output(case, message):
    first_id = _commit_id(1)
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add unsafe legacy parser path"},
        {
            "paths": ("src/parser.py",),
            "message": f'This reverts commit {first_id}.',
        },
    )
    candidates = _candidate_result(history)

    def invalid(request):
        response = _valid_response(request)
        anti = response.anti_pattern
        if case == "unknown_evidence":
            return replace(response, decision_evidence_ids=("unknown-evidence",))
        if case == "unknown_commit":
            anti = replace(
                anti, negative_evidence_summary=f"Commit {_commit_id(99)} is relevant."
            )
        elif case == "unknown_path":
            anti = replace(anti, avoid_behavior="Avoid src/service.py.")
        elif case == "unknown_path_suffix":
            anti = replace(anti, avoid_behavior="Avoid src/parser.py2.")
        elif case == "fabricated_scope":
            anti = replace(anti, scope=replace(
                anti.scope, selectors=("src/service.py",)
            ))
        elif case == "repository_mismatch":
            return replace(response, repository_id="different-repository")
        elif case == "candidate_mismatch":
            return replace(response, candidate_id="different-candidate")
        elif case == "accepted_without_object":
            return replace(response, anti_pattern=None)
        elif case == "rejection_with_object":
            return replace(
                response,
                decision=SemanticAntiPatternDecisionKind.INSUFFICIENT_EVIDENCE,
            )
        elif case == "empty_evidence":
            return replace(response, decision_evidence_ids=())
        elif case == "bad_confidence":
            return replace(response, confidence_score=float("nan"))
        elif case == "duplicate_evidence":
            first = anti.name_evidence_ids[0]
            anti = replace(anti, name_evidence_ids=(first, first))
        elif case == "closure_mismatch":
            first = anti.name_evidence_ids[0]
            one_id = (first,)
            anti = replace(
                anti,
                name_evidence_ids=one_id,
                description_evidence_ids=one_id,
                avoid_behavior_evidence_ids=one_id,
                negative_evidence_ids=one_id,
                scope=replace(anti.scope, evidence_ids=one_id),
                evidence_ids=response.anti_pattern.evidence_ids,
            )
        return replace(response, anti_pattern=anti)

    model = FakeModel(invalid)
    with pytest.raises(ValueError, match=message):
        SemanticAntiPatternAbstracter().abstract(history, candidates, model)


def test_pathless_candidates_allow_unknown_or_evidence_grounded_repository_scope():
    history = _history({
        "message": "Revert \"Remove cache fallback\"",
    })
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    request = build_semantic_antipattern_request(history, candidates, candidate)
    assert request.affected_paths == ()
    response = _valid_response(request)
    assert response.anti_pattern.scope.kind == SemanticAntiPatternScopeKind.UNKNOWN

    repository_scope = replace(
        response.anti_pattern.scope,
        kind=SemanticAntiPatternScopeKind.REPOSITORY,
        description="This message-only signal applies to repository-level behavior.",
        reason="The pathless Git log Evidence is the only scope context.",
    )
    anti = replace(response.anti_pattern, scope=repository_scope)
    repository_response = replace(response, anti_pattern=anti)
    output = SemanticAntiPatternAbstracter().abstract(
        history, candidates, FakeModel(lambda _request: repository_response)
    )
    assert output.anti_patterns[0].scope.kind == SemanticAntiPatternScopeKind.REPOSITORY


def test_duplicate_conflicting_and_candidate_input_mismatched_evidence_are_rejected():
    history = _history({"paths": ("src/a.py",), "message": "Fix parser"})
    candidates = _candidate_result(history)
    candidate = candidates.candidates[0]

    with pytest.raises(ValueError, match="Duplicate Git history Evidence ID"):
        build_semantic_antipattern_request(
            replace(history, evidence=history.evidence + (history.evidence[0],)),
            candidates,
            candidate,
        )
    conflicting = replace(history.evidence[0], description="different evidence content")
    with pytest.raises(ValueError, match="Conflicting Git history Evidence ID"):
        build_semantic_antipattern_request(
            replace(history, evidence=history.evidence + (conflicting,)),
            candidates,
            candidate,
        )

    changed = replace(candidates.evidence[0], description="candidate-side mutation")
    broken = replace(
        candidates,
        evidence=tuple(sorted(
            (changed, *candidates.evidence[1:]), key=lambda item: item.evidence_id
        )),
    )
    with pytest.raises(ValueError, match="Candidate/input Evidence mismatch"):
        build_semantic_antipattern_request(history, broken, candidate)


def test_one_candidate_cannot_receive_two_semantic_decisions():
    history = _history({"paths": ("src/a.py",), "message": "Fix parser"})
    candidates = _candidate_result(history)
    candidate, = candidates.candidates
    model = FakeModel()

    with pytest.raises(ValueError, match="only one semantic decision"):
        SemanticAntiPatternAbstracter().abstract(
            history, candidates, model, selected_candidates=(candidate, candidate)
        )
    assert model.requests == []


def test_result_contract_rejects_evidence_outside_actual_output_closure():
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add parser"},
        {"paths": ("src/parser.py",), "message": 'Revert "Add parser"'},
    )
    result = SemanticAntiPatternAbstracter().abstract(
        history, _candidate_result(history), FakeModel()
    )
    orphan = Evidence(
        "orphan-evidence", REPOSITORY_ID, EvidenceType.GIT,
        SourceLocation(".", _commit_id(1)), "orphan", "git_log",
    )
    with pytest.raises(ValueError, match="Evidence must match output references"):
        replace(
            result,
            evidence=tuple(sorted(
                (*result.evidence, orphan), key=lambda item: item.evidence_id
            )),
        )
    with pytest.raises(ValueError, match="Evidence must be unique and sorted"):
        replace(result, evidence=result.evidence + (result.evidence[0],))
    draft = result.anti_patterns[0]
    one_id = (draft.name_evidence_ids[0],)
    broken_draft = replace(
        draft,
        name_evidence_ids=one_id,
        description_evidence_ids=one_id,
        avoid_behavior_evidence_ids=one_id,
        negative_evidence_ids=one_id,
        scope=replace(draft.scope, evidence_ids=one_id),
        decision_evidence_ids=one_id,
    )
    with pytest.raises(ValueError, match="Evidence must match output references"):
        replace(result, anti_patterns=(broken_draft,))


def test_request_prompt_and_draft_are_deterministic_and_schema_has_no_support():
    history = _history(
        {"paths": ("src/parser.py",), "message": "Add parser"},
        {"paths": ("src/parser.py",), "message": 'Revert "Add parser"'},
    )
    candidates = _candidate_result(history)
    candidate = candidates.candidates[0]
    first = build_semantic_antipattern_request(history, candidates, candidate)
    second = build_semantic_antipattern_request(history, candidates, candidate)
    assert first == second
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    assert build_semantic_antipattern_prompt(first) == build_semantic_antipattern_prompt(
        second
    )
    schema = semantic_antipattern_response_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["anti_pattern"]["type"] == ["object", "null"]
    assert "support" not in json.dumps(schema)
    assert "SHORT_WINDOW_REPEAT_EDIT" in build_semantic_antipattern_prompt(first)
    assert "never use slash-separated conceptual shorthand" in (
        build_semantic_antipattern_prompt(first)
    )
    assert "Use '/' only when reproducing an exact repository path" in (
        build_semantic_antipattern_prompt(first)
    )
    assert "not a probability" in build_semantic_antipattern_prompt(first)

    first_result = SemanticAntiPatternAbstracter().abstract(
        history, candidates, FakeModel()
    )
    second_result = SemanticAntiPatternAbstracter().abstract(
        history, candidates, FakeModel()
    )
    assert first_result.anti_patterns[0].draft_id == second_result.anti_patterns[0].draft_id
    assert first_result.evidence == second_result.evidence


def test_empty_candidate_result_does_not_call_model():
    history = _history({"paths": ("src/a.py",), "message": "Add parser"})
    empty = NegativePatternCandidateResult(REPOSITORY_ID, (), (), (), examined_commits=1)
    model = FakeModel()

    result = SemanticAntiPatternAbstracter().abstract(history, empty, model)

    assert result.anti_patterns == result.rejections == result.evidence == ()
    assert model.requests == []


def test_domain_and_repository_packages_export_the_new_contract_and_service():
    import skillforge.domain as public_domain
    import skillforge.repository as public_repository

    assert public_domain.SemanticAntiPatternRequest.__name__ == "SemanticAntiPatternRequest"
    assert public_repository.AntiPatternSemanticModel.__name__ == "AntiPatternSemanticModel"
    assert public_repository.SemanticAntiPatternAbstracter is SemanticAntiPatternAbstracter
    assert (
        public_repository.SemanticAntiPatternPromptBuilder.__name__
        == "SemanticAntiPatternPromptBuilder"
    )
    assert (
        public_repository.build_semantic_antipattern_request
        is build_semantic_antipattern_request
    )
