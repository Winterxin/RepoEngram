import json
from dataclasses import replace
from itertools import combinations

import pytest

from skillforge.domain.change import (
    ChangePattern,
    ChangePatternItem,
    ChangePatternKind,
    ChangePatternResult,
    ChangeSet,
    FileChange,
    FileChangeKind,
    GitHistoryResult,
    NoiseSignal,
    NoiseSignalKind,
    PairAssociation,
)
from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation, Support
from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticDecisionResponse,
    SemanticEntityReferences,
    SemanticStepClassification,
    SemanticWorkflowException,
    SemanticWorkflowResponse,
    SemanticWorkflowRole,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.static_analysis import (
    ImportRelation,
    ImportResolution,
    PythonModuleAnalysis,
    PythonParseStatus,
    StaticAnalysisResult,
)
from skillforge.domain.workflow import WorkflowCandidateResult
from skillforge.repository.semantic_workflow_prompt import build_semantic_workflow_prompt
from skillforge.repository.semantic_workflows import (
    CONFIDENCE_METHOD,
    SemanticWorkflowAbstracter,
    build_semantic_workflow_request,
)
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner

REPOSITORY_ID = "repo"
MODULE_KIND = ChangePatternKind.MODULE_CO_CHANGE


def _evidence(
    evidence_id,
    path,
    *,
    commit=None,
    kind=EvidenceType.SOURCE,
    provenance="fixture",
    description=None,
):
    return Evidence(
        evidence_id=evidence_id,
        repository_id=REPOSITORY_ID,
        kind=kind,
        location=SourceLocation(path, commit),
        description=description or evidence_id,
        provenance=provenance,
    )


def _file_path(module):
    return module.replace(".", "/") + ".py"


def _make_pattern(kind, values, commits, evidence_ids, *, examined=10):
    values = tuple(sorted(values))
    commits = tuple(sorted(commits))
    association = None
    if len(values) == 2:
        association = PairAssociation(len(commits), len(commits), 1.0, 1.0, 1.0)
    return ChangePattern(
        pattern_id=f"{kind.value}:{','.join(values)}",
        repository_id=REPOSITORY_ID,
        kind=kind,
        items=tuple(ChangePatternItem(value) for value in values),
        supporting_commits=commits,
        evidence_ids=tuple(sorted(set(evidence_ids))),
        support=Support(len(commits), examined),
        support_ratio=len(commits) / examined if examined else 0.0,
        item_occurrences=(len(commits),) * len(values),
        pair_association=association,
    )


def _fixture(*, subset_support=False, irrelevant=False):
    members = ("pkg.a", "pkg.b", "pkg.c", "pkg.d") if subset_support else (
        "pkg.a", "pkg.b", "pkg.c"
    )
    commits = tuple(f"c{i}" for i in range(1, 9 if subset_support else 3))
    file_evidence = {}
    evidence_by_id = {}
    changesets = []
    static_evidence = []

    for commit in commits:
        log_id = f"{commit}:log"
        log = _evidence(
            log_id, ".", commit=commit, kind=EvidenceType.GIT, provenance="git_log"
        )
        evidence_by_id[log_id] = log
        changed_modules = members if not subset_support or int(commit[1:]) <= 5 else members[:3]
        file_changes = []
        changeset_evidence_ids = [log_id]
        for module in changed_modules:
            path = _file_path(module)
            diff_id = f"{commit}:{path}:diff"
            numstat_id = f"{commit}:{path}:numstat"
            diff = _evidence(
                diff_id, path, commit=commit, kind=EvidenceType.GIT,
                provenance="git_diff", description=f"Changed {path}",
            )
            numstat = _evidence(
                numstat_id, path, commit=commit, kind=EvidenceType.GIT,
                provenance="git_numstat", description=f"Counts for {path}",
            )
            evidence_by_id[diff_id] = diff
            evidence_by_id[numstat_id] = numstat
            file_evidence[(commit, module)] = (diff_id, numstat_id)
            changeset_evidence_ids.extend((diff_id, numstat_id))
            file_changes.append(FileChange(
                path=path,
                kind=FileChangeKind.MODIFIED,
                additions=3,
                deletions=1,
                evidence_ids=(diff_id, numstat_id),
            ))
        changesets.append(ChangeSet(
            repository_id=REPOSITORY_ID,
            commit_id=commit,
            committed_at=f"2026-09-2{int(commit[1:])}T12:00:00+00:00",
            files=tuple(file_changes),
            evidence_ids=tuple(changeset_evidence_ids),
            message=f"Coordinate modules for {commit}",
        ))

    for module in members:
        path = _file_path(module)
        item = _evidence(f"source:{module}", path, description=f"Current module {module}")
        evidence_by_id[item.evidence_id] = item
        static_evidence.append(item)

    import_evidence = _evidence("import:pkg.a:pkg.b", _file_path("pkg.a"))
    evidence_by_id[import_evidence.evidence_id] = import_evidence
    static_evidence.append(import_evidence)
    import_relation = ImportRelation(
        repository_id=REPOSITORY_ID,
        source_module="pkg.a",
        target_module="pkg.b",
        raw_module="pkg.b",
        imported_names=(),
        is_relative=False,
        resolution=ImportResolution.INTERNAL,
        path=_file_path("pkg.a"),
        line=4,
        evidence_ids=(import_evidence.evidence_id,),
    )
    modules = tuple(PythonModuleAnalysis(
        module=module,
        path=_file_path(module),
        is_package=False,
        symbols=(),
        imports=(import_relation,) if module == "pkg.a" else (),
        parse_status=PythonParseStatus.PARSED,
        parse_issue=None,
        evidence_ids=(f"source:{module}",),
    ) for module in members)
    static_analysis = StaticAnalysisResult(
        REPOSITORY_ID,
        modules,
        tuple(sorted(static_evidence, key=lambda item: item.evidence_id)),
        (),
    )

    pattern_specs = []
    if subset_support:
        support_by_pattern = {}
        for pair in combinations(members, 2):
            support_by_pattern[pair] = commits if "pkg.d" not in pair else commits[:5]
        support_by_pattern[tuple(members[:3])] = commits
        support_by_pattern[members] = commits[:5]
        patterns_to_make = [(values, source_commits)
                            for values, source_commits in support_by_pattern.items()]
    else:
        patterns_to_make = [
            (pair, commits) for pair in combinations(members, 2)
        ] + [(members, commits)]

    for values, source_commits in patterns_to_make:
        pattern_evidence = {f"{commit}:log" for commit in source_commits}
        for commit in source_commits:
            for module in values:
                pattern_evidence.update(file_evidence[(commit, module)])
        pattern_evidence.update(f"source:{module}" for module in values)
        pattern_specs.append(_make_pattern(
            MODULE_KIND, values, source_commits, pattern_evidence,
            examined=10 if subset_support else 4,
        ))

    noise = NoiseSignal(
        NoiseSignalKind.UBIQUITOUS_ITEM,
        "module_co_change:pkg.a",
        "occurrence_count",
        ("noise:pkg.a",),
    )
    noise_evidence = _evidence("noise:pkg.a", _file_path("pkg.a"))
    evidence_by_id[noise_evidence.evidence_id] = noise_evidence
    static_analysis = replace(
        static_analysis,
        evidence=tuple(sorted(
            (*static_analysis.evidence, noise_evidence), key=lambda item: item.evidence_id
        )),
    )
    pattern_evidence_ids = {
        evidence_id for pattern in pattern_specs for evidence_id in pattern.evidence_ids
    }
    pattern_evidence_ids.add(noise_evidence.evidence_id)
    extra_patterns = []
    if irrelevant:
        commit = "c999"
        log_id = f"{commit}:log"
        diff_id = f"{commit}:outside.py:diff"
        numstat_id = f"{commit}:outside.py:numstat"
        for item in (
            _evidence(log_id, ".", commit=commit, kind=EvidenceType.GIT, provenance="git_log"),
            _evidence(diff_id, "outside.py", commit=commit, kind=EvidenceType.GIT,
                      provenance="git_diff"),
            _evidence(numstat_id, "outside.py", commit=commit, kind=EvidenceType.GIT,
                      provenance="git_numstat"),
        ):
            evidence_by_id[item.evidence_id] = item
        changesets.append(ChangeSet(
            repository_id=REPOSITORY_ID,
            commit_id=commit,
            committed_at="2026-09-25T12:00:00+00:00",
            files=(FileChange(
                "outside.py", FileChangeKind.ADDED, additions=20, deletions=0,
                evidence_ids=(diff_id, numstat_id),
            ),),
            evidence_ids=(log_id, diff_id, numstat_id),
            message="Unrelated documentation helper",
        ))
        unrelated_evidence = _evidence("unrelated:pattern", "outside.py")
        evidence_by_id[unrelated_evidence.evidence_id] = unrelated_evidence
        extra_patterns.append(_make_pattern(
            ChangePatternKind.FILE_CO_CHANGE,
            ("outside-a.py", "outside-b.py"),
            (commit,),
            (unrelated_evidence.evidence_id,),
            examined=4,
        ))

    all_pattern_evidence = set(pattern_evidence_ids)
    all_pattern_evidence.update(
        evidence_id for pattern in extra_patterns for evidence_id in pattern.evidence_ids
    )
    change_patterns = ChangePatternResult(
        REPOSITORY_ID,
        tuple((*pattern_specs, *extra_patterns)),
        tuple(evidence_by_id[key] for key in sorted(all_pattern_evidence)),
        (noise,),
    )
    history = GitHistoryResult(
        REPOSITORY_ID,
        tuple(changesets),
        tuple(sorted((item for item in evidence_by_id.values()
                      if item.kind == EvidenceType.GIT), key=lambda item: item.evidence_id)),
    )
    result = WorkflowCandidateMiner().mine(history, static_analysis, change_patterns)
    return history, static_analysis, result


def _valid_response(request):
    evidence_id = request.evidence[0].evidence_id
    step_one = SemanticWorkflowStep("step-1", "Define the shared contract", (evidence_id,))
    step_two = SemanticWorkflowStep("step-2", "Coordinate the related changes", (evidence_id,))
    return SemanticWorkflowResponse(
        repository_id=request.repository_id,
        candidate_id=request.candidate_id,
        name="Coordinated implementation workflow",
        name_evidence_ids=(evidence_id,),
        intent="Describe the recurring coordinated engineering activity",
        intent_evidence_ids=(evidence_id,),
        roles=(SemanticWorkflowRole(
            "role-1", "Implementer", "Coordinates the evidenced changes", (evidence_id,)
        ),),
        ordered_steps=(step_one, step_two),
        required_steps=(SemanticStepClassification(
            "step-1", "This action appears in the supporting changes", (evidence_id,)
        ),),
        optional_steps=(SemanticStepClassification(
            "step-2", "The source family shows varying support", (evidence_id,)
        ),),
        conditional_steps=(),
        scope=SemanticWorkflowScope(
            SemanticWorkflowScopeKind.UNKNOWN,
            (),
            "The repository evidence does not establish a narrower scope",
            "No supplied Evidence grounds a narrower scope",
            (evidence_id,),
        ),
        exceptions=(SemanticWorkflowException(
            "No supported exception is established", (evidence_id,)
        ),),
        confidence_score=0.73,
        references=SemanticEntityReferences(
            commit_ids=(request.supporting_commits[0].commit_id,),
            module_names=request.current_modules[:1],
            paths=(),
        ),
    )


class FakeModel:
    def __init__(self, response_factory=_valid_response):
        self.response_factory = response_factory
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        response = self.response_factory(request)
        if isinstance(response, SemanticDecisionResponse):
            return response
        return SemanticDecisionResponse(
            request.repository_id, request.candidate_id, SemanticDecisionKind.WORKFLOW,
            "Historical examples support a repeatable engineering sequence",
            (request.evidence[0].evidence_id,), response.confidence_score,
            request.support, response,
        )


def test_candidate_context_includes_metrics_structure_noise_family_and_supporting_commit_files():
    history, static, result = _fixture()
    candidate = result.candidates[0]
    request = build_semantic_workflow_request(history, static, result, candidate)

    assert request.candidate_id == candidate.candidate_id
    assert request.repository_id == REPOSITORY_ID
    assert tuple(member.value for member in request.members) == tuple(
        member.value for member in candidate.members
    )
    assert request.current_modules == candidate.current_modules
    assert request.support == candidate.support
    assert request.support_ratio == candidate.support_ratio
    assert request.pair_associations == candidate.pair_associations
    assert request.structural_links == candidate.structural_links
    assert request.structural_coherence == candidate.structural_coherence
    assert request.noise_hints == candidate.noise_signals
    assert {pattern.pattern_id for pattern in request.source_patterns} == set(
        candidate.source_pattern_ids
    )
    assert tuple(item.commit_id for item in request.supporting_commits) == (
        *candidate.supporting_commits,
    )
    assert all(
        {file.path for file in item.relevant_changed_files} == {
            _file_path(module) for module in candidate.current_modules
        }
        for item in request.supporting_commits
    )


def test_context_and_result_omit_unrelated_history_and_evidence():
    history, static, result = _fixture(irrelevant=True)
    candidate = next(item for item in result.candidates if len(item.members) == 3)
    request = build_semantic_workflow_request(history, static, result, candidate)

    assert {item.commit_id for item in request.supporting_commits} == set(
        candidate.supporting_commits
    )
    assert "c999" not in {item.commit_id for item in request.supporting_commits}
    assert "outside.py" not in {
        file.path for commit in request.supporting_commits
        for file in commit.relevant_changed_files
    }
    assert "unrelated:pattern" not in {item.evidence_id for item in request.evidence}
    assert all(item.commit_id != "c999" for item in request.evidence)

    model = FakeModel()
    output = SemanticWorkflowAbstracter().abstract(history, static, result, model)
    assert "unrelated:pattern" not in {item.evidence_id for item in output.evidence}
    assert all(item.location.commit != "c999" for item in output.evidence)


def test_source_pattern_subset_support_family_is_passed_without_extra_commit_payloads():
    history, static, result = _fixture(subset_support=True, irrelevant=True)
    four_member = next(item for item in result.candidates if len(item.members) == 4)
    request = build_semantic_workflow_request(history, static, result, four_member)

    by_items = {pattern.items: pattern for pattern in request.source_patterns}
    assert by_items[("pkg.a", "pkg.b", "pkg.c")].support.supporting == 8
    assert by_items[("pkg.a", "pkg.b", "pkg.c", "pkg.d")].support.supporting == 5
    assert by_items[("pkg.a", "pkg.b", "pkg.c")].supporting_commits == tuple(
        f"c{i}" for i in range(1, 9)
    )
    assert by_items[("pkg.a", "pkg.b", "pkg.c", "pkg.d")].supporting_commits == tuple(
        f"c{i}" for i in range(1, 6)
    )
    assert len(request.supporting_commits) == 5
    assert len(request.family_supporting_commits) == 8
    assert {item.commit_id for item in request.family_supporting_commits} == {
        f"c{i}" for i in range(1, 9)
    }
    assert all(item.message and item.evidence_ids for item in request.family_supporting_commits)
    assert all(item.supports_seed for item in request.family_supporting_commits[:5])
    assert all(not item.supports_seed for item in request.family_supporting_commits[5:])
    triple_id = by_items[("pkg.a", "pkg.b", "pkg.c")].pattern_id
    four_id = by_items[("pkg.a", "pkg.b", "pkg.c", "pkg.d")].pattern_id
    assert all(triple_id in item.source_pattern_ids for item in request.family_supporting_commits)
    assert all(four_id in item.source_pattern_ids for item in request.family_supporting_commits[:5])
    assert all(four_id not in item.source_pattern_ids
               for item in request.family_supporting_commits[5:])
    assert "c999" not in {item.commit_id for item in request.family_supporting_commits}
    assert all("pkg/d.py" not in {file.path for file in item.relevant_changed_files}
               for item in request.family_supporting_commits[5:])
    assert {item.commit_id for item in request.supporting_commits} == set(
        four_member.supporting_commits
    )
    assert all(
        "pkg/d.py" in {file.path for file in item.relevant_changed_files}
        for item in request.supporting_commits
    )


def test_valid_fake_model_returns_draft_with_method_specific_confidence_and_closed_evidence():
    history, static, result = _fixture()
    model = FakeModel()
    output = SemanticWorkflowAbstracter().abstract(history, static, result, model)

    assert len(model.requests) == len(result.candidates) == 1
    draft = output.workflows[0]
    assert draft.repository_id == result.repository_id
    assert draft.candidate_id == result.candidates[0].candidate_id
    assert draft.draft_id
    assert draft.confidence.method == CONFIDENCE_METHOD
    assert draft.confidence.support == result.candidates[0].support
    assert draft.confidence.score == 0.73
    assert {item.evidence_id for item in output.evidence} == set(draft.evidence_ids)
    assert draft.evidence_ids == tuple(sorted(draft.evidence_ids))
    assert {item.step_id for item in draft.ordered_steps} == {
        item.step_id for group in (
            draft.required_steps, draft.optional_steps, draft.conditional_steps
        ) for item in group
    }


def test_dangling_evidence_id_is_rejected():
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), name_evidence_ids=("missing-evidence",))

    with pytest.raises(ValueError, match="unknown Candidate Evidence"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize(
    ("field", "value", "label"),
    [
        ("commit_ids", ("deadbeef0",), "commit"),
        ("module_names", ("pkg.not_real",), "module"),
        ("paths", ("src/not_real.py",), "path"),
    ],
)
def test_fabricated_explicit_commit_module_or_path_is_rejected(field, value, label):
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        references = replace(valid.references, **{field: value})
        return replace(valid, references=references)

    with pytest.raises(ValueError, match=f"unknown {label}"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_id", "other-candidate", "candidate_id mismatch"),
        ("repository_id", "other-repository", "repository_id mismatch"),
    ],
)
def test_identity_mismatch_is_rejected(field, value, message):
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), **{field: value})

    with pytest.raises(ValueError, match=message):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_duplicate_step_id_is_rejected():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        return replace(valid, ordered_steps=(valid.ordered_steps[0], valid.ordered_steps[0]))

    with pytest.raises(ValueError, match="duplicate step ID"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_required_optional_conditional_classification_conflict_is_rejected():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        duplicate = SemanticStepClassification(
            "step-1", "Conflicting classification", valid.ordered_steps[0].evidence_ids
        )
        return replace(valid, optional_steps=(*valid.optional_steps, duplicate))

    with pytest.raises(ValueError, match="multiple required/optional/conditional"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_conditional_step_requires_evidenced_nonempty_condition():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        conditional = replace(valid.optional_steps[0], condition=None)
        return replace(valid, optional_steps=(), conditional_steps=(conditional,))

    with pytest.raises(ValueError, match="conditional_steps condition"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_conditional_step_with_condition_passes_and_retains_evidence():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        conditional = replace(
            valid.optional_steps[0],
            condition="when the related contract is needed",
        )
        return replace(valid, optional_steps=(), conditional_steps=(conditional,))

    output = SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))
    classification = output.workflows[0].conditional_steps[0]
    assert classification.condition == "when the related contract is needed"
    assert classification.evidence_ids
    assert set(classification.evidence_ids) <= set(output.workflows[0].evidence_ids)


def test_conditional_condition_cannot_fabricate_repository_entity():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        conditional = replace(
            valid.optional_steps[0], condition="when invented.module is needed"
        )
        return replace(valid, optional_steps=(), conditional_steps=(conditional,))

    with pytest.raises(ValueError, match="unknown module"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize("group", ["required_steps", "optional_steps"])
def test_unconditional_step_rejects_condition(group):
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        item = getattr(valid, group)[0]
        return replace(valid, **{group: (replace(item, condition="when needed"),)})

    with pytest.raises(ValueError, match=f"{group} condition must be None"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_every_ordered_step_must_be_classified():
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), optional_steps=())

    with pytest.raises(ValueError, match="ordered_steps missing classification"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_empty_ordered_steps_are_rejected():
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), ordered_steps=())

    with pytest.raises(ValueError, match="ordered_steps must be a non-empty tuple"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("name", "  ", "workflow name"), ("intent", "", "workflow intent")],
)
def test_empty_name_or_intent_is_rejected(field, value, message):
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), **{field: value})

    with pytest.raises(ValueError, match=message):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_confidence_score_out_of_range_or_non_finite_is_rejected(score):
    history, static, result = _fixture()

    def response(request):
        return replace(_valid_response(request), confidence_score=score)

    with pytest.raises(ValueError, match="Confidence score"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


@pytest.mark.parametrize(
    ("action", "message"),
    [("Open invented.py", "unknown path"), ("Call invented.module", "unknown module")],
)
def test_fabricated_path_or_module_inside_step_text_is_rejected(action, message):
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        step = replace(valid.ordered_steps[0], action=action)
        return replace(valid, ordered_steps=(step, valid.ordered_steps[1]))

    with pytest.raises(ValueError, match=message):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_unresolved_supporting_commit_fails_explicitly():
    history, static, result = _fixture()
    candidate = replace(result.candidates[0], supporting_commits=("missing-commit",))
    invalid_result = replace(result, candidates=(candidate,))

    with pytest.raises(ValueError, match="Missing supporting commits"):
        build_semantic_workflow_request(history, static, invalid_result, candidate)


def test_candidate_source_evidence_missing_from_inputs_fails():
    history, static, result = _fixture()
    missing_id = "noise:pkg.a"
    assert missing_id in result.candidates[0].evidence_ids
    static = replace(
        static,
        evidence=tuple(item for item in static.evidence if item.evidence_id != missing_id),
    )

    with pytest.raises(ValueError, match=f"Missing input Evidence: {missing_id}"):
        build_semantic_workflow_request(history, static, result, result.candidates[0])


def test_candidate_input_evidence_content_conflict_fails():
    history, static, result = _fixture()
    evidence = static.evidence[0]
    static = replace(static, evidence=tuple(
        replace(item, description="conflicting content") if item == evidence else item
        for item in static.evidence
    ))

    with pytest.raises(ValueError, match="Candidate/input Evidence mismatch"):
        build_semantic_workflow_request(history, static, result, result.candidates[0])


def test_step_cannot_cite_evidence_outside_its_candidate_request():
    history, static, result = _fixture()

    def response(request):
        valid = _valid_response(request)
        return replace(
            valid,
            ordered_steps=(replace(valid.ordered_steps[0], evidence_ids=("other-evidence",)),
                           valid.ordered_steps[1]),
        )

    with pytest.raises(ValueError, match="unknown Candidate Evidence"):
        SemanticWorkflowAbstracter().abstract(history, static, result, FakeModel(response))


def test_empty_candidate_result_returns_empty_without_calling_model():
    history, static, _ = _fixture()
    empty = WorkflowCandidateResult(REPOSITORY_ID, (), (), ())
    model = FakeModel()

    result = SemanticWorkflowAbstracter().abstract(history, static, empty, model)

    assert result.workflows == ()
    assert result.evidence == ()
    assert model.requests == []


def test_serialization_and_draft_identity_are_deterministic():
    history, static, candidates = _fixture()
    first = SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel())
    second = SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel())

    assert first == second
    assert json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":")) == json.dumps(
        second.to_dict(), sort_keys=True, separators=(",", ":")
    )


def test_prompt_builder_states_evidence_only_constraints_and_is_deterministic():
    history, static, result = _fixture()
    request = build_semantic_workflow_request(
        history, static, result, result.candidates[0]
    )
    first = build_semantic_workflow_prompt(request)
    second = build_semantic_workflow_prompt(request)

    assert first == second
    assert "Evidence-only" in first
    assert "Do not discover or invent facts" in first
    assert "never use slash-separated conceptual shorthand" in first
    assert "Use '/' only when reproducing an exact repository path" in first
    assert "An import direction is not a development order" in first
    assert "A pair association is not causation" in first
    assert "Noise hints are heuristics only" in first
    assert "RESPONSE_SCHEMA" in first
    assert request.candidate_id in first
    assert "Classify each supported logical step exactly once" in first
    assert "Conditional steps must state a non-empty trigger condition" in first
    schema = json.loads(first.split("RESPONSE_SCHEMA:\n", 1)[1].split(
        "\nCANDIDATE_CONTEXT:", 1
    )[0])
    props = schema["properties"]
    props = props["workflow"]["properties"]
    assert props["ordered_steps"]["minItems"] == 1
    assert schema["properties"]["classification_protocol"]["enum"] == [
        "ordered-step-classification-v2"
    ]
    assert not {"required_steps", "optional_steps", "conditional_steps"} & set(props)
    step_props = props["ordered_steps"]["items"]["properties"]
    assert "allOf" not in props["ordered_steps"]["items"]
    assert step_props["classification_kind"]["enum"] == [
        "required", "optional", "conditional"
    ]
    assert step_props["classification_evidence_ids"]["minItems"] == 1
    assert step_props["classification_condition"] == {"type": ["string", "null"]}


def test_duplicate_evidence_ids_are_canonicalized_across_decision_and_workflow():
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        evidence_id = request.evidence[0].evidence_id
        duplicate = (evidence_id, evidence_id)
        workflow = replace(
            valid,
            name_evidence_ids=duplicate,
            intent_evidence_ids=duplicate,
            roles=(replace(valid.roles[0], evidence_ids=duplicate),),
            ordered_steps=tuple(
                replace(step, evidence_ids=duplicate) for step in valid.ordered_steps
            ),
            required_steps=(replace(valid.required_steps[0], evidence_ids=duplicate),),
            optional_steps=(replace(valid.optional_steps[0], evidence_ids=duplicate),),
            scope=replace(valid.scope, evidence_ids=duplicate),
            exceptions=(replace(valid.exceptions[0], evidence_ids=duplicate),),
        )
        return replace(
            FakeModel(lambda _: workflow).complete(request),
            decision_evidence_ids=duplicate,
        )

    output = SemanticWorkflowAbstracter().abstract(
        history, static, candidates, FakeModel(response)
    )
    draft = output.workflows[0]
    assert draft.name_evidence_ids == (draft.name_evidence_ids[0],)
    assert draft.roles[0].evidence_ids == draft.name_evidence_ids
    assert draft.evidence_ids == tuple(sorted(set(draft.evidence_ids)))


def test_duplicate_role_id_still_fails():
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        return replace(valid, roles=(valid.roles[0], valid.roles[0]))

    with pytest.raises(ValueError, match="duplicate workflow role ID"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


def test_module_scope_accepts_multiple_exact_current_modules():
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        scope = replace(
            valid.scope, kind=SemanticWorkflowScopeKind.MODULE,
            selectors=tuple(reversed(request.current_modules[:2])),
        )
        return replace(valid, scope=scope)

    output = SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))
    assert output.workflows[0].scope.selectors == tuple(
        sorted(candidates.candidates[0].current_modules[:2])
    )


def test_joined_module_selector_is_rejected():
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        scope = replace(
            valid.scope, kind=SemanticWorkflowScopeKind.MODULE,
            selectors=(", ".join(request.current_modules[:2]),),
        )
        return replace(valid, scope=scope)

    with pytest.raises(ValueError, match="unknown module"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


@pytest.mark.parametrize(
    "kind", [SemanticWorkflowScopeKind.FILE, SemanticWorkflowScopeKind.DIRECTORY]
)
def test_file_and_directory_scopes_require_exact_allowed_selectors(kind):
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        path = next(item.path for item in request.evidence if "/" in item.path)
        selector = path if kind == SemanticWorkflowScopeKind.FILE else path.split("/")[0]
        return replace(
            valid,
            scope=replace(valid.scope, kind=kind, selectors=(selector,)),
        )

    output = SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))
    assert len(output.workflows[0].scope.selectors) == 1


def test_entity_scope_requires_at_least_one_selector():
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        return replace(
            valid,
            scope=replace(valid.scope, kind=SemanticWorkflowScopeKind.MODULE, selectors=()),
        )

    with pytest.raises(ValueError, match="at least one entity"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


@pytest.mark.parametrize(
    "kind", [SemanticWorkflowScopeKind.UNKNOWN, SemanticWorkflowScopeKind.REPOSITORY]
)
def test_broad_scope_requires_no_selectors(kind):
    history, static, candidates = _fixture()

    def response(request):
        valid = _valid_response(request)
        return replace(valid, scope=replace(valid.scope, kind=kind, selectors=("pkg.a",)))

    with pytest.raises(ValueError, match="must not have selectors"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


def test_workflow_decision_requires_complete_workflow():
    history, static, candidates = _fixture()

    def response(request):
        return replace(FakeModel().complete(request), workflow=None)

    with pytest.raises(ValueError, match="requires a complete workflow"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


@pytest.mark.parametrize(
    "decision", [SemanticDecisionKind.INCOHERENT, SemanticDecisionKind.INSUFFICIENT_EVIDENCE]
)
def test_rejection_cannot_carry_a_workflow(decision):
    history, static, candidates = _fixture()

    def response(request):
        return replace(FakeModel().complete(request), decision=decision)

    with pytest.raises(ValueError, match="must not include a workflow"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


def test_rejection_requires_known_evidence_and_retains_candidate():
    history, static, candidates = _fixture()

    def rejection(request):
        return replace(
            FakeModel().complete(request),
            decision=SemanticDecisionKind.INSUFFICIENT_EVIDENCE,
            reason="The supplied changes only establish co-change",
            decision_evidence_ids=(request.evidence[0].evidence_id,) * 2,
            workflow=None,
        )

    output = SemanticWorkflowAbstracter().abstract(
        history, static, candidates, FakeModel(rejection)
    )
    assert output.workflows == ()
    assert len(output.rejections) == 1
    item = output.rejections[0]
    assert item.candidate_id == candidates.candidates[0].candidate_id
    assert item.evidence_ids == (item.evidence_ids[0],)
    assert item.confidence.method == CONFIDENCE_METHOD
    assert output.evidence[0].evidence_id == item.evidence_ids[0]


def test_rejection_with_unknown_evidence_is_rejected():
    history, static, candidates = _fixture()

    def response(request):
        return replace(
            FakeModel().complete(request),
            decision=SemanticDecisionKind.INCOHERENT,
            workflow=None,
            decision_evidence_ids=("unknown-evidence",),
        )

    with pytest.raises(ValueError, match="unknown Candidate Evidence"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))


def test_rejection_reason_cannot_fabricate_module():
    history, static, candidates = _fixture()

    def response(request):
        return replace(
            FakeModel().complete(request),
            decision=SemanticDecisionKind.INCOHERENT,
            workflow=None,
            reason="The invented.module patterns are incompatible",
        )

    with pytest.raises(ValueError, match="unknown module"):
        SemanticWorkflowAbstracter().abstract(history, static, candidates, FakeModel(response))
