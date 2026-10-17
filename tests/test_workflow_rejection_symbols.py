"""A rejected workflow may explain cited symbols without publishing them."""

from dataclasses import replace

import pytest
from test_semantic_workflows import FakeModel, _fixture

from skillforge.domain.semantic_workflow import (
    SemanticDecisionKind,
    SemanticEvidenceRef,
    SemanticWorkflowScopeKind,
)
from skillforge.repository.semantic_workflows import (
    _validate_decision,
    build_semantic_workflow_request,
)


def request():
    history, static, candidates = _fixture()
    req = build_semantic_workflow_request(history, static, candidates, candidates.candidates[0])
    return replace(req, family_supporting_commits=tuple(
        replace(commit, message="Fix Store.lookup and Parser.match; inspect Notes.md")
        for commit in req.family_supporting_commits
    ))


def rejection(req, reason, evidence_ids=None):
    return replace(
        FakeModel().complete(req), decision=SemanticDecisionKind.INCOHERENT,
        workflow=None, reason=reason,
        decision_evidence_ids=(tuple(
            item.evidence_id for item in req.evidence if item.provenance == "git_log"
        ) if evidence_ids is None else tuple(evidence_ids)),
    )


def test_exact_cited_members_explain_rejection_without_creating_knowledge():
    req = request()
    draft, rejected, _ = _validate_decision(
        req, rejection(req, "Store.lookup and Parser.match are separate repairs.")
    )
    assert draft is None
    assert rejected.decision == SemanticDecisionKind.INCOHERENT


def test_explicit_method_groups_are_not_specific_to_one_library_or_method_pair():
    req = request()
    req = replace(req, family_supporting_commits=tuple(
        replace(commit, message='Store.read/write/close have distinct contracts')
        for commit in req.family_supporting_commits
    ))
    draft, rejected, _ = _validate_decision(
        req, rejection(req, 'Store.read, Store.write and Store.close differ.')
    )
    assert draft is None and rejected.decision == SemanticDecisionKind.INCOHERENT
    with pytest.raises(ValueError, match='unknown module'):
        _validate_decision(req, rejection(req, 'Store.delete is also involved.'))


@pytest.mark.parametrize("token", [
    "Unknown.lookup", "Store.lookups", "src/Store.lookup", "Other.secret",
])
def test_unknown_symbols_and_files_remain_invalid(token):
    req = request()
    with pytest.raises(ValueError, match="unknown (path|module)"):
        _validate_decision(req, rejection(req, f"{token} is relevant."))


def test_uncited_log_does_not_authorize_member():
    req = request()
    cited_numstat = next(item for item in req.evidence if item.provenance == "git_numstat")
    response = replace(rejection(req, "Store.lookup is relevant."),
                       decision_evidence_ids=(cited_numstat.evidence_id,))
    with pytest.raises(ValueError, match="unknown module"):
        _validate_decision(req, response)


def test_cited_prose_filename_can_be_named_in_rejection_reason():
    req = request()
    draft, rejected, _ = _validate_decision(
        req, rejection(req, "The cited commit mentions Notes.md.")
    )
    assert draft is None
    assert rejected.decision == SemanticDecisionKind.INCOHERENT


def test_cited_diff_authorizes_only_exact_methods_from_its_commit_message():
    req = request()
    first, second = req.family_supporting_commits
    messages = {
        first.commit_id: "Fix Query.any/all",
        second.commit_id: "Fix Query.test",
    }
    req = replace(req, family_supporting_commits=tuple(
        replace(commit, message=messages[commit.commit_id])
        for commit in req.family_supporting_commits
    ))
    cited_diff = next(item for item in req.evidence
                      if item.provenance == "git_diff" and item.commit_id == first.commit_id)

    draft, rejected, _ = _validate_decision(
        req, rejection(req, "Query.any and Query.all are named by this change.",
                       (cited_diff.evidence_id,))
    )
    assert draft is None
    assert rejected.decision == SemanticDecisionKind.INCOHERENT
    with pytest.raises(ValueError, match="unknown module: Query.test"):
        _validate_decision(
            req, rejection(req, "Query.test belongs to a different change.",
                           (cited_diff.evidence_id,))
        )
    with pytest.raises(ValueError, match="unknown path: Query.any/other"):
        _validate_decision(
            req, rejection(req, "Query.any/other is not an any/all pair.",
                           (cited_diff.evidence_id,))
        )


def test_cited_numstat_alone_does_not_authorize_commit_message_symbols():
    req = request()
    first = req.family_supporting_commits[0]
    req = replace(req, family_supporting_commits=(
        replace(first, message="Fix Query.any/all"),
        *req.family_supporting_commits[1:],
    ))
    cited_numstat = next(item for item in req.evidence
                         if item.provenance == "git_numstat" and item.commit_id == first.commit_id)
    with pytest.raises(ValueError, match="unknown module: Query.any"):
        _validate_decision(
            req, rejection(req, "Query.any is not grounded by file counts.",
                           (cited_numstat.evidence_id,))
        )


def test_cited_python_implementation_authorizes_only_its_heading_symbol():
    req = request()
    implementation = SemanticEvidenceRef(
        evidence_id="cited-python-implementation",
        kind="source",
        path="tinydb/queries.py",
        commit_id=None,
        line=383,
        description=("Python implementation tinydb.queries.Query.test: "
                     "def test(self, fn, *args): return QueryInstance("
                     "lambda value: fn(value, *args), ('Query.forged', args))"),
        provenance="python_ast",
    )
    req = replace(req, evidence=(*req.evidence, implementation))
    draft, rejected, _ = _validate_decision(
        req, rejection(req, "Query.test is the cited implementation.",
                       (implementation.evidence_id,))
    )
    assert draft is None
    assert rejected.decision == SemanticDecisionKind.INCOHERENT
    with pytest.raises(ValueError, match="unknown module: Query.forged"):
        _validate_decision(
            req, rejection(req, "Query.forged appears only in the code body.",
                           (implementation.evidence_id,))
        )


@pytest.mark.parametrize(("kind", "selector", "message"), [
    (SemanticWorkflowScopeKind.MODULE, "Query.any", "unknown module"),
    (SemanticWorkflowScopeKind.FILE, "Query.any", "unknown path"),
    (SemanticWorkflowScopeKind.FILE, "Notes.md", "unknown path"),
])
def test_rejection_symbols_do_not_authorize_scope_selectors(kind, selector, message):
    req = request()
    response = FakeModel().complete(req)
    workflow = replace(response.workflow, scope=replace(
        response.workflow.scope, kind=kind, selectors=(selector,),
    ))
    with pytest.raises(ValueError, match=message):
        _validate_decision(req, replace(response, workflow=workflow))


def test_accepted_reason_and_steps_keep_strict_validation():
    req = request()
    response = FakeModel().complete(req)
    with pytest.raises(ValueError, match="unknown module"):
        _validate_decision(req, replace(response, reason="Store.lookup is relevant."))
    step = response.workflow.ordered_steps[0]
    with pytest.raises(ValueError, match="unknown module"):
        _validate_decision(req, replace(response, workflow=replace(
            response.workflow, ordered_steps=(
                replace(step, action="Repair Store.lookup."),
                *response.workflow.ordered_steps[1:],
            ),
        )))
