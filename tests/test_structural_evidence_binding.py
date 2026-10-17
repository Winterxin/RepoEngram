import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_static_testing import Model, fixture

from skillforge.domain.semantic_structural_rule import SemanticStructuralDecisionKind as Decision
from skillforge.domain.structural_rules import StructuralRuleMiningResult
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
    validate_semantic_structural_response,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.structural_rules import StructuralRuleMiner


@pytest.fixture
def case(tmp_path):
    facts = fixture(tmp_path)
    mined = StructuralRuleMiner().mine(facts)
    candidate = next(c for c in mined.candidates if c.test_linkages)
    request = build_semantic_structural_request(mined, candidate)
    response = replace(Model().complete(request), decision_evidence_ids=(),
                       scope_evidence_ids=(),
                       supporting_fact_ids=tuple(g.fact_id for g in request.evidence_groups))
    return mined, candidate, request, response


def test_complete_selection_expands_exact_evidence_and_real_scope(case):
    _, candidate, request, response = case
    accepted = validate_semantic_structural_response(request, response, candidate)
    assert accepted.decision_evidence_ids == candidate.rule.evidence_ids
    expected = tuple(sorted({key for g in request.evidence_groups for key in g.scope_evidence_ids}))
    assert accepted.scope_evidence_ids == expected
    assert set(expected) <= set(accepted.decision_evidence_ids)
    assert all(e.path == "src/nebula/entry.py" for e in request.evidence
               if e.evidence_id in expected)


@pytest.mark.parametrize("change", ["missing", "unknown", "empty", "mixed"])
def test_invalid_semantic_selection_fails_without_repair(case, change):
    _, candidate, request, response = case
    refs = response.supporting_fact_ids
    changes = {
        "missing": {"supporting_fact_ids": refs[:-1]},
        "unknown": {"supporting_fact_ids": (*refs, "fact-forged")},
        "empty": {"supporting_fact_ids": ()},
        "mixed": {"decision_evidence_ids": candidate.rule.evidence_ids},
    }
    with pytest.raises(ValueError):
        validate_semantic_structural_response(
            request, replace(response, **changes[change]), candidate)


@pytest.mark.parametrize("change", ["missing_evidence", "forged_id", "missing_scope",
                                     "wrong_scope", "forged_group"])
def test_corrupt_structured_provenance_fails(case, change):
    _, candidate, request, response = case
    group = request.evidence_groups[0]
    if change == "missing_evidence":
        request = replace(request, evidence=tuple(e for e in request.evidence
                                                 if e.evidence_id != group.scope_evidence_ids[0]))
    elif change == "wrong_scope":
        request = replace(request, evidence=tuple(replace(e, path="src/nebula/other.py")
                          if e.evidence_id in group.scope_evidence_ids else e
                          for e in request.evidence))
    else:
        fields = {"forged_id": {"evidence_ids": (*group.evidence_ids, "forged")},
                  "missing_scope": {"scope_evidence_ids": ()},
                  "forged_group": {"fact_id": "fact-forged"}}[change]
        request = replace(request, evidence_groups=(replace(group, **fields),
                                                   *request.evidence_groups[1:]))
    with pytest.raises(ValueError):
        validate_semantic_structural_response(request, response, candidate)


def test_duplicate_ordered_references_have_deterministic_closure(case):
    _, candidate, request, response = case
    expected = validate_semantic_structural_response(request, response, candidate)
    repeated = replace(response, supporting_fact_ids=tuple(reversed(
        (*response.supporting_fact_ids, *response.supporting_fact_ids))))
    assert validate_semantic_structural_response(request, repeated, candidate) == expected


def test_binding_accept_dna_compiler_and_single_adapter_call(case):
    mined, candidate, request, response = case
    subset = StructuralRuleMiningResult(mined.repository_id, (candidate,), tuple(
        e for e in mined.evidence if e.evidence_id in candidate.rule.evidence_ids))
    calls = []
    raw = response.to_dict()
    raw.pop("decision_evidence_ids")
    raw.pop("scope_evidence_ids")

    def runner(command, **kwargs):
        calls.append(command)
        assert len(calls) == 1
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert "supporting_fact_ids" in schema["required"]
        assert "decision_evidence_ids" not in schema["properties"]
        assert "evidence_groups" in kwargs["input"]
        Path(command[command.index("-o") + 1]).write_text(json.dumps(raw))
        return subprocess.CompletedProcess(command, 0, "", "")

    semantic = SemanticStructuralRuleEvaluator().evaluate(
        subset, CodexCliStructuralSemanticModel(runner=runner))
    assert len(calls) == len(semantic.accepted) == 1
    assert not semantic.failures
    dna = ProjectDNABuilder().build(mined.repository_id, semantic_structural_rules=semantic)
    assert len(dna.rules) == len(SkillCompiler().compile(dna, semantic).skills) == 1


def test_structured_rejection_still_has_no_dna_rule(case):
    mined, candidate, request, response = case
    response = replace(response, decision=Decision.INSUFFICIENT_EVIDENCE,
                       scope_modules=(), supporting_fact_ids=response.supporting_fact_ids[:1])
    rejected = validate_semantic_structural_response(request, response, candidate)
    assert set(rejected.evidence_ids) == set(request.evidence_groups[0].evidence_ids)

    class Reject:
        def complete(self, request):
            return response

    subset = StructuralRuleMiningResult(mined.repository_id, (candidate,), tuple(
        e for e in mined.evidence if e.evidence_id in candidate.rule.evidence_ids))
    result = SemanticStructuralRuleEvaluator().evaluate(subset, Reject())
    assert len(result.rejections) == 1 and not result.accepted
    assert not ProjectDNABuilder().build(mined.repository_id,
                                        semantic_structural_rules=result).rules


def test_groups_stable_when_request_rebuilt(case):
    mined, candidate, request, _ = case
    assert request == build_semantic_structural_request(mined, candidate)
