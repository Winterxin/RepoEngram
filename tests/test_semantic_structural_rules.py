"""Semantic structural gate, failure paths, and Project DNA integration."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.evidence import Confidence, Support
from skillforge.domain.project_dna import ProjectDNA, ProjectKnowledgeStatus
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_structural_rule import (
    SEMANTIC_METHOD,
    SemanticStructuralAcceptance,
    SemanticStructuralException,
    SemanticStructuralRequest,
    SemanticStructuralResponse,
)
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralDecisionKind as Decision,
)
from skillforge.domain.semantic_workflow import (
    SemanticStepClassification,
    SemanticWorkflowDraft,
    SemanticWorkflowResult,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.skills import RuleGuidanceSkill, SkillConstraintEnforcement
from skillforge.domain.structural_rules import StructuralRuleCandidate
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.repo_local import RepoLocalSkillSync
from skillforge.productization.evolution_artifacts import _restore
from skillforge.repository.codex_cli_structural import (
    CodexCliStructuralSemanticModel,
    parse_semantic_structural_response,
)
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.semantic_structural_prompt import build_semantic_structural_prompt
from skillforge.repository.semantic_structural_rules import (
    SemanticStructuralRuleEvaluator,
    build_semantic_structural_request,
    validate_semantic_structural_response,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


@pytest.fixture
def case(tmp_path):
    for package in ("api", "service"):
        path = tmp_path / "src" / "app" / package / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for index in range(3):
        (tmp_path / "src" / "app" / "api" / f"route{index}.py").write_text(
            f"import app.service.unit{index % 2}\n", encoding="utf-8"
        )
    for index in range(2):
        (tmp_path / "src" / "app" / "service" / f"unit{index}.py").write_text(
            "", encoding="utf-8"
        )
    mined = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(tmp_path))
    request = build_semantic_structural_request(mined, mined.candidates[0])
    refs = tuple(item.evidence_id for item in request.evidence)
    response = SemanticStructuralResponse(
        request.repository_id, request.candidate_id, Decision.ARCHITECTURE_RULE,
        "Observed imports connect app.api to app.service across source modules.",
        refs, request.source_package, request.target_package,
        request.source_modules, refs, (),
    )
    return mined, request, response


class FixedModel:
    def __init__(self, response):
        self.response = response

    def complete(self, request):
        return self.response


def test_accepted_candidate_stays_observational_and_advisory(case):
    mined, request, response = case
    accepted = validate_semantic_structural_response(request, response, mined.candidates[0])
    assert isinstance(accepted, SemanticStructuralAcceptance)
    evaluator = SemanticStructuralRuleEvaluator()
    result = evaluator.evaluate(mined, FixedModel(response))
    assert result.accepted == (accepted,)
    assert accepted.candidate is mined.candidates[0]
    assert accepted.decision == Decision.ARCHITECTURE_RULE
    assert accepted.reason == response.reason
    assert accepted.decision_evidence_ids == response.decision_evidence_ids
    assert accepted.scope_evidence_ids == response.scope_evidence_ids
    assert accepted.method == SEMANTIC_METHOD
    serialized = result.to_dict()["accepted"][0]
    assert serialized["reason"] == response.reason
    assert serialized["decision_evidence_ids"] == list(response.decision_evidence_ids)
    assert serialized["method"] == SEMANTIC_METHOD
    assert result.evidence == mined.evidence
    dna = ProjectDNABuilder().build(mined.repository_id, semantic_structural_rules=result)
    assert dna.rules == (mined.candidates[0].rule,)
    assert dna.rules[0].status == ProjectKnowledgeStatus.CANDIDATE
    assert dna.rules[0].confidence is None
    assert dna.rules[0].statement.startswith("Observed structural pattern:")
    assert SkillCompiler().compile(dna).skills == ()
    assert result.to_dict() == evaluator.evaluate(mined, FixedModel(response)).to_dict()
    assert dna.snapshot_id == ProjectDNABuilder().build(
        mined.repository_id, semantic_structural_rules=result
    ).snapshot_id
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    evidence = mined.evidence[0]
    ref = evidence.evidence_id
    workflow = SemanticWorkflowResult(mined.repository_id, (
        SemanticWorkflowDraft(
            draft_id="workflow", repository_id=mined.repository_id,
            candidate_id="workflow-candidate", name="Update application",
            name_evidence_ids=(ref,), intent="Update application",
            intent_evidence_ids=(ref,), roles=(),
            ordered_steps=(SemanticWorkflowStep("inspect", "Inspect current application", (ref,)),),
            required_steps=(SemanticStepClassification("inspect", "source step", (ref,)),),
            optional_steps=(), conditional_steps=(),
            scope=SemanticWorkflowScope(
                SemanticWorkflowScopeKind.MODULE, (request.source_modules[0],),
                "module", "observed", (ref,),
            ),
            exceptions=(), evidence_ids=(ref,), support=Support(2, 2),
            confidence=Confidence(0.8, "fixture", Support(2, 2)),
        ),
    ), (evidence,), ())
    dna_with_workflow = ProjectDNABuilder().build(
        mined.repository_id, workflows=workflow, semantic_structural_rules=result
    )
    skills = SkillCompiler().compile(dna_with_workflow).skills
    assert len(skills) == 1
    assert skills[0].constraints[0].enforcement == SkillConstraintEnforcement.ADVISORY
    assert skills[0].verify_rule_ids == ()
    combined = SkillCompiler().compile(dna_with_workflow, result)
    assert skills[0] in combined.skills
    assert len(combined.skills) == 2
    assert len({item.skill_id for item in combined.skills}) == 2
    registry = SkillRegistryBuilder().build(combined)
    assert tuple(item.skill_id for item in registry.skills) == tuple(sorted(
        item.skill_id for item in combined.skills
    ))
    workflow_route = SkillRouter().route(registry, SkillRoutingRequest(
        dna_with_workflow.repository_id, "Update application",
    ))
    workflow_package = next(item for item in AgentSkillExporter().plan(
        workflow_route, CodeAgentTarget.CODEX
    ).packages if item.source_skill_id == skills[0].skill_id)
    marker = json.loads(next(item.content for item in workflow_package.files
                             if item.relative_path == ".skillforge.json"))
    assert "source_kind" not in marker


def test_rule_guidance_compiles_routes_and_exports(case, tmp_path):
    mined, request, response = case
    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, FixedModel(response))
    dna = ProjectDNABuilder().build(mined.repository_id, semantic_structural_rules=semantic)
    compiler = SkillCompiler()
    assert compiler.compile(dna).skills == ()
    compiled = compiler.compile(dna, semantic)
    assert compiled.to_dict() == compiler.compile(dna, semantic).to_dict()
    assert len(compiled.skills) == 1
    skill = compiled.skills[0]
    assert isinstance(skill, RuleGuidanceSkill)
    assert skill.description == dna.rules[0].statement
    assert skill.source_scope == dna.rules[0].scope
    assert skill.source_confidence is None
    assert skill.source_status == ProjectKnowledgeStatus.CANDIDATE
    assert skill.constraint.enforcement == SkillConstraintEnforcement.ADVISORY
    assert not hasattr(skill, "workflow_id")
    assert not hasattr(skill, "steps")
    assert not hasattr(skill, "completion_conditions")
    assert not hasattr(skill, "verify_rule_ids")
    assert compiled.evidence == dna.evidence
    with pytest.raises(ValueError, match="snapshot"):
        replace(compiled, source_snapshot_id="other-snapshot")
    registry = SkillRegistryBuilder().build(compiled)
    route = SkillRouter().route(registry, SkillRoutingRequest(
        dna.repository_id, "unrelated wording", current_module=request.source_modules[0],
    ))
    assert tuple(match.skill for match in route.matches) == (skill,)
    unrelated = SkillRouter().route(registry, SkillRoutingRequest(
        dna.repository_id, "unrelated wording", current_module="app.other.module",
    ))
    assert unrelated.matches == ()
    for target in CodeAgentTarget:
        plan = AgentSkillExporter().plan(route, target)
        package = plan.packages[0]
        files = {item.relative_path: item.content for item in package.files}
        assert "# Decision procedure / Steps" in files["SKILL.md"]
        assert "# Verify" in files["SKILL.md"]
        assert "Source authority: advisory" in files["SKILL.md"]
        assert semantic.accepted[0].reason in files["SKILL.md"]
        assert semantic.accepted[0].reason in files["references/evidence.md"]
        assert json.loads(files[".skillforge.json"])["source_kind"] == "RULE"
        first = RepoLocalSkillSync().sync(tmp_path, plan)
        second = RepoLocalSkillSync().sync(tmp_path, plan)
        assert len(first.written_files) == 3
        assert len(second.unchanged_files) == 3


def test_rule_guidance_rejects_mismatched_audit(case):
    mined, _, response = case
    semantic = SemanticStructuralRuleEvaluator().evaluate(mined, FixedModel(response))
    dna = ProjectDNABuilder().build(mined.repository_id, semantic_structural_rules=semantic)
    accepted = semantic.accepted[0]
    with pytest.raises(ValueError):
        SkillCompiler().compile(dna, replace(semantic, repository_id="other"))
    with pytest.raises(ValueError):
        SkillCompiler().compile(dna, replace(
            semantic, evidence=(replace(semantic.evidence[0], description="tampered"),
                                *semantic.evidence[1:]),
        ))
    with pytest.raises(ValueError):
        SkillCompiler().compile(dna, replace(
            semantic, accepted=(replace(accepted, reason="Modules must always align."),),
        ))
    with pytest.raises(ValueError):
        SkillCompiler().compile(dna, replace(
            semantic, accepted=(replace(accepted, candidate=replace(
                accepted.candidate, rule=replace(accepted.candidate.rule,
                    statement="Observed structural pattern: different."))),),
        ))
    rejected = SemanticStructuralRuleEvaluator().evaluate(
        mined, FixedModel(replace(response, decision=Decision.INCOHERENT,
                                  scope_modules=(), scope_evidence_ids=())),
    )
    assert SkillCompiler().compile(dna, rejected).skills == ()
    deprecated = ProjectDNA(
        dna.repository_id,
        (replace(dna.rules[0], status=ProjectKnowledgeStatus.DEPRECATED),),
        (), (), dna.evidence,
    )
    assert SkillCompiler().compile(deprecated, semantic).skills == ()


@pytest.mark.parametrize("decision", [Decision.INCOHERENT, Decision.INSUFFICIENT_EVIDENCE])
def test_rejections_do_not_enter_dna(case, decision):
    mined, request, response = case
    response = replace(response, decision=decision, scope_modules=(),
                       scope_evidence_ids=())
    result = SemanticStructuralRuleEvaluator().evaluate(mined, FixedModel(response))
    assert result.accepted == ()
    assert result.rejections[0].decision == decision
    assert {item.evidence_id for item in result.evidence} == set(response.decision_evidence_ids)
    assert ProjectDNABuilder().build(
        mined.repository_id, semantic_structural_rules=result
    ).rules == ()


@pytest.mark.parametrize("changes", [
    {"repository_id": "other"},
    {"candidate_id": "unknown"},
    {"decision_evidence_ids": ("invented",)},
    {"scope_evidence_ids": ("invented",)},
    {"scope_modules": ("app.api.invented",)},
    {"source_package": "app.other"},
    {"target_package": "app.other"},
    {"reason": "Observed app.other.unit import."},
    {"reason": "Observed src/app/other.py import."},
    {"reason": "app.api MUST import app.service."},
    {"reason": "app.api should use app.service."},
    {"reason": "app.api ought to use app.service."},
    {"reason": "app.api requires app.service."},
    {"reason": "app.api require app.service."},
    {"reason": "app.api mandates app.service."},
    {"reason": "app.api mandate app.service."},
    {"exceptions": (SemanticStructuralException("Exception for app.api", ("invented",)),)},
])
def test_hallucinations_fail_closed(case, changes):
    mined, request, response = case
    altered = replace(response, **changes)
    with pytest.raises(ValueError):
        validate_semantic_structural_response(request, altered, mined.candidates[0])
    result = SemanticStructuralRuleEvaluator().evaluate(mined, FixedModel(altered))
    assert result.accepted == ()
    assert len(result.failures) == 1


def test_known_evidence_path_is_allowed_in_explanation(case):
    mined, request, response = case
    grounded = replace(response, reason=f"Observed import at {request.evidence[0].path}")
    assert isinstance(
        validate_semantic_structural_response(request, grounded, mined.candidates[0]),
        SemanticStructuralAcceptance,
    )


@pytest.mark.parametrize("field", ["decision_evidence_ids", "scope_evidence_ids"])
def test_shuffled_evidence_ids_are_canonicalized(case, field):
    mined, request, response = case
    shuffled = replace(response, **{field: tuple(reversed(getattr(response, field)))})
    accepted = validate_semantic_structural_response(request, shuffled, mined.candidates[0])
    assert accepted.decision_evidence_ids == tuple(sorted(response.decision_evidence_ids))
    assert accepted.scope_evidence_ids == tuple(sorted(response.scope_evidence_ids))
    assert getattr(shuffled, field) == tuple(reversed(getattr(response, field)))


@pytest.mark.parametrize("field", ["decision_evidence_ids", "scope_evidence_ids"])
@pytest.mark.parametrize("invalid", ["duplicate", "unknown", "empty", "list", "type"])
def test_invalid_evidence_ids_are_not_repaired(case, field, invalid):
    mined, request, response = case
    refs = getattr(response, field)
    values = {
        "duplicate": (*refs, refs[0]), "unknown": (*refs, "foreign-evidence"),
        "empty": (), "list": list(refs), "type": (*refs, []),
    }
    with pytest.raises(ValueError, match="Evidence IDs"):
        validate_semantic_structural_response(
            request, replace(response, **{field: values[invalid]}), mined.candidates[0])


@pytest.mark.parametrize("claim", [
    "application.tasks", "apps.api.app.application.tasks.task_service",
])
def test_grounded_namespace_shorthand_and_exact_identifier_pass(case, claim):
    mined, request, response = case
    request = replace(request, target_modules=(*request.target_modules,
                      "apps.api.app.application.tasks.task_service",
                      "apps.api.app.application.tasks.task_service.TaskService"))
    accepted = validate_semantic_structural_response(
        request, replace(response, reason=f"Observed {claim} delegation."), mined.candidates[0])
    assert isinstance(accepted, SemanticStructuralAcceptance)


@pytest.mark.parametrize("claim", [
    "application.magic", "unknown.namespace", "application.task",
    "alias.tasks", "invented/application.tasks",
])
def test_ungrounded_namespace_references_fail(case, claim):
    mined, request, response = case
    request = replace(request, target_modules=(*request.target_modules,
                      "apps.api.app.application.tasks.task_service"))
    with pytest.raises(ValueError, match="unknown module|unknown path"):
        validate_semantic_structural_response(
            request, replace(response, reason=f"Observed {claim} delegation."),
            mined.candidates[0])


def test_namespace_shorthand_with_unrelated_groundings_fails(case):
    mined, request, response = case
    request = replace(request, target_modules=(*request.target_modules,
                      "apps.api.app.application.tasks.task_service",
                      "other.application.tasks.worker"))
    with pytest.raises(ValueError, match="unknown module: application.tasks"):
        validate_semantic_structural_response(
            request, replace(response, reason="Observed application.tasks delegation."),
            mined.candidates[0])


def test_saved_real_u04_response_is_accepted():
    saved = json.loads((Path(__file__).parent / "fixtures" /
                        "u04_structural_semantic_response.json").read_text(encoding="utf-8"))
    candidate = _restore(StructuralRuleCandidate, saved["candidate"])
    request = _restore(SemanticStructuralRequest, saved["request"])
    response = parse_semantic_structural_response(saved["response"])
    assert candidate.delegation_pattern == "REPEATED_OWNER_DELEGATION"
    assert candidate.rule.support == Support(5, 6)
    assert "application.tasks" in response.reason
    assert response.decision_evidence_ids != tuple(sorted(response.decision_evidence_ids))
    assert response.scope_evidence_ids != tuple(sorted(response.scope_evidence_ids))
    accepted = validate_semantic_structural_response(request, response, candidate)
    assert isinstance(accepted, SemanticStructuralAcceptance)
    assert accepted.reason == response.reason
    assert accepted.candidate == candidate
    assert accepted.decision_evidence_ids == tuple(sorted(response.decision_evidence_ids))
    assert accepted.scope_evidence_ids == tuple(sorted(response.scope_evidence_ids))


def test_accepted_artifact_identity_decision_and_evidence_are_closed(case):
    mined, request, response = case
    accepted = validate_semantic_structural_response(request, response, mined.candidates[0])
    with pytest.raises(ValueError, match="identity"):
        replace(accepted, candidate_id="other")
    with pytest.raises(ValueError, match="ARCHITECTURE_RULE"):
        replace(accepted, decision=Decision.INCOHERENT)
    with pytest.raises(ValueError, match="Evidence"):
        replace(accepted, decision_evidence_ids=("invented",))
    with pytest.raises(ValueError, match="Evidence"):
        replace(accepted, scope_evidence_ids=("invented",))
    with pytest.raises(ValueError, match="method"):
        replace(accepted, method="other")


def test_model_failure_and_malformed_response_fail_closed(case):
    mined, request, response = case
    result = SemanticStructuralRuleEvaluator().evaluate(mined, FixedModel(None))
    assert not result.accepted and len(result.failures) == 1
    with pytest.raises(ValueError):
        parse_semantic_structural_response({"decision": "ARCHITECTURE_RULE"})
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_semantic_structural_response({**response.to_dict(), "name": "Controller Layer"})
    prompt = build_semantic_structural_prompt(request)
    assert request.candidate_id in prompt
    assert request.evidence[0].path in prompt
    assert "src/app/other.py" not in prompt


def test_codex_adapter_only_parses_and_gate_still_rejects_hallucination(case):
    mined, request, response = case
    forged = replace(response, decision_evidence_ids=("invented",))

    def runner(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps(forged.to_dict()), encoding="utf-8"
        )
        assert request.candidate_id in kwargs["input"]
        return subprocess.CompletedProcess(command, 0, "", "")

    model = CodexCliStructuralSemanticModel(runner=runner)
    parsed = model.complete(request)
    assert parsed == forged
    result = SemanticStructuralRuleEvaluator().evaluate(mined, model)
    assert result.accepted == () and len(result.failures) == 1
