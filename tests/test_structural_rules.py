"""Evidence-backed structural candidate mining and integration boundaries."""

import json
from dataclasses import replace

import pytest

from skillforge.domain.evidence import Confidence, EvidenceType, Support
from skillforge.domain.project_dna import ProjectKnowledgeStatus, RuleType
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.semantic_workflow import (
    SemanticStepClassification,
    SemanticWorkflowDraft,
    SemanticWorkflowResult,
    SemanticWorkflowScope,
    SemanticWorkflowScopeKind,
    SemanticWorkflowStep,
)
from skillforge.domain.skills import SkillConstraintEnforcement
from skillforge.domain.static_analysis import ImportResolution
from skillforge.domain.structural_rules import (
    StructuralRuleMiningResult,
    structural_rule_id,
    structural_rule_statement,
)
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner


def write(root, path, contents=""):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")


def repository(root, *, reverse=False, extra=""):
    for name in ("api", "service"):
        write(root, f"src/app/{name}/__init__.py")
    for index in range(3):
        write(root, f"src/app/api/route{index}.py", f"import app.service.unit{index % 2}\n")
    for index in range(2):
        write(root, f"src/app/service/unit{index}.py")
    if reverse:
        write(root, "src/app/service/back.py", "import app.api.route0\n")
    if extra:
        write(root, "src/app/api/extra.py", extra)
    facts = PythonStaticAnalyzer().analyze(root)
    return facts, StructuralRuleMiner().mine(facts)


def workflow(repository_id, evidence, source_module):
    ref = evidence.evidence_id
    draft = SemanticWorkflowDraft(
        draft_id="workflow",
        repository_id=repository_id,
        candidate_id="workflow-candidate",
        name="Update application",
        name_evidence_ids=(ref,),
        intent="Update application",
        intent_evidence_ids=(ref,),
        roles=(),
        ordered_steps=(
            SemanticWorkflowStep("inspect", "Inspect observed source structure", (ref,)),
        ),
        required_steps=(),
        optional_steps=(SemanticStepClassification("inspect", "Source observation", (ref,)),),
        conditional_steps=(),
        scope=SemanticWorkflowScope(
            SemanticWorkflowScopeKind.MODULE,
            (source_module,),
            "module",
            "observed",
            (ref,),
        ),
        exceptions=(),
        evidence_ids=(ref,),
        support=Support(2, 2),
        confidence=Confidence(0.8, "fixture", Support(2, 2)),
    )
    return SemanticWorkflowResult(repository_id, (draft,), (evidence,), ())


def test_repeated_pattern_is_stable_evidence_closed_and_advisory(tmp_path):
    facts, result = repository(tmp_path)
    again = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(tmp_path))
    assert result == again
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    rule = candidate.rule
    assert candidate.method == "python_ast_sibling_package_imports_v1"
    assert (candidate.source_package, candidate.target_package) == ("app.api", "app.service")
    assert len(candidate.source_modules) == 3
    assert len(candidate.target_modules) == 2
    assert rule.type == RuleType.ARCHITECTURE
    assert rule.status == ProjectKnowledgeStatus.CANDIDATE
    assert rule.support == Support(3, 3)
    assert rule.confidence is None
    assert rule.scope.selectors == candidate.source_modules
    assert set(rule.evidence_ids) == {item.evidence_id for item in result.evidence}
    assert all(item.provenance == "python_ast" for item in result.evidence)
    assert all(item.repository_id == facts.repository_id for item in result.evidence)
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    assert result.to_dict() == again.to_dict()

    dna = ProjectDNABuilder().build(facts.repository_id, structural_candidates=result)
    assert dna.rules == (rule,)
    assert dna.evidence == result.evidence
    assert json.loads(json.dumps(dna.to_dict())) == dna.to_dict()
    assert SkillCompiler().compile(dna).skills == ()

    wf = workflow(facts.repository_id, facts.evidence[0], candidate.source_modules[0])
    dna_with_workflow = ProjectDNABuilder().build(
        facts.repository_id, structural_candidates=result, workflows=wf
    )
    skills = SkillCompiler().compile(dna_with_workflow).skills
    assert len(skills) == 1
    constraint = skills[0].constraints[0]
    assert constraint.rule_id == rule.rule_id
    assert constraint.enforcement == SkillConstraintEnforcement.ADVISORY
    assert skills[0].verify_rule_ids == ()
    other_workflow = workflow(facts.repository_id, facts.evidence[0], candidate.target_modules[0])
    unrelated = ProjectDNABuilder().build(
        facts.repository_id, structural_candidates=result, workflows=other_workflow
    )
    assert SkillCompiler().compile(unrelated).skills[0].constraints == ()

    baseline = ProjectDNABuilder().build(facts.repository_id, workflows=wf)
    request = SkillRoutingRequest(
        facts.repository_id, "Update application", current_module=candidate.source_modules[0]
    )
    def route(project_dna):
        registry = SkillRegistryBuilder().build(SkillCompiler().compile(project_dna))
        return SkillRouter().route(registry, request)

    before = route(baseline)
    after = route(dna_with_workflow)
    assert tuple((item.skill.skill_id, item.rank_key) for item in before.matches) == tuple(
        (item.skill.skill_id, item.rank_key) for item in after.matches
    )


def test_isolated_external_unresolved_and_parse_error_do_not_mine(tmp_path):
    write(tmp_path, "src/app/api/one.py", "import app.service.unit0\nimport requests\n")
    write(tmp_path, "src/app/api/two.py", "import app.missing.part\n")
    write(tmp_path, "src/app/api/bad.py", "def broken(:\n")
    write(tmp_path, "src/app/service/unit0.py")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    resolutions = {item.resolution for module in facts.modules for item in module.imports}
    assert ImportResolution.INTERNAL in resolutions
    assert ImportResolution.EXTERNAL in resolutions
    assert ImportResolution.UNRESOLVED in resolutions
    assert facts.issues
    assert StructuralRuleMiner().mine(facts).candidates == ()


def test_reverse_direction_suppresses_candidate(tmp_path):
    _, result = repository(tmp_path, reverse=True)
    assert result.candidates == ()
    assert result.evidence == ()


def test_from_import_of_known_internal_modules_is_counted(tmp_path):
    for index in range(3):
        write(
            tmp_path,
            f"src/app/api/route{index}.py",
            f"from app.service.unit{index % 2} import VALUE\n",
        )
    for index in range(2):
        write(tmp_path, f"src/app/service/unit{index}.py", "VALUE = 1\n")
    result = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(tmp_path))
    assert len(result.candidates) == 1
    assert len(result.candidates[0].target_modules) == 2


def test_ambiguous_package_from_import_is_not_treated_as_module_dependency(tmp_path):
    for index in range(3):
        write(tmp_path, f"src/app/api/route{index}.py", "from app.service import unit0\n")
    write(tmp_path, "src/app/service/unit0.py")
    write(tmp_path, "src/app/service/unit1.py")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    assert StructuralRuleMiner().mine(facts).candidates == ()


def test_duplicate_imports_in_one_source_do_not_count_as_multiple_occurrences(tmp_path):
    write(tmp_path, "src/app/api/one.py", "import app.service.a\nimport app.service.b\n")
    write(tmp_path, "src/app/service/a.py")
    write(tmp_path, "src/app/service/b.py")
    facts = PythonStaticAnalyzer().analyze(tmp_path)
    assert StructuralRuleMiner().mine(facts).candidates == ()


def test_duplicate_evidence_cross_repository_and_status_tampering_are_rejected(tmp_path):
    facts, result = repository(tmp_path)
    with pytest.raises(ValueError, match="duplicate static Evidence"):
        StructuralRuleMiner().mine(replace(facts, evidence=(*facts.evidence, facts.evidence[0])))
    with pytest.raises(ValueError, match="repository_id mismatch"):
        ProjectDNABuilder().build("other", structural_candidates=result)
    with pytest.raises(ValueError, match="repository_id mismatch"):
        StructuralRuleMiner().mine(
            replace(
                facts,
                evidence=(
                    replace(facts.evidence[0], repository_id="other"),
                    *facts.evidence[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="CANDIDATE"):
        replace(result.candidates[0], rule=replace(
            result.candidates[0].rule, status=ProjectKnowledgeStatus.CONFIRMED
        ))


def test_identity_ignores_additional_occurrence_and_unrelated_files(tmp_path):
    _, first = repository(tmp_path)
    write(tmp_path, "src/app/api/route3.py", "import app.service.unit1\n")
    write(tmp_path, "src/unrelated.py", "import json\n")
    _, second = repository(tmp_path)
    assert first.candidates[0].rule.rule_id == second.candidates[0].rule.rule_id
    assert first.candidates[0].rule.evidence_ids != second.candidates[0].rule.evidence_ids


def test_support_examines_only_direct_source_package_modules(tmp_path):
    repository(tmp_path)
    write(tmp_path, "src/app/api/route3.py", "VALUE = 1\n")
    for index in range(3):
        write(tmp_path, f"src/app/api/internal/helper{index}.py", "VALUE = 1\n")
    result = StructuralRuleMiner().mine(PythonStaticAnalyzer().analyze(tmp_path))
    candidate = result.candidates[0]
    assert candidate.rule.support == Support(3, 4)
    assert candidate.examined_modules == (
        "app.api.route0", "app.api.route1", "app.api.route2", "app.api.route3"
    )


def _replace_import_relation(facts, relation_change):
    modules = list(facts.modules)
    for index, module in enumerate(modules):
        if module.module == "app.api.route0":
            relation = relation_change(module.imports[0])
            modules[index] = replace(module, imports=(relation, *module.imports[1:]))
            return replace(facts, modules=tuple(modules))
    raise AssertionError("route0 missing")


@pytest.mark.parametrize("change", [
    lambda item: replace(item, provenance="other"),
    lambda item: replace(item, kind=EvidenceType.GIT),
    lambda item: replace(item, location=replace(item.location, path="src/app/other.py")),
    lambda item: replace(item, location=replace(item.location, line=item.location.line + 1)),
    lambda item: replace(item, description="Python symbol app.api.route0"),
])
def test_import_evidence_metadata_tampering_is_rejected(tmp_path, change):
    facts, _ = repository(tmp_path)
    relation = next(
        module.imports[0] for module in facts.modules if module.module == "app.api.route0"
    )
    evidence_id = relation.evidence_ids[0]
    altered = replace(
        facts,
        evidence=tuple(
            change(item) if item.evidence_id == evidence_id else item
            for item in facts.evidence
        ),
    )
    with pytest.raises(ValueError, match="not bound"):
        StructuralRuleMiner().mine(altered)


def test_other_ast_evidence_and_relation_content_cannot_be_substituted(tmp_path):
    facts, _ = repository(tmp_path)
    module = next(item for item in facts.modules if item.module == "app.api.route0")
    other_ast_id = module.evidence_ids[0]
    altered = _replace_import_relation(
        facts, lambda relation: replace(relation, evidence_ids=(other_ast_id,))
    )
    with pytest.raises(ValueError, match="not bound"):
        StructuralRuleMiner().mine(altered)
    altered_relation = _replace_import_relation(
        facts, lambda relation: replace(relation, raw_module="app.service.other")
    )
    with pytest.raises(ValueError, match="not bound"):
        StructuralRuleMiner().mine(altered_relation)


def test_candidate_canonical_identity_statement_modules_and_support(tmp_path):
    _, result = repository(tmp_path)
    candidate = result.candidates[0]
    rule = candidate.rule
    changes = (
        ("ID", {"rule": replace(rule, rule_id="tampered")}),
        ("statement", {"rule": replace(rule, statement="MUST use service layer")}),
        ("source modules", {"source_package": "app.other"}),
        ("sibling packages", {"source_package": "app.service"}),
        ("target modules", {"target_package": "app.other"}),
        ("Support", {"rule": replace(rule, support=Support(2, 3))}),
        ("Support", {"rule": replace(rule, support=None)}),
        ("examined modules", {"examined_modules": ("app.api.internal.helper",)}),
    )
    for message, change in changes:
        with pytest.raises(ValueError, match=message):
            replace(candidate, **change)


def test_result_rejects_duplicate_or_noncanonical_ordering(tmp_path):
    _, result = repository(tmp_path)
    candidate = result.candidates[0]
    with pytest.raises(ValueError, match="candidate IDs"):
        StructuralRuleMiningResult(result.repository_id, (candidate, candidate), result.evidence)
    with pytest.raises(ValueError, match="Evidence IDs"):
        StructuralRuleMiningResult(
            result.repository_id, result.candidates, tuple(reversed(result.evidence))
        )
    other_target = "app.storage"
    second = replace(
        candidate,
        target_package=other_target,
        target_modules=("app.storage.unit0", "app.storage.unit1"),
        rule=replace(
            candidate.rule,
            rule_id=structural_rule_id(result.repository_id, "app.api", other_target),
            statement=structural_rule_statement("app.api", other_target),
        ),
    )
    with pytest.raises(ValueError, match="candidate IDs"):
        StructuralRuleMiningResult(
            result.repository_id,
            tuple(sorted((candidate, second), key=lambda item: item.candidate_id, reverse=True)),
            result.evidence,
        )
