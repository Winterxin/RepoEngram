"""Deterministic promotion of validated knowledge into Project DNA."""

from __future__ import annotations

from collections.abc import Iterable

from skillforge.domain.evidence import Evidence
from skillforge.domain.project_dna import (
    ProjectAntiPattern,
    ProjectDNA,
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectWorkflow,
)
from skillforge.domain.semantic_antipattern import SemanticAntiPatternResult
from skillforge.domain.semantic_structural_rule import SemanticStructuralRuleResult
from skillforge.domain.semantic_tendency import SemanticTendencyResult
from skillforge.domain.semantic_workflow import SemanticWorkflowResult
from skillforge.domain.structural_rules import StructuralRuleMiningResult


def _index_evidence(
    items: Iterable[Evidence], repository_id: str, target: dict[str, Evidence]
) -> None:
    for item in items:
        if not item.evidence_id or not item.evidence_id.strip():
            raise ValueError("Evidence ID is required")
        if item.repository_id != repository_id:
            raise ValueError("Evidence repository_id mismatch")
        prior = target.get(item.evidence_id)
        if prior is not None and prior != item:
            raise ValueError(f"conflicting Evidence ID: {item.evidence_id}")
        target[item.evidence_id] = item


def _unique_ids(items: Iterable[object], attribute: str) -> None:
    identifiers = [getattr(item, attribute) for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate canonical {attribute}")


class ProjectDNABuilder:
    """Assembly does not infer rules, status, Support, or scope."""

    def build(
        self,
        repository_id: str,
        rules: Iterable[ProjectRule] = (),
        rule_evidence: Iterable[Evidence] = (),
        workflows: SemanticWorkflowResult | None = None,
        anti_patterns: SemanticAntiPatternResult | None = None,
        structural_candidates: StructuralRuleMiningResult | None = None,
        semantic_structural_rules: SemanticStructuralRuleResult | None = None,
        semantic_tendencies: SemanticTendencyResult | None = None,
    ) -> ProjectDNA:
        if not isinstance(repository_id, str) or not repository_id.strip():
            raise ValueError("repository_id is required")
        if (
            structural_candidates is not None
            and structural_candidates.repository_id != repository_id
        ):
            raise ValueError("structural candidate repository_id mismatch")
        if (semantic_structural_rules is not None
                and semantic_structural_rules.repository_id != repository_id):
            raise ValueError("semantic structural repository_id mismatch")
        if semantic_tendencies is not None and semantic_tendencies.repository_id != repository_id:
            raise ValueError("semantic tendency repository_id mismatch")
        rule_items = tuple(rules) + tuple(
            item.rule for item in (semantic_tendencies.accepted if semantic_tendencies else ())
        ) + tuple(
            item.rule for item in (
                structural_candidates.candidates if structural_candidates is not None else ()
            )
        ) + tuple(
            item.candidate.rule for item in (
                semantic_structural_rules.accepted if semantic_structural_rules is not None else ()
            )
        )
        _unique_ids(rule_items, "rule_id")
        if any(item.repository_id != repository_id for item in rule_items):
            raise ValueError("rule repository_id mismatch")
        for result in (workflows, anti_patterns):
            if result is not None and result.repository_id != repository_id:
                raise ValueError("semantic result repository_id mismatch")

        rule_sources: dict[str, Evidence] = {}
        _index_evidence(rule_evidence, repository_id, rule_sources)
        if semantic_tendencies is not None:
            _index_evidence(semantic_tendencies.evidence, repository_id, rule_sources)
        if structural_candidates is not None:
            _index_evidence(structural_candidates.evidence, repository_id, rule_sources)
        if semantic_structural_rules is not None:
            _index_evidence(semantic_structural_rules.evidence, repository_id, rule_sources)
        rule_refs = {key for rule in rule_items for key in rule.evidence_ids}
        missing_rules = rule_refs - rule_sources.keys()
        if missing_rules:
            raise ValueError(f"unknown rule Evidence IDs: {sorted(missing_rules)}")

        workflow_items = tuple(
            ProjectWorkflow(
                draft.draft_id, repository_id, ProjectKnowledgeStatus.CANDIDATE, draft
            )
            for draft in (workflows.workflows if workflows is not None else ())
        )
        anti_pattern_items = tuple(
            ProjectAntiPattern(
                draft.draft_id, repository_id, ProjectKnowledgeStatus.CANDIDATE, draft
            )
            for draft in (anti_patterns.anti_patterns if anti_patterns is not None else ())
        )
        _unique_ids(workflow_items, "workflow_id")
        _unique_ids(anti_pattern_items, "anti_pattern_id")

        references = set(rule_refs)
        merged: dict[str, Evidence] = {}
        _index_evidence(
            (rule_sources[key] for key in sorted(rule_refs)), repository_id, merged
        )
        for result, accepted in (
            (workflows, workflow_items),
            (anti_patterns, anti_pattern_items),
        ):
            if result is None:
                continue
            referenced = {key for entry in accepted for key in entry.draft.evidence_ids}
            source = {item.evidence_id: item for item in result.evidence}
            missing = referenced - source.keys()
            if missing:
                raise ValueError(f"unknown semantic Evidence IDs: {sorted(missing)}")
            _index_evidence(
                (source[key] for key in sorted(referenced)), repository_id, merged
            )
            references.update(referenced)
        return ProjectDNA(
            repository_id=repository_id,
            rules=tuple(sorted(rule_items, key=lambda item: item.rule_id)),
            workflows=tuple(sorted(workflow_items, key=lambda item: item.workflow_id)),
            anti_patterns=tuple(
                sorted(anti_pattern_items, key=lambda item: item.anti_pattern_id)
            ),
            evidence=tuple(merged[key] for key in sorted(references)),
        )
