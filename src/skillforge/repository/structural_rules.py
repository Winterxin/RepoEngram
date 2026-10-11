"""Conservative mining of repeated, observed Python package import direction."""

from __future__ import annotations

from collections import defaultdict

from skillforge.domain.evidence import Evidence, EvidenceType, Support
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.static_analysis import (
    ImportRelation,
    ImportResolution,
    PythonParseStatus,
    StaticAnalysisResult,
)
from skillforge.domain.structural_rules import (
    StructuralRuleCandidate,
    StructuralRuleMiningResult,
    structural_rule_id,
    structural_rule_statement,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.static_delegation import structural_delegation_candidates
from skillforge.repository.static_testing import testing_candidates


def _validated_import_evidence(
    relation: ImportRelation, sources: dict[str, Evidence], repository_id: str
) -> tuple[str, ...]:
    """Bind relation references to the analyzer's exact import Evidence identity."""
    if len(relation.evidence_ids) != 1:
        raise ValueError("import must reference exactly one Evidence")
    evidence_id = relation.evidence_ids[0]
    item = sources.get(evidence_id)
    if item is None:
        raise ValueError("import references missing Evidence")
    if item.repository_id != repository_id or item.repository_id != relation.repository_id:
        raise ValueError("import Evidence repository_id mismatch")
    if (
        item.kind != EvidenceType.SOURCE
        or item.provenance != "python_ast"
        or item.location.path != relation.path
        or item.location.line != relation.line
        or item.location.column is None
        or item.description != f"Python import {relation.raw_module}"
        or evidence_id != PythonStaticAnalyzer.expected_import_evidence_id(
            relation, item.location.column
        )
    ):
        raise ValueError("import Evidence is not bound to the import relation")
    return relation.evidence_ids


class StructuralRuleMiner:
    """Report repeated sibling-package imports, never a mandatory dependency policy."""

    def mine(self, facts: StaticAnalysisResult) -> StructuralRuleMiningResult:
        if not isinstance(facts, StaticAnalysisResult):
            raise TypeError("StructuralRuleMiner requires StaticAnalysisResult")
        sources = {item.evidence_id: item for item in facts.evidence}
        if len(sources) != len(facts.evidence):
            raise ValueError("duplicate static Evidence")
        if any(item.repository_id != facts.repository_id for item in facts.evidence):
            raise ValueError("static Evidence repository_id mismatch")
        parsed_modules = {
            item.module
            for item in facts.modules
            if item.parse_status == PythonParseStatus.PARSED and item.module is not None
        }

        # Each edge retains its raw import Evidence. Repeated imports in one module
        # count as one occurrence, preventing a single file from making a rule.
        edges: dict[tuple[str, str], dict[str, tuple[set[str], set[str]]]] = defaultdict(dict)
        for module in facts.modules:
            if module.parse_status != PythonParseStatus.PARSED or module.module is None:
                continue
            for relation in module.imports:
                if relation.repository_id != facts.repository_id:
                    raise ValueError("import repository_id mismatch")
                if (
                    relation.resolution != ImportResolution.INTERNAL
                    or relation.source_module != module.module
                    or relation.target_module is None
                ):
                    continue
                if relation.path != module.path:
                    raise ValueError("import path does not match source module")
                refs = _validated_import_evidence(relation, sources, facts.repository_id)
                # The explicit target is known; `from package import name` may
                # resolve to an attribute instead of a same-named submodule.
                for target_module in ({relation.target_module} & parsed_modules):
                    source_parts = relation.source_module.split(".")
                    target_parts = target_module.split(".")
                    if len(source_parts) < 3 or len(target_parts) < 3:
                        continue
                    source_package = ".".join(source_parts[:-1])
                    target_package = ".".join(target_parts[:-1])
                    if (
                        source_package == target_package
                        or source_parts[:-2] != target_parts[:-2]
                    ):
                        continue
                    pair = (source_package, target_package)
                    prior = edges[pair].get(module.module)
                    if prior is None:
                        edges[pair][module.module] = (
                            {target_module}, set(refs)
                        )
                    else:
                        # A source may import several targets; preserve all provenance.
                        targets_for_source, refs = prior
                        targets_for_source.add(target_module)
                        refs.update(relation.evidence_ids)

        candidates = []
        for (source_package, target_package), occurrences in sorted(edges.items()):
            if (target_package, source_package) in edges:
                continue  # Observed reverse direction makes the pattern ambiguous.
            source_modules = tuple(sorted(occurrences))
            target_modules = tuple(
                sorted({target for targets, _ in occurrences.values() for target in targets})
            )
            if len(source_modules) < 3 or len(target_modules) < 2:
                continue
            refs = tuple(sorted({key for _, keys in occurrences.values() for key in keys}))
            examined_modules = tuple(
                sorted(
                    module for module in parsed_modules
                    if module.rpartition(".")[0] == source_package
                )
            )
            rule = ProjectRule(
                rule_id=structural_rule_id(
                    facts.repository_id, source_package, target_package
                ),
                repository_id=facts.repository_id,
                type=RuleType.ARCHITECTURE,
                statement=structural_rule_statement(source_package, target_package),
                scope=ProjectScope(ProjectScopeKind.MODULE, source_modules),
                status=ProjectKnowledgeStatus.CANDIDATE,
                evidence_ids=refs,
                support=Support(len(source_modules), len(examined_modules)),
            )
            candidates.append(
                StructuralRuleCandidate(
                    rule, source_package, target_package, source_modules, target_modules,
                    examined_modules,
                )
            )
        candidates.extend(structural_delegation_candidates(facts))
        candidates.extend(testing_candidates(facts))
        used = {key for candidate in candidates for key in candidate.rule.evidence_ids}
        return StructuralRuleMiningResult(
            facts.repository_id,
            tuple(sorted(candidates, key=lambda item: item.candidate_id)),
            tuple(sources[key] for key in sorted(used)),
        )
