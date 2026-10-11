"""Deterministic structural observations proposed as architecture rule candidates."""

import hashlib
from dataclasses import dataclass

from skillforge.domain.evidence import Evidence
from skillforge.domain.project_dna import (
    ProjectKnowledgeStatus,
    ProjectRule,
    ProjectScope,
    ProjectScopeKind,
    RuleType,
)
from skillforge.domain.serialization import to_json_value
from skillforge.domain.static_analysis import PythonTestLinkage

METHOD = "python_ast_sibling_package_imports_v1"
DELEGATION_METHOD = "python_ast_static_delegation_v1"
REPEATED_OWNER_DELEGATION = "REPEATED_OWNER_DELEGATION"
TESTING_METHOD = "python_ast_public_behavior_testing_v1"


def structural_rule_id(repository_id: str, source_package: str, target_package: str) -> str:
    identity = f"{repository_id}:{source_package}->{target_package}"
    return "structural-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def structural_rule_statement(source_package: str, target_package: str) -> str:
    return (
        f"Observed structural pattern: modules in {source_package} import "
        f"modules in {target_package}."
    )


def _canonical_modules(modules: tuple[str, ...], package: str) -> bool:
    return (
        bool(modules)
        and modules == tuple(sorted(set(modules)))
        and all(module.rpartition(".")[0] == package for module in modules)
    )


@dataclass(frozen=True, slots=True)
class StructuralRuleCandidate:
    rule: ProjectRule
    source_package: str
    target_package: str
    source_modules: tuple[str, ...]
    target_modules: tuple[str, ...]
    examined_modules: tuple[str, ...]
    method: str = METHOD
    operation_symbols: tuple[str, ...] = ()
    delegation_topology: tuple[str, ...] = ()
    operation_family: tuple[str, ...] = ()
    test_linkages: tuple[PythonTestLinkage, ...] = ()
    delegation_pattern: str = ""
    examined_operation_symbols: tuple[str, ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.rule.rule_id

    def __post_init__(self) -> None:
        if self.rule.type != RuleType.ARCHITECTURE:
            raise ValueError("structural candidate must be an architecture rule")
        if self.rule.status != ProjectKnowledgeStatus.CANDIDATE:
            raise ValueError("structural candidate must remain CANDIDATE")
        if self.rule.confidence is not None or self.method not in {
            METHOD, DELEGATION_METHOD, TESTING_METHOD,
        }:
            raise ValueError("structural candidate cannot invent confidence or method")
        if self.delegation_pattern and (
            self.method != DELEGATION_METHOD
            or self.delegation_pattern != REPEATED_OWNER_DELEGATION
        ):
            raise ValueError("unknown delegation pattern")
        if self.examined_operation_symbols and not self.delegation_pattern:
            raise ValueError("examined operations require repeated owner delegation")
        if self.method == TESTING_METHOD:
            if (len(self.operation_symbols) < 2 or len(self.operation_family) != 3
                    or self.operation_symbols != tuple(sorted(set(self.operation_symbols)))
                    or {r.operation_symbol for r in self.test_linkages}
                    != set(self.operation_symbols)
                    or len({r.test_symbol for r in self.test_linkages})
                    < len(self.operation_symbols)
                    or len({r.path for r in self.test_linkages}) != 1
                    or not all(r.call_line > 0 and r.assertion_line > 0
                               and r.assertion_kind in {"result", "raises"}
                               for r in self.test_linkages)
                    or {key for r in self.test_linkages for key in r.evidence_ids}
                    != set(self.rule.evidence_ids)
                    or self.rule.support is None
                    or self.rule.support.supporting != len(self.operation_symbols)
                    or self.rule.support.examined != len(self.operation_symbols)
                    or self.rule.scope != ProjectScope(ProjectScopeKind.MODULE,
                                                       self.source_modules)
                    or not set(self.source_modules) <= set(self.examined_modules)
                    or not set(self.target_modules) <= set(self.examined_modules)):
                raise ValueError("testing candidate lacks repeated public behavior linkage")
            return
        if self.method == DELEGATION_METHOD:
            repeated_owner = self.delegation_pattern == REPEATED_OWNER_DELEGATION
            if (len(self.operation_symbols) < 2
                    or self.operation_symbols != tuple(sorted(set(self.operation_symbols)))
                    or len(self.delegation_topology) < (2 if repeated_owner else 3)
                    or (repeated_owner and len(self.delegation_topology) != 2)
                    or len(set(self.delegation_topology)) != len(self.delegation_topology)
                    or (not repeated_owner and len(self.operation_family) != 3)
                    or (repeated_owner and self.operation_family)
                    or (repeated_owner and (
                        self.examined_operation_symbols
                        != tuple(sorted(set(self.examined_operation_symbols)))
                        or not set(self.operation_symbols) <= set(self.examined_operation_symbols)
                        or self.rule.support is None
                        or self.rule.support.examined != len(self.examined_operation_symbols)))
                    or self.source_package != self.delegation_topology[0]
                    or self.target_package != self.delegation_topology[-1]
                    or self.rule.support is None
                    or self.rule.support.supporting != len(self.operation_symbols)
                    or (not repeated_owner
                        and self.rule.support.examined != len(self.operation_symbols))
                    or self.rule.scope != ProjectScope(ProjectScopeKind.MODULE,
                                                       self.source_modules)
                    or not set(self.source_modules) <= set(self.examined_modules)
                    or not set(self.target_modules) <= set(self.examined_modules)):
                raise ValueError("delegation candidate lacks complete repeated topology")
            return
        source_parts = self.source_package.split(".")
        target_parts = self.target_package.split(".")
        if (
            len(source_parts) < 2
            or source_parts[:-1] != target_parts[:-1]
            or self.source_package == self.target_package
            or not all(part.isidentifier() for part in (*source_parts, *target_parts))
        ):
            raise ValueError("structural candidate requires distinct sibling packages")
        if not _canonical_modules(self.source_modules, self.source_package):
            raise ValueError("source modules must directly belong to source package")
        if not _canonical_modules(self.target_modules, self.target_package):
            raise ValueError("target modules must directly belong to target package")
        if len(self.source_modules) < 3 or len(self.target_modules) < 2:
            raise ValueError("structural candidate lacks repeated module support")
        if not _canonical_modules(self.examined_modules, self.source_package):
            raise ValueError("examined modules must directly belong to source package")
        if not set(self.source_modules) <= set(self.examined_modules):
            raise ValueError("supporting modules must be among examined modules")
        if self.rule.rule_id != structural_rule_id(
            self.rule.repository_id, self.source_package, self.target_package
        ):
            raise ValueError("structural candidate rule ID is not canonical")
        if self.rule.statement != structural_rule_statement(
            self.source_package, self.target_package
        ):
            raise ValueError("structural candidate statement must remain observational")
        if self.rule.support is None or (
            self.rule.support.supporting != len(self.source_modules)
            or self.rule.support.examined != len(self.examined_modules)
        ):
            raise ValueError("structural candidate Support does not match modules")
        if (
            self.rule.scope.kind != ProjectScopeKind.MODULE
            or self.rule.scope.selectors != self.source_modules
        ):
            raise ValueError("structural candidate scope must name observed source modules")


@dataclass(frozen=True, slots=True)
class StructuralRuleMiningResult:
    repository_id: str
    candidates: tuple[StructuralRuleCandidate, ...]
    evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)

    def __post_init__(self) -> None:
        if not self.repository_id:
            raise ValueError("repository_id is required")
        if any(item.rule.repository_id != self.repository_id for item in self.candidates):
            raise ValueError("candidate repository_id mismatch")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("candidate IDs must be sorted and unique")
        if any(item.repository_id != self.repository_id for item in self.evidence):
            raise ValueError("Evidence repository_id mismatch")
        ids = tuple(item.evidence_id for item in self.evidence)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("candidate Evidence IDs must be sorted and unique")
        if {key for item in self.candidates for key in item.rule.evidence_ids} != set(ids):
            raise ValueError("candidate Evidence closure mismatch")
