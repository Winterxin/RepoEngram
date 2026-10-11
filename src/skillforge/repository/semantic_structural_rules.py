"""One-candidate semantic evaluation with a deterministic evidence gate."""

from __future__ import annotations

import re
from typing import Protocol

from skillforge.domain.evidence import EvidenceType
from skillforge.domain.semantic_structural_rule import (
    SemanticStructuralAcceptance,
    SemanticStructuralDecisionKind,
    SemanticStructuralFailure,
    SemanticStructuralRejection,
    SemanticStructuralRequest,
    SemanticStructuralResponse,
    SemanticStructuralRuleResult,
    StructuralEvidenceRef,
)
from skillforge.domain.structural_rules import (
    TESTING_METHOD,
    StructuralRuleCandidate,
    StructuralRuleMiningResult,
)
from skillforge.repository.structural_evidence_binding import (
    bind_structural_evidence,
    evidence_groups,
)

_POLICY = re.compile(
    r"\b(must|required|requires?|mandates?|should|ought|shall|always|never|"
    r"forbidden|prohibited)\b", re.I,
)
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_PATH = re.compile(r"(?<!\w)(?:[\w.-]+/)+[\w.-]+")


class StructuralSemanticModel(Protocol):
    def complete(self, request: SemanticStructuralRequest) -> SemanticStructuralResponse: ...


def build_semantic_structural_request(
    mined: StructuralRuleMiningResult, candidate: StructuralRuleCandidate,
) -> SemanticStructuralRequest:
    if candidate not in mined.candidates:
        raise ValueError("candidate is not present unchanged in mining result")
    evidence = {item.evidence_id: item for item in mined.evidence}
    refs = []
    for evidence_id in candidate.rule.evidence_ids:
        item = evidence[evidence_id]
        if item.kind != EvidenceType.SOURCE or item.provenance != "python_ast":
            raise ValueError("structural Evidence must be Python source observations")
        refs.append(StructuralEvidenceRef(
            evidence_id, item.location.path, item.location.line, item.description,
            item.description.removeprefix("Python import "),
        ))
    return SemanticStructuralRequest(
        mined.repository_id, candidate.candidate_id, candidate.rule.statement,
        candidate.source_package, candidate.target_package, candidate.rule.support,
        candidate.examined_modules, candidate.source_modules, candidate.target_modules,
        tuple(refs),
        "static_testing" if candidate.method == TESTING_METHOD else
        "static_delegation" if candidate.operation_symbols else "imports",
        candidate.operation_symbols, candidate.delegation_topology, candidate.operation_family,
        candidate.test_linkages,
        evidence_groups(candidate, tuple(refs)),
        examined_operation_symbols=candidate.examined_operation_symbols,
    )


def _ids(values: tuple[str, ...], allowed: set[str], label: str) -> tuple[str, ...]:
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values)) or not set(values) <= allowed):
        raise ValueError(f"{label} has unknown, duplicate, or missing Evidence IDs")
    return tuple(sorted(values))


def _grounded_namespace(token: str, identifiers: set[str]) -> bool:
    """Ground complete segments at one namespace, shared by its descendants."""
    shorthand = tuple(token.split("."))
    namespaces = set()
    for identifier in identifiers:
        parts = tuple(identifier.split("."))
        for start in range(len(parts) - len(shorthand) + 1):
            end = start + len(shorthand)
            if parts[start:end] == shorthand:
                namespaces.add(parts[:end])
    return len(namespaces) == 1


def _text(value: str, request: SemanticStructuralRequest, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    if _POLICY.search(value):
        raise ValueError(f"{label} attempts to introduce policy")
    if "\\" in value:
        raise ValueError(f"{label} references a noncanonical path")
    namespaces = set(request.examined_modules) | set(request.target_modules)
    namespaces |= {request.source_package, request.target_package}
    known_modules = namespaces | set(request.operation_symbols) | set(request.delegation_topology)
    known_modules |= {r.test_symbol for r in request.test_linkages}
    known_paths = {item.path for item in request.evidence}
    path_free = value
    for path in sorted(known_paths, key=len, reverse=True):
        path_free = re.sub(
            rf"(?<![\w/.-]){re.escape(path)}(?![\w/.-])", " ", path_free
        )
    for token in _DOTTED.findall(path_free):
        if token not in known_modules and not _grounded_namespace(token, namespaces):
            raise ValueError(f"{label} references unknown module: {token}")
    for token in _PATH.findall(path_free):
        raise ValueError(f"{label} references unknown path: {token}")


def validate_semantic_structural_response(
    request: SemanticStructuralRequest, response: SemanticStructuralResponse,
    candidate: StructuralRuleCandidate,
) -> SemanticStructuralAcceptance | SemanticStructuralRejection:
    if not isinstance(response, SemanticStructuralResponse):
        raise ValueError("semantic response has invalid structure")
    if response.repository_id != request.repository_id:
        raise ValueError("semantic response repository_id mismatch")
    if response.candidate_id != request.candidate_id:
        raise ValueError("semantic response candidate_id mismatch")
    if candidate.candidate_id != request.candidate_id:
        raise ValueError("semantic request candidate identity mismatch")
    if not isinstance(response.decision, SemanticStructuralDecisionKind):
        raise ValueError("semantic decision is invalid")
    if (response.source_package != request.source_package
            or response.target_package != request.target_package):
        raise ValueError("semantic response package mismatch")
    response = bind_structural_evidence(request, response, candidate)
    allowed = {item.evidence_id for item in request.evidence}
    decision_ids = _ids(response.decision_evidence_ids, allowed, "decision")
    _text(response.reason, request, "decision reason")
    if response.decision == SemanticStructuralDecisionKind.ARCHITECTURE_RULE:
        if request.observation_kind == "static_testing":
            cited = set(response.decision_evidence_ids)
            sources = {item.evidence_id: item for item in request.evidence}
            for link in request.test_linkages:
                linked = [sources[key] for key in link.evidence_ids if key in cited]
                required = (
                    f"Python implementation {link.operation_symbol}:",
                    f"Python implementation {link.test_symbol}:",
                    f"Python public test call {link.test_symbol} -> {link.operation_symbol}:",
                    f"Python behavior assertion {link.test_symbol}:",
                )
                if not all(any(e.description.startswith(prefix) for e in linked)
                           for prefix in required):
                    raise ValueError("testing acceptance lacks cited behavior linkage")
        if not isinstance(response.scope_modules, tuple) or not response.scope_modules:
            raise ValueError("accepted scope requires modules")
        if response.scope_modules != request.source_modules:
            raise ValueError("accepted scope must match observed source modules")
        scope_ids = _ids(response.scope_evidence_ids, allowed, "scope")
        if not set(response.scope_evidence_ids) <= set(response.decision_evidence_ids):
            raise ValueError("scope Evidence must be decision Evidence")
        cited_paths = {
            item.path for item in request.evidence
            if item.evidence_id in response.scope_evidence_ids
        }
        for module in response.scope_modules:
            suffix = module.replace(".", "/") + ".py"
            if not any(path == suffix or path.endswith("/" + suffix)
                       for path in cited_paths):
                raise ValueError("scope module lacks cited import Evidence")
        # Phase 1 contains positive import observations only. It cannot prove exceptions.
        if response.exceptions:
            raise ValueError("exceptions lack structural Evidence")
        return SemanticStructuralAcceptance(
            candidate, request.candidate_id, response.decision, response.reason,
            decision_ids, response.scope_modules, scope_ids,
        )
    if response.scope_modules or response.scope_evidence_ids:
        raise ValueError("rejection must not contain an accepted draft")
    if response.exceptions:
        raise ValueError("rejection must not contain exceptions")
    return SemanticStructuralRejection(
        request.candidate_id, response.decision, response.reason,
        decision_ids,
    )


class SemanticStructuralRuleEvaluator:
    """Keep model failures visible and accept only gate-validated candidates."""

    def evaluate(
        self, mined: StructuralRuleMiningResult, model: StructuralSemanticModel,
    ) -> SemanticStructuralRuleResult:
        accepted = []
        rejections = []
        failures = []
        for candidate in mined.candidates:
            # Request/Evidence corruption is an input failure, not a model failure.
            request = build_semantic_structural_request(mined, candidate)
            try:
                response = model.complete(request)
            except (ValueError, TypeError, RuntimeError) as error:
                failures.append(SemanticStructuralFailure(candidate.candidate_id, str(error)))
                continue
            try:
                decision = validate_semantic_structural_response(request, response, candidate)
            except (ValueError, TypeError) as error:
                failures.append(SemanticStructuralFailure(candidate.candidate_id, str(error)))
                continue
            if isinstance(decision, SemanticStructuralAcceptance):
                accepted.append(decision)
            else:
                rejections.append(decision)
        used = {key for item in accepted for key in item.candidate.rule.evidence_ids}
        used.update(key for item in accepted for key in item.decision_evidence_ids)
        used.update(key for item in accepted for key in item.scope_evidence_ids)
        used.update(key for item in rejections for key in item.evidence_ids)
        return SemanticStructuralRuleResult(
            mined.repository_id, tuple(accepted), tuple(rejections), tuple(failures),
            tuple(item for item in mined.evidence if item.evidence_id in used),
        )
