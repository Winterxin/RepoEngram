"""Repeated public, value-producing cross-module topology as neutral candidates."""

import hashlib
from collections import defaultdict

from skillforge.domain.change import ChangePatternKind
from skillforge.domain.evidence import Support
from skillforge.domain.semantic_workflow import SemanticWorkflowRequest
from skillforge.domain.workflow import WorkflowCandidate, WorkflowCandidateMember
from skillforge.repository.semantic_workflows import _evidence_reference


def delegation_candidates(facts):
    symbols = {s.qualified_name: s for m in facts.modules for s in m.symbols}
    evidence = {e.evidence_id: e for e in facts.evidence}
    outgoing = defaultdict(list)
    for edge in facts.calls:
        source, target = symbols[edge.source_symbol], symbols[edge.target_symbol]
        if edge.returns_value and source.module != target.module:
            outgoing[edge.source_symbol].append(edge)
    families = defaultdict(list)
    for name, symbol in sorted(symbols.items()):
        if (symbol.operation_family and not name.rsplit(".", 1)[-1].startswith("_")
                and symbol.kind.value in {"method", "async_method"}):
            families[(name.rpartition(".")[0], symbol.operation_family)].append(name)
    candidates = []
    for (owner, family), entries in sorted(families.items()):
        if len(entries) < 2:
            continue
        chains, refs, topologies = [], set(), set()
        for entry in entries:
            chain, seen = [entry], {entry}
            while outgoing.get(chain[-1]):
                edges = outgoing[chain[-1]]
                if len(edges) != 1:
                    break
                edge = edges[0]
                if (edge.target_symbol in seen
                        or edge.target_symbol.rsplit(".", 1)[-1].startswith("_")):
                    break
                refs.update(edge.evidence_ids)
                chain.append(edge.target_symbol)
                seen.add(edge.target_symbol)
            # Reject incomplete, branching, cyclic and inconsistent analogous operations.
            if len(chain) < 3 or outgoing.get(chain[-1]):
                chains = []
                break
            topology = tuple(symbols[n].qualified_name.rpartition(".")[0] for n in chain)
            if len(set(topology)) != len(topology):
                chains = []
                break
            topologies.add(topology)
            chains.append(tuple(chain))
            for name in chain:
                refs.update(symbols[name].evidence_ids)
        if not chains or len(topologies) != 1:
            continue
        modules = tuple(sorted({symbols[n].module for chain in chains for n in chain}))
        paths = tuple(sorted({symbols[n].path for chain in chains for n in chain}))
        identity = "\0".join((facts.repository_id, owner, *family, *next(iter(topologies))))
        candidate_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        if any(key not in evidence or evidence[key].repository_id != facts.repository_id
               for key in refs):
            raise ValueError("delegation Evidence closure mismatch")
        observation = (f"Public operations with first parameter {family[0]} typed {family[1]} "
                       f"and return type {family[2]} currently follow repeated delegation: "
                       + "; ".join(" -> ".join(c) for c in chains))
        candidates.append(WorkflowCandidate(
            candidate_id, facts.repository_id, ChangePatternKind.STATIC_DELEGATION,
            tuple(WorkflowCandidateMember(p, next(s.module for s in symbols.values()
                                                  if s.path == p)) for p in paths),
            "", (), (), Support(len(chains), len(entries)), len(chains) / len(entries),
            (), modules, (), None, (), tuple(sorted(refs)), observation=observation,
        ))
    return tuple(sorted(candidates, key=lambda c: (-c.support.supporting, c.candidate_id)))


def build_delegation_request(facts, result, candidate):
    if candidate not in result.candidates or candidate not in delegation_candidates(facts):
        raise ValueError("static delegation candidate does not match current source facts")
    evidence = {e.evidence_id: e for e in facts.evidence}
    if any(evidence[key] != next(e for e in result.evidence if e.evidence_id == key)
           for key in candidate.evidence_ids):
        raise ValueError("static delegation Evidence changed")
    return SemanticWorkflowRequest(
        facts.repository_id, candidate.candidate_id, candidate.kind, candidate.members,
        candidate.current_modules, candidate.support, candidate.support_ratio,
        (), (), None, (), (), (), (),
        tuple(_evidence_reference(evidence[key]) for key in candidate.evidence_ids),
    )


def structural_delegation_candidates(facts):
    """Reuse frozen fact extraction, carrying observations as architecture knowledge."""
    from skillforge.domain.project_dna import (
        ProjectKnowledgeStatus,
        ProjectRule,
        ProjectScope,
        ProjectScopeKind,
        RuleType,
    )
    from skillforge.domain.structural_rules import DELEGATION_METHOD, StructuralRuleCandidate

    symbols = {s.qualified_name: s for m in facts.modules for s in m.symbols}
    result = []
    for observed in delegation_candidates(facts):
        edges = [edge for edge in facts.calls
                 if edge.returns_value and set(edge.evidence_ids) <= set(observed.evidence_ids)
                 and symbols[edge.source_symbol].module != symbols[edge.target_symbol].module]
        targets = {edge.target_symbol for edge in edges}
        entries = tuple(sorted({edge.source_symbol for edge in edges} - targets))
        outgoing = {edge.source_symbol: edge.target_symbol for edge in edges}
        chain = [entries[0]]
        while chain[-1] in outgoing:
            chain.append(outgoing[chain[-1]])
        topology = tuple(n.rpartition(".")[0] for n in chain)
        source_modules = tuple(sorted({symbols[n].module for n in entries}))
        rule = ProjectRule(
            "structural-delegation-" + observed.candidate_id, facts.repository_id,
            RuleType.ARCHITECTURE, observed.observation,
            ProjectScope(ProjectScopeKind.MODULE, source_modules),
            ProjectKnowledgeStatus.CANDIDATE, observed.evidence_ids, observed.support,
        )
        result.append(StructuralRuleCandidate(
            rule, topology[0], topology[-1], source_modules,
            (symbols[chain[-1]].module,), observed.current_modules, DELEGATION_METHOD,
            entries, topology, symbols[entries[0]].operation_family,
        ))
    # Strict complete chains take precedence over equivalent observed hops.
    covered = set()
    for candidate in result:
        refs = set(candidate.rule.evidence_ids)
        for source_owner, target_owner in zip(candidate.delegation_topology,
                                              candidate.delegation_topology[1:], strict=False):
            operations = tuple(sorted({
                edge.source_symbol for edge in facts.calls
                if edge.returns_value and set(edge.evidence_ids) <= refs
                and symbols[edge.source_symbol].qualified_name.rpartition(".")[0] == source_owner
                and symbols[edge.target_symbol].qualified_name.rpartition(".")[0] == target_owner
            }))
            covered.add((source_owner, target_owner, operations))
    result.extend(candidate for candidate in repeated_owner_delegation_candidates(facts)
                  if (candidate.source_package, candidate.target_package,
                      candidate.operation_symbols) not in covered)
    return tuple(result)


def repeated_owner_delegation_candidates(facts):
    """Observe bounded owner pairs, without inferring business layers from names.

    Two independent returned-call operations establish an owner. A side-effect
    operation can only corroborate that established owner when every resolved
    cross-module call has that same public owner. Pure helper call repetition
    cannot establish a rule. Mixed returned owners never select a primary call.
    """
    from skillforge.domain.evidence import EvidenceType
    from skillforge.domain.project_dna import (
        ProjectKnowledgeStatus,
        ProjectRule,
        ProjectScope,
        ProjectScopeKind,
        RuleType,
    )
    from skillforge.domain.static_analysis import PythonParseStatus, PythonSymbolKind
    from skillforge.domain.structural_rules import (
        DELEGATION_METHOD,
        REPEATED_OWNER_DELEGATION,
        StructuralRuleCandidate,
    )

    groups = defaultdict(list)
    for module in facts.modules:
        if module.module and module.parse_status == PythonParseStatus.PARSED:
            for symbol in module.symbols:
                if symbol.repository_id == facts.repository_id and symbol.module == module.module:
                    groups[symbol.qualified_name].append(symbol)
    symbols = {name: values[0] for name, values in groups.items() if len(values) == 1}
    evidence = {item.evidence_id: item for item in facts.evidence}
    owners = {}
    for name, symbol in symbols.items():
        if name.rsplit(".", 1)[-1].startswith("_"):
            continue
        parent = name.rpartition(".")[0]
        if symbol.kind in {PythonSymbolKind.FUNCTION, PythonSymbolKind.ASYNC_FUNCTION}:
            if parent == symbol.module:
                owners[name] = parent
        elif symbol.kind in {PythonSymbolKind.METHOD, PythonSymbolKind.ASYNC_METHOD}:
            owner = symbols.get(parent)
            if (owner and owner.kind == PythonSymbolKind.CLASS
                    and not parent.rsplit(".", 1)[-1].startswith("_")):
                owners[name] = parent

    entries, outgoing = defaultdict(list), defaultdict(list)
    for name, owner in sorted(owners.items()):
        entries[owner].append(name)
    for edge in facts.calls:
        source, target = symbols.get(edge.source_symbol), symbols.get(edge.target_symbol)
        if (source and target and edge.source_symbol in owners
                and source.module != target.module):
            outgoing[edge.source_symbol].append(edge)

    def validated(edge):
        refs = [evidence.get(key) for key in edge.evidence_ids]
        prefix = f"Python call {edge.source_symbol} -> {edge.target_symbol}: "
        if (not refs or any(item is None or item.repository_id != facts.repository_id
                            for item in refs)
                or not any(item.kind == EvidenceType.SOURCE and item.provenance == "python_ast"
                           and item.location.path == edge.path and item.location.line == edge.line
                           and item.description.startswith(prefix) for item in refs)):
            raise ValueError("repeated delegation lacks PythonCallEdge Evidence")
        return edge

    candidates = []
    for source_owner, operations in sorted(entries.items()):
        returned = defaultdict(dict)
        for name in operations:
            edges = [edge for edge in outgoing[name] if edge.returns_value]
            targets = {owners.get(edge.target_symbol) for edge in edges}
            if edges and None not in targets and len(targets) == 1:
                returned[next(iter(targets))][name] = edges
        repeated = [owner for owner, supports in sorted(returned.items()) if len(supports) >= 2]
        # Several independently stable owners do not imply a single owner rule.
        if len(repeated) != 1:
            continue
        target_owner = repeated[0]
        supports = dict(returned[target_owner])
        for name in operations:
            edges = outgoing[name]
            if (edges and not any(edge.returns_value for edge in edges)
                    and {owners.get(edge.target_symbol) for edge in edges} == {target_owner}):
                supports[name] = edges
        operation_symbols = tuple(sorted(supports))
        keys = {key for edges in supports.values() for edge in edges
                for key in validated(edge).evidence_ids}
        # Preserve the examined population's actual declarations/bodies as well
        # as positive calls, so the bounded scope and denominator are auditable.
        keys.update(key for name in operations for key in symbols[name].evidence_ids)
        if any(key not in evidence or evidence[key].repository_id != facts.repository_id
               for key in keys):
            raise ValueError("repeated delegation Evidence closure mismatch")
        refs = tuple(sorted(keys))
        source_modules = tuple(sorted({symbols[name].module for name in supports}))
        target_modules = tuple(sorted({symbols[edge.target_symbol].module
                                       for edges in supports.values() for edge in edges}))
        excluded = tuple(name for name in operations if name not in supports)
        observation = (
            f"Observed {REPEATED_OWNER_DELEGATION}: public operations "
            + ", ".join(operation_symbols) + f" delegate across modules from {source_owner} "
            f"to {target_owner}. This observation is bounded to the listed operations; "
            "other operations require current Evidence. Not covered: "
            + (", ".join(excluded) or "none")
        )
        identity = "\0".join((facts.repository_id, REPEATED_OWNER_DELEGATION,
                               source_owner, target_owner, *operation_symbols))
        rule = ProjectRule(
            "structural-delegation-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            facts.repository_id, RuleType.ARCHITECTURE, observation,
            ProjectScope(ProjectScopeKind.MODULE, source_modules),
            ProjectKnowledgeStatus.CANDIDATE, refs, Support(len(supports), len(operations)),
        )
        candidates.append(StructuralRuleCandidate(
            rule, source_owner, target_owner, source_modules, target_modules,
            tuple(sorted(set(source_modules + target_modules))), DELEGATION_METHOD,
            operation_symbols, (source_owner, target_owner),
            delegation_pattern=REPEATED_OWNER_DELEGATION,
            examined_operation_symbols=tuple(operations),
        ))
    return tuple(candidates)
