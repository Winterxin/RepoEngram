"""Choose a bounded, source-based semantic workload before model dispatch."""
from dataclasses import replace
from fractions import Fraction

from skillforge.domain.negative_patterns import NegativePatternCandidateResult
from skillforge.domain.structural_rules import StructuralRuleMiningResult
from skillforge.domain.workflow import WorkflowCandidateResult

TOTAL_SEMANTIC_LIMIT = 11
STRUCTURAL_SEMANTIC_LIMIT = TOTAL_SEMANTIC_LIMIT


def _related(module, scope):
    return bool(module) and (module == scope or module.startswith(scope + ".")
                            or scope.startswith(module + "."))


def candidate_modules(candidate, facts):
    if hasattr(candidate, "members"):
        return tuple(member.current_module for member in candidate.members
                     if member.current_module is not None)
    if hasattr(candidate, "affected_paths"):
        paths = set(candidate.affected_paths)
        return tuple(item.module for item in facts.modules
                     if item.path in paths and item.module is not None)
    return tuple(sorted(set(candidate.source_modules + candidate.target_modules)))


def scoped_candidates(result, facts, current_module=None):
    if current_module is None:
        return result
    if not isinstance(current_module, str) or not current_module.strip():
        raise ValueError("candidate module scope must be nonempty")
    # Keep the complete candidate, including companion modules and evidence.
    selected = tuple(
        candidate for candidate in result.candidates
        if any(_related(module, current_module)
               for module in candidate_modules(candidate, facts))
    )
    return _with_candidates(result, selected)


def _with_candidates(result, selected):
    changes = {"candidates": tuple(selected)}
    if isinstance(result, StructuralRuleMiningResult):
        changes["candidates"] = tuple(sorted(selected, key=lambda item: item.candidate_id))
        used = {key for item in selected for key in item.rule.evidence_ids}
        changes["evidence"] = tuple(item for item in result.evidence if item.evidence_id in used)
    elif isinstance(result, WorkflowCandidateResult):
        patterns = {key for item in selected for key in item.source_pattern_ids}
        used = {key for item in selected for key in item.evidence_ids}
        changes["source_patterns"] = tuple(
            item for item in result.source_patterns if item.pattern_id in patterns
        )
        changes["evidence"] = tuple(item for item in result.evidence if item.evidence_id in used)
    elif isinstance(result, NegativePatternCandidateResult):
        signals = {key for item in selected for key in item.signal_ids}
        used = {key for item in selected for key in item.evidence_ids}
        changes["signals"] = tuple(item for item in result.signals if item.signal_id in signals)
        changes["evidence"] = tuple(item for item in result.evidence if item.evidence_id in used)
    return replace(result, **changes)


def select_structural_candidates(result, *, limit=STRUCTURAL_SEMANTIC_LIMIT):
    if type(limit) is not int or limit < 0:
        raise ValueError("structural candidate limit must be nonnegative")
    remaining = list(result.candidates)
    selected, covered, methods = [], set(), set()
    while remaining and len(selected) < limit:
        def rank(item):
            support = item.rule.support
            units = set(item.source_modules + item.target_modules)
            return (item.method not in methods,
                    Fraction(support.supporting, support.examined),
                    len(units - covered), support.supporting,
                    item.candidate_id)
        candidate = max(remaining, key=rank)
        selected.append(candidate)
        methods.add(candidate.method)
        covered.update(candidate.source_modules + candidate.target_modules)
        remaining.remove(candidate)
    return tuple(selected)


def bounded_structural(result, facts, current_module, workflows, anti_patterns):
    available = max(0, TOTAL_SEMANTIC_LIMIT - len(workflows) - len(anti_patterns))
    selected = select_structural_candidates(
        scoped_candidates(result, facts, current_module),
        limit=min(STRUCTURAL_SEMANTIC_LIMIT, available),
    )
    return _with_candidates(result, selected)


def selection_rows(result, selected, facts, current_module, channel):
    identifiers = {item.candidate_id for item in selected}
    eligible = {item.candidate_id for item in scoped_candidates(
        result, facts, current_module).candidates}
    return tuple({
        "candidate_id": item.candidate_id, "kind": channel, "rank": rank,
        "selected": item.candidate_id in identifiers,
        "reason": ("selected within channel budget" if item.candidate_id in identifiers
                   else "outside declared module scope" if item.candidate_id not in eligible
                   else "outside channel budget"),
    } for rank, item in enumerate(result.candidates, 1))
