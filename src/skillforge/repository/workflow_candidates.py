"""Mine workflow candidates from existing change and structural facts."""

import hashlib
from collections import defaultdict
from itertools import combinations

from skillforge.domain.change import (
    ChangePattern,
    ChangePatternKind,
    ChangePatternResult,
    GitHistoryResult,
    NoiseSignal,
    NoiseSignalKind,
)
from skillforge.domain.evidence import Evidence
from skillforge.domain.static_analysis import ImportResolution, StaticAnalysisResult
from skillforge.domain.workflow import (
    CandidatePairAssociation,
    StructuralCoherence,
    StructuralLink,
    StructuralLinkKind,
    WorkflowCandidate,
    WorkflowCandidateMember,
    WorkflowCandidateResult,
)
from skillforge.repository.workflow_code_evidence import WorkflowCodeEvidence


def _values(pattern: ChangePattern) -> tuple[str, ...]:
    return tuple(item.value for item in pattern.items)


def _closed_seeds(seeds: list[ChangePattern]) -> list[ChangePattern]:
    """Preserve the original strict-superset rule without comparing every pair."""
    groups = defaultdict(list)
    values = [frozenset(_values(pattern)) for pattern in seeds]
    for index, pattern in enumerate(seeds):
        groups[(pattern.kind, pattern.supporting_commits)].append(index)
    retained = set()
    for indices in groups.values():
        sizes = {len(values[index]) for index in indices}
        if max(sizes, default=0) > 8:
            # Custom unbounded itemsets must not generate exponentially many subsets.
            retained.update(index for index in indices if not any(
                values[index] < values[other] for other in indices
            ))
            continue
        dominated = set()
        for index in sorted(indices, key=lambda item: -len(values[item])):
            members = values[index]
            if members in dominated:
                continue
            retained.add(index)
            for size in sizes:
                if size < len(members):
                    dominated.update(frozenset(subset) for subset in
                                     combinations(sorted(members), size))
    return [pattern for index, pattern in enumerate(seeds) if index in retained]


def _pattern_families(retained, patterns):
    """Index subset families, retaining duplicates and original stable ID order."""
    index = defaultdict(list)
    by_kind = defaultdict(list)
    for position, pattern in enumerate(patterns):
        members = frozenset(_values(pattern))
        index[(pattern.kind, members)].append((position, pattern))
        by_kind[pattern.kind].append((position, pattern, members))
    families = []
    for seed in retained:
        members = frozenset(_values(seed))
        if len(members) > 8:
            matches = [(position, pattern) for position, pattern, values in
                       by_kind[seed.kind] if values <= members]
        else:
            matches = [entry for size in range(len(members) + 1)
                       for subset in combinations(sorted(members), size)
                       for entry in index.get((seed.kind, frozenset(subset)), ())]
        family = tuple(pattern for _, pattern in sorted(
            matches, key=lambda item: (item[1].pattern_id, item[0])))
        families.append((seed, family))
    return families


def _signal_key(signal: NoiseSignal) -> tuple[str, str, str, tuple[str, ...]]:
    return signal.kind.value, signal.target, signal.method, signal.evidence_ids


class WorkflowCandidateMiner:
    """Annotate repeated patterns without inferring workflow semantics."""

    def mine(
        self,
        history: GitHistoryResult,
        static_analysis: StaticAnalysisResult,
        change_patterns: ChangePatternResult,
    ) -> WorkflowCandidateResult:
        repository_id = history.repository_id
        if (static_analysis.repository_id != repository_id
                or change_patterns.repository_id != repository_id):
            raise ValueError("all inputs must describe the same repository_id")
        if any(change.repository_id != repository_id for change in history.changesets):
            raise ValueError("Git history changeset repository_id mismatch")
        if any(pattern.repository_id != repository_id for pattern in change_patterns.patterns):
            raise ValueError("change pattern repository_id mismatch")

        pair_patterns: dict[tuple[ChangePatternKind, tuple[str, ...]], ChangePattern] = {}
        seeds: list[ChangePattern] = []
        independent_commits = {
            change.commit_id for change in history.changesets
            if change.parent_ids and change.files and not change.is_merge
        }
        for pattern in change_patterns.patterns:
            if pattern.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE:
                seeds.append(pattern)
            elif len(pattern.items) == 2:
                key = (pattern.kind, _values(pattern))
                if key in pair_patterns and pair_patterns[key] != pattern:
                    raise ValueError(f"conflicting pair pattern: {key}")
                pair_patterns[key] = pattern
                # A baseline/root occurrence cannot supply repeated collaboration.
                # Preserve the source metrics; semantics still belong to the evaluator.
                if len(set(pattern.supporting_commits) & independent_commits) >= 2:
                    seeds.append(pattern)
            elif len(pattern.items) >= 3:
                seeds.append(pattern)

        retained = _closed_seeds(seeds)
        families = _pattern_families(retained, change_patterns.patterns)
        referenced_patterns = {
            pattern.pattern_id: pattern
            for _, family in families for pattern in family
        }
        commit_ids = {change.commit_id for change in history.changesets}
        for pattern in referenced_patterns.values():
            missing_commits = set(pattern.supporting_commits) - commit_ids
            if missing_commits:
                raise ValueError(
                    f"Missing supporting commits for source pattern {pattern.pattern_id}: "
                    f"{', '.join(sorted(missing_commits))}"
                )

        module_by_path: dict[str, str] = {}
        module_evidence_by_path: dict[str, set[str]] = {}
        module_evidence_by_name: dict[str, set[str]] = {}
        current_module_names: set[str] = set()
        for module in static_analysis.modules:
            if module.module is None:
                continue
            previous = module_by_path.setdefault(module.path, module.module)
            if previous != module.module:
                raise ValueError(f"Conflicting current module mapping for {module.path}")
            current_module_names.add(module.module)
            module_evidence_by_path.setdefault(module.path, set()).update(module.evidence_ids)
            module_evidence_by_name.setdefault(module.module, set()).update(module.evidence_ids)

        static_evidence = {item.evidence_id: item for item in static_analysis.evidence}
        if len(static_evidence) != len(static_analysis.evidence):
            raise ValueError("duplicate StaticAnalysis Evidence ID")
        code_evidence = WorkflowCodeEvidence(static_analysis)
        patch_ids = {item.evidence_id for item in change_patterns.evidence
                     if item.provenance == "git_patch"}
        patches = {item.evidence_id: item for item in change_patterns.evidence
                   if item.evidence_id in patch_ids}
        import_edges: dict[tuple[str, str], set[str]] = {}
        for module in static_analysis.modules:
            for relation in module.imports:
                if relation.repository_id != repository_id:
                    raise ValueError("ImportRelation repository_id mismatch")
                for evidence_id in relation.evidence_ids:
                    evidence = static_evidence.get(evidence_id)
                    if evidence is None:
                        raise ValueError(f"Missing ImportRelation Evidence: {evidence_id}")
                    if evidence.repository_id != repository_id:
                        raise ValueError(f"Evidence repository_id mismatch: {evidence_id}")
                if (relation.resolution == ImportResolution.INTERNAL
                        and relation.source_module is not None
                        and relation.target_module is not None):
                    import_edges.setdefault(
                        (relation.source_module, relation.target_module), set()
                    ).update(relation.evidence_ids)

        candidates: list[WorkflowCandidate] = []
        for pattern, family in families:
            values = _values(pattern)
            associations = []
            evidence_ids = {
                evidence_id for source in family for evidence_id in source.evidence_ids
            }
            for first, second in combinations(values, 2):
                pair = pair_patterns.get((pattern.kind, (first, second)))
                if pair is None or pair.pair_association is None:
                    raise ValueError(
                        f"Missing pair ChangePattern for {pattern.kind.value}: {first}, {second}"
                    )
                association = pair.pair_association
                associations.append(CandidatePairAssociation(
                    first, second, pair.pattern_id,
                    association.first_occurrences, association.second_occurrences,
                    association.jaccard, association.second_given_first,
                    association.first_given_second,
                ))
                evidence_ids.update(pair.evidence_ids)

            if pattern.kind == ChangePatternKind.MODULE_CO_CHANGE:
                missing_modules = set(values) - current_module_names
                if missing_modules:
                    raise ValueError(
                        "Missing current module for candidate: "
                        f"{', '.join(sorted(missing_modules))}"
                    )
                current_modules = values
                for value in values:
                    evidence_ids.update(module_evidence_by_name[value])
            else:
                current_modules = tuple(sorted({
                    module_by_path[value] for value in values if value in module_by_path
                }))
                for value in values:
                    evidence_ids.update(module_evidence_by_path.get(value, ()))

            links: list[StructuralLink] = []
            coherence = None
            if len(current_modules) >= 2:
                members = set(current_modules)
                for (source, target), link_evidence in import_edges.items():
                    if source in members and target in members and source != target:
                        links.append(StructuralLink(
                            source, target, StructuralLinkKind.INTERNAL_IMPORT,
                            tuple(sorted(link_evidence)),
                        ))
                        evidence_ids.update(link_evidence)
                links.sort(key=lambda link: (link.kind.value, link.source, link.target))
                linked_pairs = len({tuple(sorted((link.source, link.target))) for link in links})
                possible_pairs = len(current_modules) * (len(current_modules) - 1) // 2
                coherence = StructuralCoherence(
                    linked_pairs, possible_pairs, linked_pairs / possible_pairs,
                )

            signals = tuple(sorted(
                (signal for signal in change_patterns.noise_signals
                 if self._related(signal, pattern)),
                key=_signal_key,
            ))
            for signal in signals:
                evidence_ids.update(signal.evidence_ids)
            if pattern.kind in {
                ChangePatternKind.FILE_CO_CHANGE, ChangePatternKind.MODULE_CO_CHANGE,
            }:
                selected_patches = patch_ids.intersection(evidence_ids)
                evidence_ids.update(
                    code_evidence.select_for_patches(
                        patches[key] for key in sorted(selected_patches))
                    if selected_patches else code_evidence.select(current_modules)
                )
            identity = "\0".join((repository_id, pattern.kind.value, *values))
            parent_package = values[0].removeprefix("ADDED_DIRECT_CHILD_PYTHON_MODULE_UNDER(")[:-1]
            candidates.append(WorkflowCandidate(
                candidate_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                repository_id=repository_id,
                kind=pattern.kind,
                members=tuple(WorkflowCandidateMember(
                    value,
                    value if pattern.kind == ChangePatternKind.MODULE_CO_CHANGE
                    else module_by_path.get(value),
                ) for value in values),
                seed_pattern_id=pattern.pattern_id,
                source_pattern_ids=tuple(source.pattern_id for source in family),
                supporting_commits=pattern.supporting_commits,
                support=pattern.support,
                support_ratio=pattern.support_ratio,
                pair_associations=tuple(associations),
                current_modules=current_modules,
                structural_links=tuple(links),
                structural_coherence=coherence,
                noise_signals=signals,
                evidence_ids=tuple(sorted(evidence_ids)),
                structural_events=pattern.structural_events,
                observation=(
                    f"Observed tendency: repeated changes added new direct child Python modules "
                    f"under {parent_package}."
                    if pattern.kind == ChangePatternKind.ADDED_PYTHON_MODULE_ROLE else None
                ),
            ))

        from skillforge.repository.static_delegation import delegation_candidates

        candidates.extend(delegation_candidates(static_analysis))
        candidates.sort(key=lambda candidate: (
            -candidate.support.supporting, candidate.kind.value,
            -len(candidate.members), tuple(member.value for member in candidate.members),
        ))
        referenced = {evidence_id for candidate in candidates
                      for evidence_id in candidate.evidence_ids}
        evidence_by_id: dict[str, Evidence] = {}
        for evidence in (*change_patterns.evidence, *static_analysis.evidence):
            if evidence.evidence_id not in referenced:
                continue
            if evidence.repository_id != repository_id:
                raise ValueError(f"Evidence repository_id mismatch: {evidence.evidence_id}")
            previous = evidence_by_id.setdefault(evidence.evidence_id, evidence)
            if previous != evidence:
                raise ValueError(f"Conflicting Evidence ID: {evidence.evidence_id}")
        missing = referenced - evidence_by_id.keys()
        if missing:
            raise ValueError(f"Missing candidate Evidence: {', '.join(sorted(missing))}")
        return WorkflowCandidateResult(
            repository_id, tuple(candidates),
            tuple(referenced_patterns[key] for key in sorted(referenced_patterns)),
            tuple(evidence_by_id[key] for key in sorted(referenced)),
        )

    @staticmethod
    def _related(signal: NoiseSignal, pattern: ChangePattern) -> bool:
        values = set(_values(pattern))
        if signal.kind == NoiseSignalKind.LOCK_FILE:
            return signal.target in values
        if signal.kind == NoiseSignalKind.UBIQUITOUS_ITEM:
            return any(signal.target == f"{pattern.kind.value}:{value}" for value in values)
        if signal.kind == NoiseSignalKind.LARGE_CHANGESET:
            return signal.target in pattern.supporting_commits
        return False
