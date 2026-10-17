"""Compose bounded repository knowledge into repository-local Skills."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.evolution import EvolutionHistoryState, EvolutionKnowledge, fingerprint
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.domain.verification import PrepareVerifyReceipt
from skillforge.errors import (
    InvalidRepositoryPath,
    SkillForgeError,
)
from skillforge.filesystem_safety import is_link
from skillforge.integration.agent_skills import AgentSkillExporter
from skillforge.integration.knowledge_trace import KnowledgeTrace, trace_operation
from skillforge.integration.repo_local import RepoLocalSkillSync
from skillforge.productization.evolution_artifacts import (
    load_baseline,
    make_baseline,
    save_baseline,
)
from skillforge.productization.semantic_assembly import (
    _merge_evidence as _merge_evidence,
)
from skillforge.productization.semantic_assembly import (
    _retry_invalid_decision as _retry_invalid_decision,
)
from skillforge.productization.semantic_assembly import (
    evaluate_semantic_knowledge,
)
from skillforge.repository.analyzer import RepositoryAnalyzer
from skillforge.repository.bounded_candidate_selection import (
    bounded_structural,
    scoped_candidates,
    selection_rows,
)
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.codex_cli_antipattern import CodexCliAntiPatternSemanticModel
from skillforge.repository.codex_cli_semantic import CodexCliSemanticModel
from skillforge.repository.codex_cli_structural import CodexCliStructuralSemanticModel
from skillforge.repository.codex_cli_tendency import CodexCliTendencySemanticModel
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.identity import repository_identity
from skillforge.repository.incremental_evolution import IncrementalEvolution, production_inputs
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.project_dna import ProjectDNABuilder
from skillforge.repository.repo_snapshot_semantic_cache import (
    RepoSnapshotSemanticCache,
    _production_version,
)
from skillforge.repository.semantic_antipattern_evaluation import (
    select_semantic_antipattern_candidates,
)
from skillforge.repository.semantic_antipatterns import (
    AntiPatternSemanticModel,
)
from skillforge.repository.semantic_evaluation import (
    select_semantic_candidates,
    semantic_selection_report,
)
from skillforge.repository.semantic_policy import (
    ANTIPATTERN_SEMANTIC_LIMIT,
    WORKFLOW_SEMANTIC_LIMIT,
)
from skillforge.repository.semantic_structural_rules import (
    StructuralSemanticModel,
)
from skillforge.repository.semantic_tendencies import (
    TendencySemanticModel,
)
from skillforge.repository.semantic_workflows import (
    WorkflowSemanticModel,
)
from skillforge.repository.skill_compiler import SkillCompiler
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.structural_rules import StructuralRuleMiner
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner
from skillforge.verify import VerifyPlanBuilder
from skillforge.verify.artifacts import save_prepare_receipt

_MANAGED_PACKAGE_FILES = (".skillforge.json", "SKILL.md", "references/evidence.md")

HISTORY_LIMIT = 50


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    path: Path
    task: str
    target: CodeAgentTarget
    current_module: str | None = None
    baseline: Path | None = None
    selection_module: str | None = None


@dataclass(frozen=True, slots=True)
class PrepareResult:
    repository_id: str
    repository_root: Path
    target: CodeAgentTarget
    task: str
    current_module: str | None
    history_participated: bool
    history_commit_count: int
    change_pattern_count: int
    workflow_candidate_count: int
    workflow_semantic_selected_count: int
    workflow_semantic_retry_count: int
    workflow_accepted_count: int
    workflow_rejection_count: int
    anti_pattern_candidate_count: int
    anti_pattern_semantic_selected_count: int
    anti_pattern_semantic_retry_count: int
    anti_pattern_accepted_count: int
    anti_pattern_rejection_count: int
    structural_candidate_count: int
    semantic_accepted_count: int
    semantic_rejection_count: int
    semantic_failure_count: int
    dna_rule_count: int
    dna_workflow_count: int
    dna_anti_pattern_count: int
    dna_snapshot_id: str
    compiled_skill_count: int
    selected_skill_ids: tuple[str, ...]
    verify_available: bool
    verify_plan_id: str | None
    verify_base_commit: str | None
    verify_receipt_path: str | None
    evolution_available: bool
    evolution_baseline_id: str | None
    evolution_baseline_path: str | None
    evolution_head_commit: str | None
    installed_packages: tuple[str, ...]
    written_files: tuple[str, ...]
    unchanged_files: tuple[str, ...]
    removed_packages: tuple[str, ...]
    prepare_mode: str = "rich"
    history_semantic_selection: tuple[dict, ...] = ()
    tendency_accepted_count: int = 0
    tendency_rejection_count: int = 0
    semantic_failures: tuple[dict, ...] = ()
    evolution_refresh: tuple[dict, ...] = ()
    evolution_statistics: dict | None = None

    def to_dict(self) -> dict[str, object]:
        target_dir = ".codex" if self.target == CodeAgentTarget.CODEX else ".claude"
        successful = {
            "workflow": self.workflow_accepted_count + self.workflow_rejection_count,
            "tendency": self.tendency_accepted_count + self.tendency_rejection_count,
            "anti-pattern": self.anti_pattern_accepted_count + self.anti_pattern_rejection_count,
            "structural": self.semantic_accepted_count + self.semantic_rejection_count,
        }
        return {
            "repository_id": self.repository_id,
            "repository_root": str(self.repository_root),
            "target": self.target.value,
            "task": self.task,
            "current_module": self.current_module,
            "prepare_mode": self.prepare_mode,
            "evolution_refresh": list(self.evolution_refresh),
            "evolution_statistics": self.evolution_statistics,
            "history_semantic_selection": list(self.history_semantic_selection),
            "tendency_accepted_count": self.tendency_accepted_count,
            "tendency_rejection_count": self.tendency_rejection_count,
            "history_participated": self.history_participated,
            "history_commit_count": self.history_commit_count,
            "change_pattern_count": self.change_pattern_count,
            "workflow_candidate_count": self.workflow_candidate_count,
            "workflow_semantic_selected_count": self.workflow_semantic_selected_count,
            "workflow_semantic_retry_count": self.workflow_semantic_retry_count,
            "workflow_accepted_count": self.workflow_accepted_count,
            "workflow_rejection_count": self.workflow_rejection_count,
            "anti_pattern_candidate_count": self.anti_pattern_candidate_count,
            "anti_pattern_semantic_selected_count": self.anti_pattern_semantic_selected_count,
            "anti_pattern_semantic_retry_count": self.anti_pattern_semantic_retry_count,
            "anti_pattern_accepted_count": self.anti_pattern_accepted_count,
            "anti_pattern_rejection_count": self.anti_pattern_rejection_count,
            "structural_candidate_count": self.structural_candidate_count,
            "semantic_accepted_count": self.semantic_accepted_count,
            "semantic_rejection_count": self.semantic_rejection_count,
            "semantic_failure_count": self.semantic_failure_count,
            "semantic_failures": list(self.semantic_failures),
            "semantic_channels": {
                channel: (
                    ("degraded" if successful[channel] else "failed")
                    if any(f["channel"] == channel for f in self.semantic_failures)
                    else "ok"
                )
                for channel in ("workflow", "tendency", "anti-pattern", "structural")
            },
            "project_dna": {
                "rule_count": self.dna_rule_count,
                "workflow_count": self.dna_workflow_count,
                "anti_pattern_count": self.dna_anti_pattern_count,
                "snapshot_id": self.dna_snapshot_id,
            },
            "compiled_skill_count": self.compiled_skill_count,
            "selected_skill_count": len(self.selected_skill_ids),
            "selected_skill_ids": list(self.selected_skill_ids),
            "verify_available": self.verify_available,
            "verify_plan_id": self.verify_plan_id,
            "verify_base_commit": self.verify_base_commit,
            "verify_receipt_path": self.verify_receipt_path,
            "evolution_available": self.evolution_available,
            "evolution_baseline_id": self.evolution_baseline_id,
            "evolution_baseline_path": self.evolution_baseline_path,
            "evolution_head_commit": self.evolution_head_commit,
            "sync": {
                "written_files": list(self.written_files),
                "unchanged_files": list(self.unchanged_files),
                "removed_files": [
                    f"{target_dir}/skills/{package}/{name}"
                    for package in self.removed_packages
                    for name in _MANAGED_PACKAGE_FILES
                ],
                "removed_packages": list(self.removed_packages),
            },
        }


class SkillForgePreparer:
    def __init__(
        self,
        semantic_model: StructuralSemanticModel | None = None,
        workflow_model: WorkflowSemanticModel | None = None,
        anti_pattern_model: AntiPatternSemanticModel | None = None,
        tendency_model: TendencySemanticModel | None = None,
        *, semantic_cache_root: Path | None = None,
    ) -> None:
        self._semantic_model = semantic_model
        self._workflow_model = workflow_model
        self._anti_pattern_model = anti_pattern_model
        self._tendency_model = tendency_model
        self._semantic_cache_root = semantic_cache_root

    def prepare(self, request: PrepareRequest, *, _loaded_baseline=None) -> PrepareResult:
        root = Path(request.path).expanduser().absolute()
        trace = KnowledgeTrace("pending")
        with trace_operation(root, trace, "pending", "prepare"):
            result = self._prepare(request, _loaded_baseline=_loaded_baseline, trace=trace)
            trace.repository_id = result.repository_id
            trace.finish("prepare", "COMPLETE", object_id=result.dna_snapshot_id)
            return result

    def _prepare(self, request: PrepareRequest, *, _loaded_baseline=None,
                 trace: KnowledgeTrace) -> PrepareResult:
        requested = Path(request.path).expanduser().absolute()
        if any(is_link(path) for path in (requested, *requested.parents)):
            raise InvalidRepositoryPath(f"Repository path must be a real directory: {requested}")
        root = requested.resolve()
        if not root.is_dir():
            raise InvalidRepositoryPath(f"Repository path is not a directory: {root}")
        if not isinstance(request.target, CodeAgentTarget):
            raise SkillForgeError(f"Unsupported target: {request.target}")
        if not isinstance(request.task, str) or not request.task.strip():
            raise SkillForgeError("Task must be nonempty")
        if request.current_module is not None and (
            not isinstance(request.current_module, str) or not request.current_module.strip()
        ):
            raise SkillForgeError("Current module must be nonempty")
        if request.selection_module is not None and (
            not isinstance(request.selection_module, str) or not request.selection_module.strip()
        ):
            raise SkillForgeError("Selection module must be nonempty")
        repository_id = repository_identity(root)
        trace.repository_id = repository_id
        if request.baseline is not None:
            trace.record("prepare", "BASELINE_REUSE")
            return self._prepare_from_baseline(
                root, repository_id, request, _loaded_baseline=_loaded_baseline,
            )
        if _loaded_baseline is not None:
            raise SkillForgeError("Loaded baseline requires a baseline request")
        semantic_cache = RepoSnapshotSemanticCache(root, self._semantic_cache_root)
        trace.record("static_analysis", "STARTED")
        trace.persist(root)
        profile = RepositoryAnalyzer().analyze(root)
        if profile.repository.repository_id != repository_id:
            raise SkillForgeError("Repository identity mismatch")
        facts = PythonStaticAnalyzer().analyze(root)
        trace.record("static_analysis", "COMPLETE", count=len(facts.modules))
        if facts.repository_id != repository_id:
            raise SkillForgeError("Repository identity mismatch")
        history = None
        patterns = None
        workflow_candidates = None
        selected_workflows = ()
        workflows = None
        workflow_retry_count = 0
        anti_pattern_candidates = None
        selected_anti_patterns = ()
        anti_patterns = None
        anti_pattern_retry_count = 0
        tendencies = None
        selection_report = ()
        if profile.git.is_git_repository:
            trace.record("history", "STARTED")
            trace.persist(root)
            history = GitHistoryMiner().mine(root, limit=HISTORY_LIMIT)
            trace.record("history", "COMPLETE", count=len(history.changesets))
            if history.repository_id != repository_id:
                raise SkillForgeError("Repository identity mismatch")
            patterns = ChangePatternMiner().mine(history, facts)
            workflow_candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
            anti_pattern_candidates = NegativePatternMiner().mine(history)
            selected_workflows = select_semantic_candidates(
                scoped_candidates(workflow_candidates, facts, request.selection_module),
                limit=WORKFLOW_SEMANTIC_LIMIT,
            )
            selection_report = semantic_selection_report(
                workflow_candidates, limit=WORKFLOW_SEMANTIC_LIMIT,
            )
            selected_anti_patterns = select_semantic_antipattern_candidates(
                scoped_candidates(anti_pattern_candidates, facts, request.selection_module),
                limit=ANTIPATTERN_SEMANTIC_LIMIT,
            )
            from skillforge.repository.history_patch_context import enrich_selected_history

            enriched = enrich_selected_history(
                root, history, facts, workflow_candidates, anti_pattern_candidates,
                workflow_limit=WORKFLOW_SEMANTIC_LIMIT,
                anti_pattern_limit=ANTIPATTERN_SEMANTIC_LIMIT,
                current_module=request.selection_module,
            )
            if enriched.history != history:
                history = enriched.history
                patterns = ChangePatternMiner().mine(history, facts)
                workflow_candidates = WorkflowCandidateMiner().mine(history, facts, patterns)
                anti_pattern_candidates = NegativePatternMiner().mine(history)
                selected_workflows = select_semantic_candidates(
                    scoped_candidates(workflow_candidates, facts, request.selection_module),
                    limit=WORKFLOW_SEMANTIC_LIMIT,
                )
                selected_anti_patterns = select_semantic_antipattern_candidates(
                    scoped_candidates(anti_pattern_candidates, facts, request.selection_module),
                    limit=ANTIPATTERN_SEMANTIC_LIMIT,
                )
                selection_report = semantic_selection_report(
                    workflow_candidates, limit=WORKFLOW_SEMANTIC_LIMIT,
                )
        all_structural = StructuralRuleMiner().mine(facts)
        mined = bounded_structural(all_structural, facts, request.selection_module,
                                   selected_workflows, selected_anti_patterns)
        selection_report = (
            selection_rows(workflow_candidates, selected_workflows, facts,
                           request.selection_module, "workflow") if workflow_candidates else ()
        ) + (
            selection_rows(anti_pattern_candidates, selected_anti_patterns, facts,
                           request.selection_module, "anti-pattern")
            if anti_pattern_candidates else ()
        ) + selection_rows(all_structural, mined.candidates, facts,
                           request.selection_module, "structural")
        for item in selection_report:
            trace.record("candidate_selection", item["reason"], object_id=item["candidate_id"])
        trace.record("semantic_knowledge", "STARTED")
        trace.persist(root)
        assembly = evaluate_semantic_knowledge(
            repository_id, facts, mined,
            semantic_cache.wrap(self._semantic_model or CodexCliStructuralSemanticModel()),
            history=history, candidates=workflow_candidates,
            selected_workflows=selected_workflows, negative=anti_pattern_candidates,
            selected_anti_patterns=selected_anti_patterns,
            workflow_model=semantic_cache.wrap(self._workflow_model or CodexCliSemanticModel()),
            tendency_model=semantic_cache.wrap(
                self._tendency_model or CodexCliTendencySemanticModel(),
            ),
            anti_pattern_model=semantic_cache.wrap(
                self._anti_pattern_model or CodexCliAntiPatternSemanticModel(),
            ),
        )
        workflows, tendencies, anti_patterns, semantic = (
            assembly.workflows, assembly.tendencies, assembly.anti_patterns, assembly.structural,
        )
        workflow_retry_count = assembly.workflow_retry_count
        anti_pattern_retry_count = assembly.anti_pattern_retry_count
        trace.record("semantic_knowledge", "COMPLETE_WITH_FAILURES" if assembly.failures
                     else "COMPLETE", count=len(assembly.failures))
        semantic_cache.publish(workflows, tendencies, anti_patterns, semantic)
        dna = ProjectDNABuilder().build(
            repository_id, workflows=workflows, anti_patterns=anti_patterns,
            semantic_structural_rules=semantic,
            semantic_tendencies=tendencies,
        )
        compiled = SkillCompiler().compile(dna, semantic, tendencies)
        trace.record("compilation", "COMPLETE", count=len(compiled.skills),
                     object_id=dna.snapshot_id)
        registry = SkillRegistryBuilder().build(compiled)
        route = SkillRouter().route(
            registry,
            SkillRoutingRequest(repository_id, request.task, current_module=request.current_module),
        )
        plan = AgentSkillExporter().plan(route, request.target, repository_root=root, trace=trace)
        try:
            sync = RepoLocalSkillSync().sync(root, plan)
            trace.record("publication", "SYNC_COMPLETE", count=len(sync.written_files))
        except ValueError as error:
            raise SkillForgeError(f"Repository-local Skill sync failed: {error}") from error
        verify_receipt = None
        verify_receipt_path = None
        evolution_baseline = None
        evolution_baseline_path = None
        if profile.git.is_git_repository and profile.git.head_commit is not None:
            source_state = EvolutionHistoryState(
                repository_id, profile.git.head_commit, HISTORY_LIMIT,
                history, workflow_candidates, anti_pattern_candidates,
                current_module=request.selection_module,
            )
            try:
                evolution_baseline = make_baseline(
                    root,
                    profile.git.head_commit,
                    HISTORY_LIMIT,
                    dna,
                    compiled,
                    source_state,
                    semantic,
                    EvolutionKnowledge(
                        production_inputs(source_state, facts),
                        fingerprint(facts),
                        _production_version(),
                        workflows,
                        tendencies,
                        anti_patterns,
                        semantic,
                        facts,
                    ),
                )
                evolution_baseline_path = save_baseline(root, evolution_baseline)
            except (OSError, ValueError, TypeError) as error:
                raise SkillForgeError(f"Cannot save Evolution baseline: {error}") from error
            seed_plan = VerifyPlanBuilder().build(route)
            try:
                target_dir = (".codex" if request.target == CodeAgentTarget.CODEX
                              else ".claude")
                managed_paths = set(sync.written_files) | set(sync.unchanged_files)
                managed_paths.update(
                    f"{target_dir}/skills/{package}/{name}"
                    for package in sync.removed_packages for name in _MANAGED_PACKAGE_FILES
                )
                verify_receipt = PrepareVerifyReceipt.create(
                    seed_plan, profile.git.head_commit, tuple(sorted(managed_paths)),
                )
                verify_receipt_path = save_prepare_receipt(root, verify_receipt)
            except (OSError, ValueError) as error:
                raise SkillForgeError(f"Cannot save prepare Verify receipt: {error}") from error
        return PrepareResult(
            repository_id=repository_id, repository_root=root, target=request.target,
            task=request.task, current_module=request.current_module,
            history_participated=history is not None,
            history_commit_count=len(history.changesets) if history is not None else 0,
            change_pattern_count=len(patterns.patterns) if patterns is not None else 0,
            workflow_candidate_count=(len(workflow_candidates.candidates)
                                      if workflow_candidates is not None else 0),
            workflow_semantic_selected_count=len(selected_workflows),
            workflow_semantic_retry_count=workflow_retry_count,
            workflow_accepted_count=len(workflows.workflows) if workflows is not None else 0,
            workflow_rejection_count=len(workflows.rejections) if workflows is not None else 0,
            anti_pattern_candidate_count=(len(anti_pattern_candidates.candidates)
                                          if anti_pattern_candidates is not None else 0),
            anti_pattern_semantic_selected_count=len(selected_anti_patterns),
            anti_pattern_semantic_retry_count=anti_pattern_retry_count,
            anti_pattern_accepted_count=(len(anti_patterns.anti_patterns)
                                         if anti_patterns is not None else 0),
            anti_pattern_rejection_count=(len(anti_patterns.rejections)
                                          if anti_patterns is not None else 0),
            structural_candidate_count=len(mined.candidates),
            semantic_accepted_count=len(semantic.accepted),
            semantic_rejection_count=len(semantic.rejections),
            semantic_failure_count=len(assembly.failures),
            semantic_failures=tuple(assembly.failures),
            dna_rule_count=len(dna.rules), dna_workflow_count=len(dna.workflows),
            dna_anti_pattern_count=len(dna.anti_patterns), dna_snapshot_id=dna.snapshot_id,
            compiled_skill_count=len(compiled.skills),
            selected_skill_ids=tuple(match.skill.skill_id for match in route.matches),
            verify_available=verify_receipt is not None,
            verify_plan_id=verify_receipt.plan.plan_id if verify_receipt else None,
            verify_base_commit=verify_receipt.base_commit if verify_receipt else None,
            verify_receipt_path=verify_receipt_path,
            evolution_available=evolution_baseline is not None,
            evolution_baseline_id=(evolution_baseline.baseline_id
                                   if evolution_baseline else None),
            evolution_baseline_path=evolution_baseline_path,
            evolution_head_commit=(evolution_baseline.head_commit
                                   if evolution_baseline else None),
            installed_packages=tuple(
                f".{('codex' if request.target == CodeAgentTarget.CODEX else 'claude')}"
                f"/skills/{item.agent_skill_name}/" for item in plan.packages
            ),
            written_files=sync.written_files, unchanged_files=sync.unchanged_files,
            removed_packages=sync.removed_packages,
            history_semantic_selection=selection_report,
            tendency_accepted_count=len(tendencies.accepted) if tendencies else 0,
            tendency_rejection_count=len(tendencies.rejections) if tendencies else 0,
        )

    def _prepare_from_baseline(
        self, root: Path, repository_id: str, request: PrepareRequest, *, _loaded_baseline=None
    ) -> PrepareResult:
        try:
            baseline = (_loaded_baseline if _loaded_baseline is not None
                        else load_baseline(root, request.baseline))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            raise SkillForgeError(f"Invalid Evolution baseline: {error}") from error
        if baseline.repository_id != repository_id:
            raise SkillForgeError("Evolution baseline repository identity mismatch")
        head = subprocess.run(
            ("git", "-c", f"safe.directory={root}", "rev-parse", "--verify", "HEAD^{commit}"),
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        if head.returncode:
            raise SkillForgeError("Repository has no committed Git HEAD")
        if head.stdout.strip() != baseline.head_commit and baseline.production_knowledge:
            from skillforge.productization.evolve_apply import evolve_apply
            from skillforge.productization.evolve_plan import evolve_plan

            try:
                planned = evolve_plan(root, Path(request.baseline)).summary
                applied = evolve_apply(
                    root,
                    Path(request.baseline),
                    Path(planned["plan_artifact"]),
                    workflow_model=self._workflow_model,
                    antipattern_model=self._anti_pattern_model,
                    tendency_model=self._tendency_model,
                    structural_model=self._semantic_model,
                    semantic_cache_root=self._semantic_cache_root,
                ).summary
            except (OSError, ValueError) as error:
                raise SkillForgeError(f"Incremental prepare failed: {error}") from error
            prepared = self._prepare_from_baseline(
                root,
                repository_id,
                replace(request, baseline=Path(applied["next_baseline"])),
            )
            return replace(
                prepared,
                prepare_mode="incremental",
                evolution_refresh=tuple(applied["refresh_decisions"]),
                evolution_statistics={
                    key: applied[key]
                    for key in (
                        "full_candidate_count",
                        "incremental_refresh_count",
                        "reused_knowledge_count",
                        "semantic_calls_incremental",
                    )
                },
                semantic_failures=tuple(applied["semantic_failures"]),
                semantic_failure_count=len(applied["semantic_failures"]),
                workflow_accepted_count=applied["channel_decisions"]["workflow"]["accepted"],
                workflow_rejection_count=applied["channel_decisions"]["workflow"]["rejected"],
                tendency_accepted_count=applied["channel_decisions"]["tendency"]["accepted"],
                tendency_rejection_count=applied["channel_decisions"]["tendency"]["rejected"],
                anti_pattern_accepted_count=applied["channel_decisions"]["anti-pattern"][
                    "accepted"
                ],
                anti_pattern_rejection_count=applied["channel_decisions"]["anti-pattern"][
                    "rejected"
                ],
                semantic_accepted_count=applied["channel_decisions"]["structural"]["accepted"],
                semantic_rejection_count=applied["channel_decisions"]["structural"]["rejected"],
            )
        if head.stdout.strip() != baseline.head_commit:
            raise SkillForgeError(
                "Repository HEAD differs from Evolution baseline; "
                "run evolve-plan/evolve-apply or use a matching baseline."
            )
        current_static = PythonStaticAnalyzer().analyze(root)
        if baseline.source_state.current_module != request.selection_module:
            raise SkillForgeError("Candidate selection scope differs from Evolution baseline")
        reconstructed = IncrementalEvolution().make_state(
            baseline.source_state.history, baseline.head_commit,
            baseline.history_limit, current_static,
            current_module=baseline.source_state.current_module,
        )
        if reconstructed != baseline.source_state:
            raise SkillForgeError(
                "Repository static facts differ from Evolution baseline; "
                "update evolution before preparing the next task."
            )
        if baseline.production_knowledge is not None:
            if fingerprint(current_static) != baseline.production_knowledge.static_hash:
                raise SkillForgeError("Repository static facts differ from Evolution baseline")
            if _production_version() != baseline.production_knowledge.production_version:
                raise SkillForgeError("Production version differs; explicitly run full prepare")
        compiled = baseline.source_compilation
        if compiled.source_snapshot_id != baseline.source_dna.snapshot_id:
            raise SkillForgeError("Evolution baseline compilation snapshot mismatch")
        registry = SkillRegistryBuilder().build(compiled)
        route = SkillRouter().route(
            registry,
            SkillRoutingRequest(repository_id, request.task, current_module=request.current_module),
        )
        plan = AgentSkillExporter().plan(route, request.target, repository_root=root)
        try:
            sync = RepoLocalSkillSync().sync(root, plan)
        except ValueError as error:
            raise SkillForgeError(f"Repository-local Skill sync failed: {error}") from error
        target_dir = ".codex" if request.target == CodeAgentTarget.CODEX else ".claude"
        managed_paths = set(sync.written_files) | set(sync.unchanged_files)
        managed_paths.update(
            f"{target_dir}/skills/{package}/{name}"
            for package in sync.removed_packages for name in _MANAGED_PACKAGE_FILES
        )
        try:
            receipt = PrepareVerifyReceipt.create(
                VerifyPlanBuilder().build(route), baseline.head_commit,
                tuple(sorted(managed_paths)),
            )
            receipt_path = save_prepare_receipt(root, receipt)
        except (OSError, ValueError) as error:
            raise SkillForgeError(f"Cannot save prepare Verify receipt: {error}") from error
        baseline_path = Path(request.baseline)
        if not baseline_path.is_absolute():
            baseline_path = root / baseline_path
        return PrepareResult(
            repository_id=repository_id, repository_root=root, target=request.target,
            task=request.task, current_module=request.current_module,
            history_participated=True,
            history_commit_count=len(baseline.source_state.history.changesets),
            change_pattern_count=0, workflow_candidate_count=0,
            workflow_semantic_selected_count=0, workflow_semantic_retry_count=0,
            workflow_accepted_count=0, workflow_rejection_count=0,
            anti_pattern_candidate_count=0, anti_pattern_semantic_selected_count=0,
            anti_pattern_semantic_retry_count=0, anti_pattern_accepted_count=0,
            anti_pattern_rejection_count=0, structural_candidate_count=0,
            semantic_accepted_count=0, semantic_rejection_count=0,
            semantic_failure_count=0,
            dna_rule_count=len(baseline.source_dna.rules),
            dna_workflow_count=len(baseline.source_dna.workflows),
            dna_anti_pattern_count=len(baseline.source_dna.anti_patterns),
            dna_snapshot_id=baseline.source_dna.snapshot_id,
            compiled_skill_count=len(compiled.skills),
            selected_skill_ids=tuple(match.skill.skill_id for match in route.matches),
            verify_available=True, verify_plan_id=receipt.plan.plan_id,
            verify_base_commit=receipt.base_commit, verify_receipt_path=receipt_path,
            evolution_available=True, evolution_baseline_id=baseline.baseline_id,
            evolution_baseline_path=baseline_path.resolve().relative_to(root).as_posix(),
            evolution_head_commit=baseline.head_commit,
            installed_packages=tuple(
                f"{target_dir}/skills/{item.agent_skill_name}/" for item in plan.packages
            ),
            written_files=sync.written_files, unchanged_files=sync.unchanged_files,
            removed_packages=sync.removed_packages, prepare_mode="evolution_baseline",
        )
