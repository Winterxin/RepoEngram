"""Minimal local CLI for the implemented repository analyzer."""

import argparse
import json
import subprocess
import sys
from math import isfinite
from pathlib import Path

from skillforge.cost_control import BudgetExceeded
from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.change import ChangePatternKind, ChangePatternResult, GitHistoryResult
from skillforge.domain.feedback import FeedbackKind, HumanFeedbackLedger
from skillforge.domain.negative_patterns import NegativePatternCandidateResult
from skillforge.domain.static_analysis import StaticAnalysisIssueType, StaticAnalysisResult
from skillforge.domain.verification import (
    VerifyCheckStatus,
    VerifyObligationKind,
    VerifyOverallStatus,
    VerifyTargetRef,
)
from skillforge.domain.workflow import WorkflowCandidateResult
from skillforge.errors import SkillForgeError
from skillforge.feedback.artifacts import (
    LEDGER_PATH,
    load_feedback_ledger,
    save_feedback_ledger,
    source_path,
)
from skillforge.feedback.service import HumanFeedbackService
from skillforge.filesystem_safety import is_link
from skillforge.logging_config import configure_logging
from skillforge.productization.evolve_apply import evolve_apply
from skillforge.productization.evolve_plan import evolve_plan
from skillforge.productization.prepare import PrepareRequest, PrepareResult, SkillForgePreparer
from skillforge.repository.analyzer import RepositoryAnalyzer
from skillforge.repository.change_patterns import ChangePatternMiner
from skillforge.repository.git_history import GitHistoryMiner
from skillforge.repository.negative_patterns import NegativePatternMiner
from skillforge.repository.semantic_antipattern_evaluation import (
    evaluate_semantic_antipatterns,
    write_semantic_antipattern_evaluation_report,
)
from skillforge.repository.semantic_evaluation import (
    evaluate_semantic_workflows,
    write_semantic_evaluation_report,
)
from skillforge.repository.static_analysis import PythonStaticAnalyzer
from skillforge.repository.workflow_candidates import WorkflowCandidateMiner
from skillforge.verify import IndependentVerifier, VerifyPlanBuilder
from skillforge.verify.artifacts import (
    load_checks,
    load_prepare_receipt,
    load_verify_artifact,
    save_verify_report,
)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be one or greater")
    return parsed


def _min_two_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be two or greater")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skillforge", description="Inspect a local repository")
    parser.add_argument("--verbose", action="store_true", help="enable diagnostic logging")
    commands = parser.add_subparsers(dest="command", required=True)
    connect = commands.add_parser(
        "connect", help="connect Codex discovery without model preparation")
    connect.add_argument("path", type=Path)
    connect.add_argument("--baseline", type=Path,
                         help="publish an existing audited baseline; no model calls")
    prepare = commands.add_parser("prepare", help="install selected repository Skills locally")
    prepare.add_argument("path", type=Path, help="local Python repository directory")
    prepare.add_argument("--task", required=True, help="task for Skill routing")
    prepare.add_argument("--target", required=True, choices=("codex", "claude_code"))
    prepare.add_argument("--current-module", help="exact current Python module for routing")
    prepare.add_argument("--baseline", type=Path, help="Evolution baseline for the next task")
    prepare.add_argument("--rebuild", action="store_true",
                         help="explicitly rebuild repository knowledge")
    prepare.add_argument("--json", action="store_true", help="print machine-readable result")
    evolve = commands.add_parser("evolve-plan", help="plan deterministic repository evolution")
    evolve.add_argument("path", type=Path, help="local Git repository root")
    evolve.add_argument("--baseline", required=True, type=Path, help="prepare Evolution baseline")
    evolve.add_argument("--json", action="store_true", help="print machine-readable summary")
    apply = commands.add_parser("evolve-apply", help="apply a frozen evolution plan")
    apply.add_argument("path", type=Path, help="local Git repository root")
    apply.add_argument("--baseline", required=True, type=Path)
    apply.add_argument("--plan", required=True, type=Path)
    apply.add_argument("--json", action="store_true", help="print machine-readable summary")
    verify = commands.add_parser("verify", help="verify repository changes from a prepare receipt")
    verify.add_argument("path", type=Path, help="local Git repository root")
    verify.add_argument("--receipt", required=True, type=Path, help="prepare Verify receipt JSON")
    verify.add_argument("--checks", type=Path, help="explicit checks JSON")
    verify.add_argument("--json", action="store_true", help="print the Verify report as JSON")
    feedback = commands.add_parser("feedback", help="record feedback on a Verify artifact")
    feedback.add_argument("path", type=Path, help="local Git repository root")
    feedback.add_argument("--report", required=True, type=Path, help="Verify artifact JSON")
    feedback.add_argument("--kind", required=True, choices=tuple(kind.value for kind in (
        FeedbackKind.FALSE_POSITIVE, FeedbackKind.FALSE_NEGATIVE,
        FeedbackKind.PERMANENT_WHITELIST, FeedbackKind.MODULE_EXCEPTION,
    )))
    feedback.add_argument("--rationale", required=True)
    feedback.add_argument("--check-id")
    feedback.add_argument("--target-kind", choices=tuple(kind.value for kind in
                          VerifyObligationKind))
    feedback.add_argument("--skill-id")
    feedback.add_argument("--ref-id")
    feedback.add_argument("--supersedes-feedback-id")
    feedback.add_argument("--json", action="store_true")
    analyze = commands.add_parser("analyze", help="build a structured repository profile")
    analyze.add_argument("path", type=Path, help="local repository directory")
    analyze.add_argument("--json", action="store_true", help="print the complete JSON profile")
    history = commands.add_parser("history", help="extract observed Git commit changes")
    history.add_argument("path", type=Path, help="local Git repository root")
    history.add_argument("--limit", type=_non_negative_int, default=20, help="maximum commits")
    history.add_argument("--json", action="store_true", help="print history and evidence as JSON")
    static = commands.add_parser("static", help="extract Python AST structure and imports")
    static.add_argument("path", type=Path, help="local repository directory")
    static.add_argument("--json", action="store_true", help="print static analysis as JSON")
    patterns = commands.add_parser("patterns", help="mine repeated file and module co-changes")
    patterns.add_argument("path", type=Path, help="local Git repository root")
    patterns.add_argument(
        "--min-support", type=_positive_int, default=2, help="minimum supporting commits"
    )
    patterns.add_argument(
        "--max-size", type=_min_two_int, default=5, help="maximum pattern size (at least 2)"
    )
    patterns.add_argument("--limit", type=_non_negative_int, default=20, help="maximum commits")
    patterns.add_argument("--json", action="store_true", help="print patterns as JSON")
    candidates = commands.add_parser(
        "workflow-candidates", help="mine evidence-backed workflow candidates"
    )
    candidates.add_argument("path", type=Path, help="local Git repository root")
    candidates.add_argument(
        "--min-support", type=_positive_int, default=2, help="minimum supporting commits"
    )
    candidates.add_argument(
        "--max-size", type=_min_two_int, default=5, help="maximum pattern size (at least 2)"
    )
    candidates.add_argument("--limit", type=_non_negative_int, default=20, help="maximum commits")
    candidates.add_argument("--json", action="store_true", help="print candidates as JSON")
    negative = commands.add_parser(
        "negative-patterns", help="mine negative-signal candidate patterns from Git history"
    )
    negative.add_argument("path", type=Path, help="local Git repository root")
    negative.add_argument("--limit", type=_non_negative_int, default=20, help="maximum commits")
    negative.add_argument(
        "--window-commits", type=_min_two_int, default=5,
        help="inclusive non-merge commit window for repeated edits",
    )
    negative.add_argument(
        "--window-days", type=_positive_float, default=14.0,
        help="maximum elapsed days for repeated edits",
    )
    negative.add_argument(
        "--json", action="store_true", help="print signals and candidates as JSON"
    )
    semantic_eval = commands.add_parser(
        "semantic-eval", help="evaluate workflow candidates with the local Codex CLI"
    )
    semantic_eval.add_argument("path", type=Path, help="local Git repository root")
    semantic_eval.add_argument("--provider", choices=("codex-cli",), required=True)
    semantic_eval.add_argument("--limit", type=_positive_int, default=5, help="maximum candidates")
    semantic_eval.add_argument("--kind", choices=("module", "file"), default="module")
    semantic_eval.add_argument("--output", type=Path, help="JSON evaluation report path")
    antipattern_eval = commands.add_parser(
        "antipattern-eval", help="evaluate negative-pattern candidates with the local Codex CLI"
    )
    antipattern_eval.add_argument("path", type=Path, help="local Git repository root")
    antipattern_eval.add_argument("--provider", choices=("codex-cli",), required=True)
    antipattern_eval.add_argument(
        "--history-limit", type=_non_negative_int, default=50, help="maximum Git commits"
    )
    antipattern_eval.add_argument(
        "--limit", type=_positive_int, default=5, help="maximum semantic candidates"
    )
    antipattern_eval.add_argument("--output", type=Path, help="JSON evaluation report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.command == "connect":
            from skillforge.integration.autonomous import install, register, repo_root
            from skillforge.productization.evolution_artifacts import load_baseline
            root = repo_root(args.path)
            if args.baseline is None:
                result = register(root)
            else:
                baseline = load_baseline(root, args.baseline)
                result = install(root, baseline.source_dna, baseline.source_compilation,
                                 baseline.head_commit)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.command == "prepare":
            from skillforge.productization.verified_prepare import automatic_prepare
            result = automatic_prepare(PrepareRequest(
                args.path, args.task, CodeAgentTarget(args.target), args.current_module,
                args.baseline,
            ), preparer=SkillForgePreparer(), rebuild=args.rebuild)
            publication = None
            if (result.target == CodeAgentTarget.CODEX
                    and result.semantic_failure_count == 0
                    and result.evolution_baseline_path and result.evolution_head_commit):
                from skillforge.integration.autonomous import refresh_connected
                try:
                    publication = refresh_connected(
                        result.repository_root, result.evolution_baseline_path,
                        expected_head=result.evolution_head_commit,
                        expected_snapshot_id=result.dna_snapshot_id,
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    raise SkillForgeError(
                        f"Prepared knowledge publication failed: {error}") from error
            if args.json:
                summary = result.to_dict()
                if publication is not None:
                    summary["discovery_publication"] = publication
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                _print_prepare(result)
                if publication is not None and publication["status"] == "REFRESHED":
                    print("Connected discovery knowledge refreshed.")
            return 0
        if args.command == "evolve-plan":
            try:
                summary = evolve_plan(args.path, args.baseline).summary
            except (OSError, ValueError, TypeError, KeyError, OverflowError,
                    subprocess.TimeoutExpired, SkillForgeError) as error:
                raise SkillForgeError(f"Evolution input or repository error: {error}") from error
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                _print_evolve_plan(summary)
            return 0
        if args.command == "evolve-apply":
            try:
                summary = evolve_apply(args.path, args.baseline, args.plan).summary
            except (OSError, ValueError, TypeError, KeyError, OverflowError,
                    RuntimeError, subprocess.TimeoutExpired, SkillForgeError) as error:
                raise SkillForgeError(f"Evolution apply error: {error}") from error
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                _print_evolve_apply(summary)
            return 0
        if args.command == "verify":
            try:
                requested = args.path.expanduser().absolute()
                if any(is_link(path) for path in (requested, *requested.parents)):
                    raise ValueError(f"Repository path must be a real directory: {requested}")
                root = requested.resolve()
                if not root.is_dir():
                    raise ValueError(f"Repository path is not a directory: {root}")
                receipt = load_prepare_receipt(root, args.receipt)
                checks, checks_path = load_checks(root, args.checks) if args.checks else ((), None)
                plan = VerifyPlanBuilder().with_checks(receipt.plan, checks)
                excluded_paths = set(receipt.ignored_patch_paths)
                if checks_path is not None:
                    excluded_paths.add(checks_path)
                report = IndependentVerifier().verify(
                    root, receipt.base_commit, plan,
                    excluded_paths=tuple(sorted(excluded_paths)),
                )
                report_path = save_verify_report(root, plan, report)
            except SkillForgeError:
                raise
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise SkillForgeError(f"Verify input or repository error: {error}") from error
            summary = _verify_summary(root, receipt.base_commit, report, report_path)
            if args.json:
                print(json.dumps({"summary": summary, "report": report.to_dict()},
                                 ensure_ascii=False, indent=2))
            else:
                _print_verify(summary, report)
            return 0 if report.overall_status == VerifyOverallStatus.PASSED else 1
        if args.command == "feedback":
            try:
                requested = args.path.expanduser().absolute()
                if any(is_link(path) for path in (requested, *requested.parents)):
                    raise ValueError(f"Repository path must be a real directory: {requested}")
                root = requested.resolve()
                if not root.is_dir():
                    raise ValueError(f"Repository path is not a directory: {root}")
                parts = (args.target_kind, args.skill_id, args.ref_id)
                if any(parts) and not all(parts):
                    raise ValueError("obligation requires --target-kind, --skill-id, and --ref-id")
                if not args.check_id and not all(parts):
                    raise ValueError("feedback requires --check-id or a complete obligation")
                relative_report = source_path(root, args.report.as_posix())
                _, report = load_verify_artifact(root, relative_report)
                target_ref = (VerifyTargetRef(VerifyObligationKind(args.target_kind),
                                              args.skill_id, args.ref_id)
                              if all(parts) else None)
                service = HumanFeedbackService()
                record = service.record_verify(
                    report, FeedbackKind(args.kind), args.rationale,
                    target_ref=target_ref, check_id=args.check_id,
                    supersedes_feedback_id=args.supersedes_feedback_id,
                )
                if (root / LEDGER_PATH).exists():
                    ledger, sources = load_feedback_ledger(root)
                else:
                    ledger, sources = HumanFeedbackLedger(report.repository_id, ()), {}
                ledger = service.append(ledger, record)
                sources[record.feedback_id] = relative_report
                ledger_path = save_feedback_ledger(root, ledger, sources)
            except SkillForgeError:
                raise
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise SkillForgeError(f"Feedback input or artifact error: {error}") from error
            summary = {"kind": record.kind.value, "feedback_id": record.feedback_id,
                       "source_verify_report": relative_report, "plan_id": report.plan_id,
                       "patch_snapshot_id": report.patch.snapshot_id,
                       "ledger": ledger_path, "ledger_id": ledger.ledger_id,
                       "active_feedback_count": len(ledger.active_records)}
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                print("Feedback recorded")
                for label, key in (("Kind", "kind"), ("Feedback ID", "feedback_id"),
                                   ("Source Verify report", "source_verify_report"),
                                   ("Plan", "plan_id"), ("Patch", "patch_snapshot_id"),
                                   ("Ledger", "ledger"), ("Ledger ID", "ledger_id"),
                                   ("Active feedback count", "active_feedback_count")):
                    print(f"{label}: {summary[key]}")
            return 0
        if args.command == "antipattern-eval":
            report = evaluate_semantic_antipatterns(
                args.path, history_limit=args.history_limit, limit=args.limit
            )
            output = (
                args.output or args.path / ".skillforge" / "reports" / "antipattern-eval.json"
            )
            saved = write_semantic_antipattern_evaluation_report(report, output)
            entries = report["entries"]
            accepted = sum(entry["status"] == "accepted_by_validator" for entry in entries)
            rejected = sum(entry["status"] == "semantic_rejected" for entry in entries)
            failures = len(entries) - accepted - rejected
            print(f"Candidates evaluated: {report['selected_count']}")
            print(f"Accepted Anti-patterns: {accepted}")
            print(f"Semantic rejections: {rejected}")
            print(f"Invalid/model failures: {failures}")
            print(f"Report: {saved.resolve()}")
            return 1 if failures else 0
        if args.command == "semantic-eval":
            kind = (
                ChangePatternKind.MODULE_CO_CHANGE if args.kind == "module"
                else ChangePatternKind.FILE_CO_CHANGE
            )
            report = evaluate_semantic_workflows(args.path, kind=kind, limit=args.limit)
            output = args.output or args.path / ".skillforge" / "reports" / "semantic-eval.json"
            saved = write_semantic_evaluation_report(report, output)
            accepted = sum(
                entry["status"] == "accepted_by_validator" for entry in report["entries"]
            )
            rejected = sum(entry["status"] == "semantic_rejected" for entry in report["entries"])
            print(f"Report: {saved.resolve()}")
            print(
                f"Candidates: {report['selected_count']}; workflows: {accepted}; "
                f"semantic rejections: {rejected}"
            )
            return 0 if accepted + rejected == report["selected_count"] else 1
        if args.command == "analyze":
            profile = RepositoryAnalyzer().analyze(args.path)
        elif args.command == "history":
            history = GitHistoryMiner().mine(args.path, limit=args.limit)
        elif args.command in {"patterns", "workflow-candidates"}:
            history = GitHistoryMiner().mine(args.path, limit=args.limit)
            static_result = PythonStaticAnalyzer().analyze(args.path)
            pattern_result = ChangePatternMiner(
                min_support=args.min_support, max_pattern_size=args.max_size
            ).mine(
                history, static_result
            )
            if args.command == "workflow-candidates":
                candidate_result = WorkflowCandidateMiner().mine(
                    history, static_result, pattern_result
                )
        elif args.command == "negative-patterns":
            history = GitHistoryMiner().mine(args.path, limit=args.limit)
            negative_result = NegativePatternMiner(
                window_commits=args.window_commits,
                window_days=args.window_days,
            ).mine(history)
        else:
            static_result = PythonStaticAnalyzer().analyze(args.path)
    except (SkillForgeError, BudgetExceeded) as error:
        if isinstance(error, BudgetExceeded):
            if getattr(args, "json", False):
                print(json.dumps({"status": "INVALID_BUDGET", "reason": str(error)}),
                      file=sys.stderr)
            else:
                print(f"skillforge: INVALID_BUDGET: {error}", file=sys.stderr)
        else:
            print(f"skillforge: {error}", file=sys.stderr)
        return 2
    if args.command == "static":
        if args.json:
            print(json.dumps(static_result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_static(static_result)
        return 0
    if args.command == "history":
        if args.json:
            print(json.dumps(history.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_history(history)
        return 0
    if args.command == "patterns":
        if args.json:
            print(json.dumps(pattern_result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_patterns(pattern_result)
        return 0
    if args.command == "workflow-candidates":
        if args.json:
            print(json.dumps(candidate_result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_workflow_candidates(candidate_result)
        return 0
    if args.command == "negative-patterns":
        if args.json:
            print(json.dumps(negative_result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_negative_patterns(negative_result)
        return 0
    if args.json:
        print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Repository: {profile.repository.root}")
        print(f"Files: {profile.file_count}")
        print(
            "Languages: "
            + (
                ", ".join(f"{item.language} {item.file_count}" for item in profile.languages)
                or "none"
            )
        )
        print("Frameworks: " + (", ".join(profile.frameworks) or "none detected"))
        print("Top-level directories: " + (", ".join(profile.top_level_directories) or "none"))
        print(f"Python modules: {len(profile.python_modules)}")
        for signal in profile.signals:
            print(f"{signal.category}: " + (", ".join(signal.paths) or "none"))
        print("Git: " + ("yes" if profile.git.is_git_repository else "no"))
        if profile.git.is_git_repository:
            print("Branch: " + (profile.git.branch or "detached/unknown"))
            print("HEAD: " + (profile.git.head_commit or "no commits"))
    return 0


def _print_prepare(result: PrepareResult) -> None:
    print("SkillForge prepare\n")
    print(f"Prepare mode: {result.prepare_mode.replace('_', ' ')}")
    print(f"Repository: {result.repository_root}")
    print(f"Repository ID: {result.repository_id}")
    print(f"Target: {result.target.value}\n")
    history_source = (
        "from baseline" if result.prepare_mode == "evolution_baseline"
        else "used" if result.history_participated else "unavailable"
    )
    print(f"Git history: {history_source} ({result.history_commit_count} commits)")
    print(
        f"Change patterns: {result.change_pattern_count}; "
        f"workflow candidates: {result.workflow_candidate_count}"
    )
    print(f"Anti-pattern candidates: {result.anti_pattern_candidate_count}")
    print(
        f"Workflow semantic decisions: {result.workflow_accepted_count} accepted, "
        f"{result.workflow_rejection_count} rejected "
        f"of {result.workflow_semantic_selected_count} selected; "
        f"{result.workflow_semantic_retry_count} retries"
    )
    print(
        f"Anti-pattern semantic decisions: {result.anti_pattern_accepted_count} accepted, "
        f"{result.anti_pattern_rejection_count} rejected "
        f"of {result.anti_pattern_semantic_selected_count} selected; "
        f"{result.anti_pattern_semantic_retry_count} retries"
    )
    print(f"Structural candidates: {result.structural_candidate_count}")
    print(f"Accepted structural rules: {result.semantic_accepted_count}")
    print(f"Semantic rejections: {result.semantic_rejection_count}")
    print(f"Semantic failures: {result.semantic_failure_count}")
    print(
        f"Project DNA: {result.dna_rule_count} rules, "
        f"{result.dna_workflow_count} workflows, "
        f"{result.dna_anti_pattern_count} anti-patterns"
    )
    print(f"Compiled skills: {result.compiled_skill_count}")
    if result.prepare_mode == "evolution_baseline":
        print(f"Source snapshot: {result.dna_snapshot_id}")
    print(f"Selected skills: {len(result.selected_skill_ids)}\n")
    if result.verify_available:
        print(f"Verify receipt: {result.verify_receipt_path}")
        print(f"Verify base: {result.verify_base_commit}")
    elif result.history_participated:
        print("Verify: unavailable (repository has no committed Git HEAD)")
    else:
        print("Verify: unavailable for non-Git repository")
    if result.evolution_available:
        print(f"Evolution baseline: {result.evolution_baseline_path}")
        print(f"Evolution head: {result.evolution_head_commit}")
    elif result.history_participated:
        print("Evolution: unavailable (repository has no committed Git HEAD)")
    else:
        print("Evolution: unavailable for non-Git repository")
    if result.installed_packages:
        print("Installed:")
        for package in result.installed_packages:
            print(f"  {package}")
        print("\nReady. Open the Code Agent normally in this repository and run the task.")
    else:
        print("No applicable project Skill was selected.")
        print("No fallback Skill was generated.")


def _print_evolve_plan(summary: dict[str, object]) -> None:
    print("SkillForge evolve-plan\n")
    for label, key in (
        ("Repository", "repository"), ("Baseline", "baseline"),
        ("Source snapshot", "source_snapshot_id"),
        ("Source HEAD", "source_head_commit"),
        ("Target HEAD", "target_head_commit"), ("New commits", "new_commits"),
    ):
        print(f"{label}: {summary[key]}")
    for label, key in (("Workflow candidates", "workflow_candidates"),
                       ("Anti-pattern candidates", "anti_pattern_candidates"),
                       ("Reviews", "reviews")):
        print(f"\n{label}:")
        for name, count in summary[key].items():
            print(f"  {name.replace('_', ' ')}: {count}")
    print("\nFeedback:")
    for label, key in (("active", "feedback_total_active_count"),
                       ("applicable", "feedback_applicable_count"),
                       ("stale ignored", "feedback_ignored_stale_count")):
        print(f"  {label}: {summary[key]}")
    print(f"\nPlan: {summary['plan_id']}")
    print(f"Plan artifact: {summary['plan_artifact']}")


def _print_evolve_apply(summary: dict[str, object]) -> None:
    print("SkillForge evolve-apply\n")
    for label, key in (("Repository", "repository"), ("Baseline", "baseline"),
                       ("Plan", "plan"), ("Source snapshot", "source_snapshot_id"),
                       ("Result snapshot", "result_snapshot_id"),
                       ("Target HEAD", "target_head_commit")):
        print(f"{label}: {summary[key]}")
    for label, key in (("Semantic refresh", "semantic_refresh"),
                       ("Knowledge", "knowledge"), ("Skills", "skills"),
                       ("Reviews remaining", "reviews_remaining")):
        print(f"\n{label}:")
        for name, count in summary[key].items():
            print(f"  {name.replace('_', ' ')}: {count}")
    print(f"\nResult: {summary['result_artifact']}")
    print(f"Next baseline: {summary['next_baseline']}")


def _verify_summary(root: Path, base_commit: str, report, report_path: str) -> dict[str, object]:
    statuses = {status: sum(result.status == status for result in report.checks)
                for status in VerifyCheckStatus}
    return {
        "repository": str(root),
        "repository_id": report.repository_id,
        "base_commit": base_commit,
        "patch_snapshot_id": report.patch.snapshot_id,
        "changed_file_count": len(report.patch.files),
        "checks": {
            "passed": statuses[VerifyCheckStatus.PASSED],
            "failed": statuses[VerifyCheckStatus.FAILED],
            "error": statuses[VerifyCheckStatus.ERROR],
            "timeout": statuses[VerifyCheckStatus.TIMED_OUT],
        },
        "obligation_count": len(report.obligations),
        "unverified_obligation_count": sum(
            finding.code == "UNVERIFIED_OBLIGATION" for finding in report.findings
        ),
        "overall_status": report.overall_status.value,
        "report_path": report_path,
    }


def _print_verify(summary: dict[str, object], report) -> None:
    checks = summary["checks"]
    print("SkillForge verify\n")
    print(f"Repository: {summary['repository']}")
    print(f"Base commit: {summary['base_commit']}")
    print(f"Patch snapshot: {summary['patch_snapshot_id']}")
    print(f"Changed files: {summary['changed_file_count']}")
    print(
        "Checks passed / failed / error / timeout: "
        f"{checks['passed']} / {checks['failed']} / {checks['error']} / {checks['timeout']}"
    )
    print(f"Obligations: {summary['obligation_count']}")
    print(f"Unverified obligations: {summary['unverified_obligation_count']}")
    print(f"Overall status: {summary['overall_status']}")
    print(f"Report: {summary['report_path']}")


def _print_history(history: GitHistoryResult) -> None:
    print(f"Repository: {history.repository_id}")
    print(f"Commits: {len(history.changesets)}")
    for change in history.changesets:
        print(f"\nCommit: {change.commit_id}")
        print(f"Time: {change.committed_at}")
        parents = ", ".join(change.parent_ids) or "none (root commit)"
        print(f"Parents: {parents}")
        print("Message:")
        for line in change.message.rstrip("\n").splitlines() or ["(empty)"]:
            print(f"  {line}")

        known_stats = [
            item
            for item in change.files
            if item.additions is not None and item.deletions is not None
        ]
        additions = sum(item.additions for item in known_stats if item.additions is not None)
        deletions = sum(item.deletions for item in known_stats if item.deletions is not None)
        unavailable = len(change.files) - len(known_stats)
        print(
            f"Changed files: {len(change.files)}; lines +{additions}/-{deletions}; "
            f"line counts unavailable: {unavailable}"
        )
        for item in change.files:
            if item.kind.value == "renamed":
                description = f"renamed {item.previous_path} -> {item.path}"
            else:
                description = f"{item.kind.value} {item.path}"
            if item.additions is None or item.deletions is None:
                counts = "(line counts unavailable)"
            else:
                counts = f"(+{item.additions}/-{item.deletions})"
            print(f"  {description} {counts}")


def _print_static(result: StaticAnalysisResult) -> None:
    symbols = [symbol for module in result.modules for symbol in module.symbols]
    imports = [relation for module in result.modules for relation in module.imports]
    parse_issues = sum(module.parse_issue is not None for module in result.modules)
    module_name_issues = sum(
        issue.issue_type == StaticAnalysisIssueType.MODULE_NAME_ERROR for issue in result.issues
    )
    print(f"Modules: {len(result.modules)}")
    print(f"Symbols: {len(symbols)}")
    print(f"Imports: {len(imports)}")
    print(f"Internal imports: {sum(relation.is_internal for relation in imports)}")
    print(f"Parse issues: {parse_issues}")
    if module_name_issues:
        print(f"Module name issues: {module_name_issues}")
    for module in result.modules[:5]:
        state = "parsed" if module.parse_issue is None else "parse issue"
        module_name = module.module if module.module is not None else "(unmapped module name)"
        print(f"  {module_name} — {module.path} ({state})")
        for symbol in module.symbols[:3]:
            print(f"    {symbol.kind.value}: {symbol.qualified_name}:{symbol.line}")


def _print_patterns(result: ChangePatternResult) -> None:
    print(f"Patterns: {len(result.patterns)}")
    for pattern in result.patterns:
        label = {
            "file_co_change": "file co-change",
            "module_co_change": "module co-change",
        }[pattern.kind.value]
        size = f" size={len(pattern.items)}" if len(pattern.items) > 2 else ""
        print(
            f"\n{label}{size} [{pattern.support.supporting}/{pattern.support.examined}, "
            f"{pattern.support_ratio:.1%}]"
        )
        for item in pattern.items:
            print(f"  {item.value}")
        if pattern.pair_association is not None:
            association = pattern.pair_association
            print(f"  jaccard: {association.jaccard:.3f}")
            print(f"  P(second|first): {association.second_given_first:.3f}")
            print(f"  P(first|second): {association.first_given_second:.3f}")


def _print_workflow_candidates(result: WorkflowCandidateResult) -> None:
    print(f"Workflow Candidates: {len(result.candidates)}")
    for candidate in result.candidates:
        label = "module" if candidate.kind.value == "module_co_change" else "file"
        print(
            f"\n{label} candidate [{candidate.support.supporting}/"
            f"{candidate.support.examined}, {candidate.support_ratio:.1%}]"
        )
        for member in candidate.members:
            if candidate.kind.value == "file_co_change" and member.current_module is not None:
                print(f"  {member.value} (current module: {member.current_module})")
            else:
                print(f"  {member.value}")
        print("pair associations:")
        for association in candidate.pair_associations:
            print(f"  {association.first} <-> {association.second}")
            print(f"    jaccard: {association.jaccard:.3f}")
            print(f"    P(second|first): {association.second_given_first:.3f}")
            print(f"    P(first|second): {association.first_given_second:.3f}")
        if candidate.structural_links:
            print("structural links:")
            for link in candidate.structural_links:
                print(f"  {link.source} -> {link.target} [internal import]")
        else:
            print("structural links: none observed")
        if candidate.structural_coherence is not None:
            coherence = candidate.structural_coherence
            print(
                "structural coherence: "
                f"{coherence.linked_pairs} / {coherence.possible_pairs} linked pairs"
            )
        if candidate.noise_signals:
            print("noise hints:")
            for signal in candidate.noise_signals:
                print(f"  {signal.kind.value}: {signal.target}")


def _print_negative_patterns(result: NegativePatternCandidateResult) -> None:
    print(f"Negative signals: {len(result.signals)}")
    print(f"Negative pattern candidates: {len(result.candidates)}")
    for signal in result.signals:
        qualifier = "heuristic hint" if signal.heuristic else "deterministic observation"
        commits = ", ".join(signal.commit_ids)
        paths = ", ".join(signal.affected_paths) or "(no changed path)"
        print(f"\n{signal.kind.value} [{qualifier}]")
        print(f"  commits: {commits}")
        print(f"  paths: {paths}")
        if signal.matched_terms:
            print(f"  matched terms: {', '.join(signal.matched_terms)}")
        if signal.referenced_commit_id is not None:
            print(f"  message references commit: {signal.referenced_commit_id}")
    for candidate in result.candidates:
        paths = ", ".join(candidate.affected_paths) or "(no changed path)"
        print(f"\ncandidate {candidate.candidate_id} [{paths}]")
        print(
            f"  signals: {candidate.signal_count}; "
            f"involved commits: {candidate.involved_commit_count}; "
            f"examined commits: {candidate.examined_commits}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
