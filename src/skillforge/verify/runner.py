"""Serial execution of explicitly configured, bounded deterministic checks."""

from __future__ import annotations

import subprocess
import os
import json
import hashlib
import shutil
import tempfile
from pathlib import Path

from skillforge.cost_control import CostLedger, digest
from skillforge.domain.serialization import to_json_value

from skillforge.domain.verification import (
    VerifyCheckResult,
    VerifyCheckStatus,
    VerifyPlan,
    VerifyReport,
    derive_verify_findings,
    derive_verify_overall_status,
)
from skillforge.filesystem_safety import no_symlink
from skillforge.verify.git_patch import GitPatchCollector

_OUTPUT_CAP = 64 * 1024
_TRUNCATION = "\n[output truncated at 65536 bytes]"


def _bounded(path: Path) -> str:
    with path.open("rb") as stream:
        content = stream.read(_OUTPUT_CAP + 1)
    return content[:_OUTPUT_CAP].decode("utf-8", "replace") + (
        _TRUNCATION if len(content) > _OUTPUT_CAP else ""
    )


class IndependentVerifier:
    def __init__(self, collector: GitPatchCollector | None = None) -> None:
        self.collector = collector or GitPatchCollector()

    @staticmethod
    def runtime_directory(root):
        from skillforge.repository.identity import repository_identity
        temp = Path(os.environ.get("TEMP") or tempfile.gettempdir()).absolute()
        no_symlink(temp, temp)
        temp = temp.resolve()
        if temp.is_relative_to(root):
            ignored = subprocess.run(["git", "-c", f"safe.directory={root}", "check-ignore", "-q",
                                      str(temp / "sf-v" / "probe")], cwd=root, capture_output=True)
            if ignored.returncode:
                raise ValueError("Verify TEMP inside repository must be Git-ignored")
        runtime = temp / "sf-v" / repository_identity(root)
        no_symlink(runtime, temp)
        runtime.mkdir(parents=True, exist_ok=True)
        return runtime

    def _run_check(self, root: Path, spec) -> VerifyCheckResult:
        from skillforge.repository.repo_snapshot_semantic_cache import _snapshot, _production_version
        cwd = root if spec.cwd == "." else root.joinpath(*spec.cwd.split("/"))
        no_symlink(cwd, root)
        if not cwd.is_dir() or not cwd.resolve().is_relative_to(root):
            raise ValueError(f"check cwd must be a directory within repo_root: {spec.cwd}")
        environment = dict(os.environ)
        runtime = self.runtime_directory(root)
        environment.update(TMPDIR=str(runtime), PYTEST_DEBUG_TEMPROOT=str(runtime))
        executable = shutil.which(spec.argv[0]) or spec.argv[0]
        executable_hash = hashlib.sha256(Path(executable).read_bytes()).hexdigest() if Path(executable).is_file() else None
        input_files = {}
        for argument in spec.argv[1:]:
            try:
                candidate = Path(argument)
                candidate = candidate if candidate.is_absolute() else cwd / candidate
                if candidate.is_file():
                    input_files[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                pass  # Inline command text is not a filesystem dependency.
        identity = {"repository_root": str(root), "snapshot": _snapshot(root), "spec": to_json_value(spec),
                    "environment": digest(environment), "executable": executable_hash,
                    "production_version": _production_version(), "argv_input_files": input_files,
                    "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        key = digest(identity)
        # Short physical paths also work in deeply nested Windows workspaces.
        # Full identity/checksum is validated, so a prefix collision fails closed.
        directory = runtime / key[:20]
        no_symlink(directory, runtime)
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / "result.json"
        ledger = CostLedger(runtime / "events.jsonl")
        # Reserve before reading/running a check. Concurrent callers must never
        # launch the same expensive subprocess; they can query after completion.
        reservation_path = directory / "execution.lock"
        try:
            reservation = reservation_path.open("x", encoding="ascii")
        except FileExistsError as error:
            raise ValueError("identical check already running; no duplicate subprocess") from error
        try:
            with reservation:
                reservation.write(str(os.getpid()))
            return self._execute_check(spec, cwd, environment, identity, directory, cached, ledger)
        finally:
            reservation_path.unlink()

    def _execute_check(self, spec, cwd, environment, identity, directory, cached, ledger):
        with ledger.measure("check", identity, cache="miss", bypass_reason=None) as event:
            if cached.exists():
                entry = json.loads(cached.read_text("utf-8"))
                if entry.get("identity") != identity or entry.get("checksum") != digest(entry.get("result")):
                    raise ValueError("check cache identity/checksum mismatch")
                value = entry["result"]
                event["cache"] = "hit"
                return VerifyCheckResult(spec.check_id, spec.kind, VerifyCheckStatus(value["status"]),
                                         spec.argv, spec.cwd, value["exit_code"], value["stdout"],
                                         value["stderr"], spec.covers)
            stdout_file = directory / "stdout"
            stderr_file = directory / "stderr"
            status = VerifyCheckStatus.ERROR
            exit_code = None
            detail = ""
            with stdout_file.open("wb") as stdout, stderr_file.open("wb") as stderr:
                try:
                    result = subprocess.run(spec.argv, cwd=cwd, stdout=stdout, stderr=stderr,
                                            timeout=spec.timeout_seconds, shell=False,
                                            check=False, env=environment)
                    exit_code = result.returncode
                    status = (VerifyCheckStatus.PASSED if exit_code == 0
                              else VerifyCheckStatus.FAILED)
                except subprocess.TimeoutExpired:
                    status = VerifyCheckStatus.TIMED_OUT
                except OSError as exc:
                    detail = f"\n[execution error: {type(exc).__name__}: {exc}]"
            result = VerifyCheckResult(spec.check_id, spec.kind, status, spec.argv,
                                       spec.cwd, exit_code, _bounded(stdout_file),
                                       _bounded(stderr_file) + detail, spec.covers)
            value = to_json_value(result)
            cached.write_text(json.dumps({"identity": identity, "result": value,
                                          "checksum": digest(value)}, ensure_ascii=False), "utf-8")
            event.update(exit_code=exit_code, logs=str(directory))
            return result

    def verify(self, repo_root: Path | str, base_commit: str,
               plan: VerifyPlan, *, excluded_paths: tuple[str, ...] = ()) -> VerifyReport:
        if not isinstance(plan, VerifyPlan):
            raise ValueError("plan must be VerifyPlan")
        patch = self.collector.snapshot(repo_root, plan.repository_id, base_commit,
                                        excluded_paths=excluded_paths)
        root = Path(repo_root).expanduser().resolve()
        results = tuple(self._run_check(root, check) for check in plan.checks)
        findings = derive_verify_findings(plan, results)
        overall = derive_verify_overall_status(plan, results, findings)
        return VerifyReport(plan.repository_id, plan.registry_id, plan.source_snapshot_id,
                            plan.plan_id, patch, results, plan.obligations,
                            findings, overall, plan.evidence, plan=plan)
