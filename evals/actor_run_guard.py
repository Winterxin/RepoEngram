from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path


class ActorLaunchGuardError(ValueError):
    """Raised when an actor launch does not satisfy its prelaunch contract."""


_MODEL_OPTIONS = {"-m", "--model"}
_CONFIG_OPTIONS = {"-c", "--config"}
_REPOSITORY_OPTIONS = {"-C", "--cd"}
_TASK_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MODEL_CONFIG_RE = re.compile(r"^\s*(?:--)?model\s*=")


def _option_value(argv: Sequence[str], index: int, option: str) -> str:
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise ActorLaunchGuardError(f"{option} requires a value")
    return argv[index + 1]


def _model_config_override(value: str) -> bool:
    return bool(_MODEL_CONFIG_RE.match(value))


def _resolved_directory(value: str, *, label: str) -> Path:
    try:
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ActorLaunchGuardError(f"{label} is not an existing path") from exc
    if not resolved.is_dir():
        raise ActorLaunchGuardError(f"{label} is not a directory")
    return resolved


def guard_luna_actor_launch(
    argv: Sequence[str],
    *,
    repository: str | os.PathLike[str],
    arm: str,
    stdin: bytes,
    task_sha256: str,
) -> dict[str, object]:
    """Validate an actor command and its actual stdin without launching or modifying anything."""
    if not isinstance(arm, str) or arm not in {"native", "treatment"}:
        raise ActorLaunchGuardError("arm must be 'native' or 'treatment'")
    if not isinstance(stdin, bytes):
        raise ActorLaunchGuardError("stdin must be the exact bytes passed to the actor")
    if not isinstance(task_sha256, str) or not _TASK_HASH_RE.fullmatch(task_sha256):
        raise ActorLaunchGuardError("task_sha256 must be a 64-character hexadecimal digest")
    actual_stdin_sha256 = hashlib.sha256(stdin).hexdigest()
    if actual_stdin_sha256 != task_sha256.lower():
        raise ActorLaunchGuardError("stdin SHA-256 does not match task_sha256")

    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
        raise ActorLaunchGuardError("argv must be a non-empty sequence of strings")
    if any(not isinstance(argument, str) for argument in argv):
        raise ActorLaunchGuardError("argv must contain only strings")

    models: list[str] = []
    repository_arguments: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in _MODEL_OPTIONS:
            models.append(_option_value(argv, index, argument))
            index += 2
            continue
        if argument.startswith("--model="):
            models.append(argument.partition("=")[2])
            index += 1
            continue
        if argument.startswith("-m="):
            models.append(argument.partition("=")[2])
            index += 1
            continue
        if argument in _CONFIG_OPTIONS:
            config = _option_value(argv, index, argument)
            if _model_config_override(config):
                raise ActorLaunchGuardError("model overrides through -c/--config are forbidden")
            index += 2
            continue
        if argument.startswith("--config=") or argument.startswith("-c="):
            config = argument.partition("=")[2]
            if _model_config_override(config):
                raise ActorLaunchGuardError("model overrides through -c/--config are forbidden")
            index += 1
            continue
        if argument in _REPOSITORY_OPTIONS:
            repository_arguments.append(_option_value(argv, index, argument))
            index += 2
            continue
        if argument.startswith("--cd=") or argument.startswith("-C="):
            repository_arguments.append(argument.partition("=")[2])
        index += 1

    if len(models) != 1:
        raise ActorLaunchGuardError("argv must contain exactly one explicit model option")
    actual_model = models[0]
    if actual_model != "gpt-6-luna":
        raise ActorLaunchGuardError("actor model must be gpt-6-luna")
    if len(repository_arguments) != 1:
        raise ActorLaunchGuardError("argv must contain exactly one -C/--cd repository option")

    try:
        expected_repository_arg = Path(repository)
    except TypeError as exc:
        raise ActorLaunchGuardError("repository must be an absolute existing directory") from exc
    if not expected_repository_arg.is_absolute():
        raise ActorLaunchGuardError("repository must be an absolute existing directory")
    expected_repository = _resolved_directory(str(expected_repository_arg), label="repository")
    actual_repository = _resolved_directory(repository_arguments[0], label="argv repository")
    if os.path.normcase(str(actual_repository)) != os.path.normcase(str(expected_repository)):
        raise ActorLaunchGuardError("argv repository does not resolve to the expected repository")

    knowledge_root = expected_repository / ".runtime" / "knowledge"
    knowledge_exists = knowledge_root.exists() or knowledge_root.is_symlink()
    if arm == "native" and knowledge_exists:
        raise ActorLaunchGuardError("native actor repository must not contain .runtime/knowledge")
    if arm == "treatment":
        required_knowledge_files = (
            knowledge_root / "current-source-rules" / "SKILL.md",
            knowledge_root / "current-source-rules" / "references" / "evidence.md",
        )
        if not all(path.is_file() for path in required_knowledge_files):
            raise ActorLaunchGuardError(
                "treatment actor requires the source-rules SKILL.md and evidence.md"
            )

    return {
        "model": actual_model,
        "arm": arm,
        "repository": str(actual_repository),
        "stdin_sha256": actual_stdin_sha256,
        "knowledge_present": knowledge_exists,
    }
