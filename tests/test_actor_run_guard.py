from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evals.actor_run_guard import ActorLaunchGuardError, guard_luna_actor_launch

TASK = b"frozen actor task\n"
TASK_SHA256 = hashlib.sha256(TASK).hexdigest()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "subject"
    repository.mkdir()
    return repository


def _argv(repository: Path, *options: str) -> list[str]:
    return ["codex", "exec", *options, "-m", "gpt-6-luna", "-C", str(repository)]


def _knowledge(repository: Path) -> None:
    root = repository / ".runtime" / "knowledge" / "current-source-rules"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("advisory skill", encoding="utf-8")
    (root / "references" / "evidence.md").write_text("source evidence", encoding="utf-8")


def test_valid_native_launch_derives_metadata_from_actual_inputs(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    metadata = guard_luna_actor_launch(
        _argv(repository, "-c", 'model_reasoning_effort="high"'),
        repository=repository,
        arm="native",
        stdin=TASK,
        task_sha256=TASK_SHA256,
    )

    assert metadata == {
        "model": "gpt-6-luna",
        "arm": "native",
        "repository": str(repository.resolve()),
        "stdin_sha256": TASK_SHA256,
        "knowledge_present": False,
    }


def test_valid_treatment_launch_requires_both_advisory_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _knowledge(repository)
    metadata = guard_luna_actor_launch(
        _argv(repository, "--config=service_tier=fast"),
        repository=repository,
        arm="treatment",
        stdin=TASK,
        task_sha256=TASK_SHA256,
    )

    assert metadata["model"] == "gpt-6-luna"
    assert metadata["arm"] == "treatment"
    assert metadata["knowledge_present"] is True


def test_wrong_model_cannot_be_overridden_by_irrelevant_metadata_text(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    argv = [
        "codex",
        "exec",
        "--comment",
        "metadata model=gpt-6-luna",
        "--model",
        "gpt-6.1-sol",
        "-C",
        str(repository),
    ]

    with pytest.raises(ActorLaunchGuardError, match="gpt-6-luna"):
        guard_luna_actor_launch(
            argv,
            repository=repository,
            arm="native",
            stdin=TASK,
            task_sha256=TASK_SHA256,
        )


@pytest.mark.parametrize(
    "model_options",
    [[], ["-m", "gpt-6-luna", "--model=gpt-6-luna"]],
    ids=["missing-model", "duplicate-model"],
)
def test_model_must_be_explicit_and_unique(tmp_path: Path, model_options: list[str]) -> None:
    repository = _repository(tmp_path)
    argv = ["codex", "exec", "-C", str(repository), *model_options]

    with pytest.raises(ActorLaunchGuardError, match="exactly one"):
        guard_luna_actor_launch(
            argv,
            repository=repository,
            arm="native",
            stdin=TASK,
            task_sha256=TASK_SHA256,
        )


@pytest.mark.parametrize(
    "config_option",
    [["-c", 'model="gpt-6.1-sol"'], ["--config=model='gpt-6-luna'"]],
    ids=["short-conflict", "long-same-model-override"],
)
def test_model_config_overrides_are_rejected_even_when_they_match(
    tmp_path: Path, config_option: list[str]
) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ActorLaunchGuardError, match="overrides"):
        guard_luna_actor_launch(
            _argv(repository, *config_option),
            repository=repository,
            arm="native",
            stdin=TASK,
            task_sha256=TASK_SHA256,
        )


def test_repository_argument_must_be_present_unique_and_exact(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    base = ["codex", "exec", "-m", "gpt-6-luna"]
    cases = [
        base,
        [*base, "-C", str(repository), "--cd", str(repository)],
        [*base, "-C", str(other)],
    ]
    for argv in cases:
        with pytest.raises(ActorLaunchGuardError, match="repository"):
            guard_luna_actor_launch(
                argv,
                repository=repository,
                arm="native",
                stdin=TASK,
                task_sha256=TASK_SHA256,
            )


def test_actual_stdin_bytes_must_match_the_frozen_task_hash(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    with pytest.raises(ActorLaunchGuardError, match="SHA-256"):
        guard_luna_actor_launch(
            _argv(repository),
            repository=repository,
            arm="native",
            stdin=b"different task",
            task_sha256=TASK_SHA256,
        )


def test_native_actor_rejects_any_knowledge_path(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    knowledge = repository / ".runtime" / "knowledge"
    knowledge.mkdir(parents=True)

    with pytest.raises(ActorLaunchGuardError, match="native actor"):
        guard_luna_actor_launch(
            _argv(repository),
            repository=repository,
            arm="native",
            stdin=TASK,
            task_sha256=TASK_SHA256,
        )


@pytest.mark.parametrize(
    "missing",
    ["SKILL.md", "evidence.md"],
    ids=["missing-skill", "missing-evidence"],
)
def test_treatment_actor_rejects_incomplete_knowledge(tmp_path: Path, missing: str) -> None:
    repository = _repository(tmp_path)
    _knowledge(repository)
    parent = "references" if missing == "evidence.md" else ""
    target = repository / ".runtime" / "knowledge" / "current-source-rules" / parent / missing
    target.unlink()

    with pytest.raises(ActorLaunchGuardError, match="requires"):
        guard_luna_actor_launch(
            _argv(repository),
            repository=repository,
            arm="treatment",
            stdin=TASK,
            task_sha256=TASK_SHA256,
        )
