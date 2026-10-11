"""Carry explicitly declared host permissions into native semantic children."""

from __future__ import annotations

import os
import subprocess

_PROFILES = {
    ":read-only": "read-only",
    ":workspace": "workspace-write",
    ":danger-full-access": "danger-full-access",
}
_MODES = frozenset(_PROFILES.values())


def resolve_semantic_sandbox(runner, explicit: str | None = None) -> str:
    """Custom transports own their policy; native children inherit built-in profiles.

    Resolve once at adapter construction, so cache identity and launch agree even
    if the coordinator environment changes later. Unknown named host profiles
    cannot be flattened to a broader built-in permission mode.
    """
    if explicit is not None:
        mode = explicit
    elif runner is not subprocess.run:
        return "read-only"
    elif "SKILLFORGE_SEMANTIC_SANDBOX" in os.environ:
        mode = os.environ["SKILLFORGE_SEMANTIC_SANDBOX"]
    elif "CODEX_PERMISSION_PROFILE" in os.environ:
        profile = os.environ["CODEX_PERMISSION_PROFILE"]
        if profile not in _PROFILES:
            raise ValueError(
                "Unsupported CODEX_PERMISSION_PROFILE for native semantic transport; "
                "use a policy-aware custom runner or explicitly set SKILLFORGE_SEMANTIC_SANDBOX"
            )
        mode = _PROFILES[profile]
    else:
        return "read-only"
    if mode not in _MODES:
        raise ValueError("Semantic sandbox must be read-only, workspace-write or danger-full-access")
    return mode


def semantic_permission_args(model) -> list[str]:
    mode = getattr(model, "_sandbox_mode", "read-only")
    if mode not in _MODES:
        raise ValueError("Invalid semantic sandbox mode")
    args = ["--sandbox", mode]
    if mode == "danger-full-access":
        args.extend(["-c", 'approval_policy="never"'])
    return args
