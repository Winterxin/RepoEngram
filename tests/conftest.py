"""Portable temporary storage and no paid model processes in release regression."""
from pathlib import Path
import sys


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = Path(config.rootpath) / ".runtime"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / "pytest-release")


def _no_paid_models(event, args):
    if event == "subprocess.Popen":
        name = Path(str(args[0])).name.lower()
        if name in {"codex", "codex.exe", "codex.cmd", "claude", "claude.exe", "claude.cmd"}:
            raise RuntimeError("Release regression must not start a paid model process")


sys.addaudithook(_no_paid_models)
