"""Cross-platform task runner (Windows has no `make`). Usage: `uv run python tasks.py <task>`.

Tasks mirror the Makefile targets in PLAN §4. Later phases add dev, leaklab, unseal, etf,
figures and app.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def task_setup(_: argparse.Namespace) -> int:
    return _run(["uv", "sync", "--frozen", "--extra", "dev"])


def task_test(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "pytest", "-m", "not slow and not network"])


def task_test_all(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "pytest"])


def task_lint(_: argparse.Namespace) -> int:
    rc = _run([sys.executable, "-m", "ruff", "check", "."])
    return rc or _run([sys.executable, "-m", "ruff", "format", "--check", "."])


def task_data(args: argparse.Namespace) -> int:
    from edge_autopsy.data.build import build_all

    build_all(ROOT, offline=args.offline, refresh=args.refresh)
    return 0


TASKS = {
    "setup": task_setup,
    "test": task_test,
    "test-all": task_test_all,
    "lint": task_lint,
    "data": task_data,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--offline", action="store_true", help="data: never download; use data/raw")
    p.add_argument("--refresh", action="store_true", help="data: re-download even if present")
    args = p.parse_args(argv)
    return TASKS[args.task](args)


if __name__ == "__main__":
    raise SystemExit(main())
