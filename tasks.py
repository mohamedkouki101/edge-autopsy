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


def task_baselines(_: argparse.Namespace) -> int:
    """B5: baselines B0-B4 on the dev period (<= 2016-12-31); appends to experiments.jsonl."""
    from edge_autopsy.baselines_dev import run_baselines
    from edge_autopsy.config import load_config
    from edge_autopsy.data.download import manifest_hash

    cfg = load_config(ROOT / "configs" / "base.yaml")
    rows = run_baselines(
        cfg,
        ROOT / "data" / "processed" / "panel.parquet",
        ROOT / "experiments.jsonl",
        ROOT / "results",
        data_version=manifest_hash(ROOT / "data" / "raw"),
    )
    cols = [
        "model",
        "sharpe_gross",
        "sharpe_net",
        "ann_return_net",
        "ann_vol",
        "mdd",
        "turnover_ann",
        "hit_rate_vs_b0",
        "ic_mean",
        "ic_t_nw",
    ]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    return 0


def task_prereg(_: argparse.Namespace) -> int:
    """V1: check PREREGISTRATION.md against PREREGISTRATION.lock (fails on any edit)."""
    from edge_autopsy.prereg import PreregError, verify

    try:
        h = verify(ROOT / "PREREGISTRATION.md", ROOT / "PREREGISTRATION.lock")
    except PreregError as e:
        print(f"NOT verified: {e}")
        return 1
    print(f"PREREGISTRATION.md verified: sha256 {h}")
    return 0


def task_splits(_: argparse.Namespace) -> int:
    """V2: print the purged walk-forward folds on the dev period (expanding and rolling)."""
    from edge_autopsy.config import load_config
    from edge_autopsy.data.calendar import rebalance_dates
    from edge_autopsy.data.load import load_universe
    from edge_autopsy.splits import split_diagram, walk_forward

    cfg = load_config(ROOT / "configs" / "base.yaml")
    sp = cfg.split
    u = load_universe(
        ROOT / "data" / "processed" / "panel.parquet",
        cfg.universe,
        end=sp.dev_test_end,
        holdout_start=sp.holdout_start,
    )
    days = u.returns.index
    weekly = rebalance_dates(days, cfg.rebalance_freq)
    weekly = weekly[weekly >= str(cfg.data.sample_start)]
    for scheme in ("expanding", "rolling"):
        folds = walk_forward(weekly, days, sp.model_copy(update={"scheme": scheme}))
        print(
            f"\n== {scheme}: {len(folds)} folds, refit every {sp.refit_every} rebalances, "
            f"purge {sp.purge_days} trading days, embargo {sp.embargo_weeks} week(s) =="
        )
        print(split_diagram(folds))
    return 0


TASKS = {
    "setup": task_setup,
    "test": task_test,
    "test-all": task_test_all,
    "lint": task_lint,
    "data": task_data,
    "baselines": task_baselines,
    "prereg": task_prereg,
    "splits": task_splits,
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
