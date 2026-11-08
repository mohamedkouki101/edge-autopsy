"""Experiment runner and log (PLAN §7.4; tasks B5, P3.0, V5).

`run_experiment` is the one pipeline every model goes through: load (hold-out behind the
guard) -> scores -> portfolio -> backtest -> metrics -> results folder -> one JSONL row.
- `RuleModel`: scores from data <= t, no fitting (B2-B4). Positions from data.sample_start,
  exactly as B5, so a rule run reproduces the B5 numbers.
- `LearnedModel`: purged walk-forward; in each outer fold, nested tuning on inner folds only
  (tuning.tune), refit of the best config on the whole training window, predictions on the
  test block. Positions exist only on test dates (a model can't trade before its first test).
Rows are never edited or deleted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from edge_autopsy.backtest import run_backtest
from edge_autopsy.config import ExperimentCfg, config_hash
from edge_autopsy.costs import CostModel
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.data.load import Universe, load_universe
from edge_autopsy.labels import build_labels
from edge_autopsy.metrics import ann_return, ic_summary, oos_r2, rank_ic, sharpe, summarize
from edge_autopsy.models.baselines import equal_weight_targets
from edge_autopsy.portfolio import construct_weights
from edge_autopsy.seeds import set_all_seeds
from edge_autopsy.splits import holdout_guard, walk_forward
from edge_autopsy.tuning import tune

ROOT = Path(__file__).resolve().parents[2]


# Provenance without git (Kouki, 2026-11-08: this build never runs git). A run is stamped with
# a sha256 over everything that can change its numbers: the code, the configs and the pinned
# dependencies. Run outputs, docs and caches are not part of it.
CODE_DIRS = ("src", "configs")
CODE_FILES = ("pyproject.toml", "uv.lock", "tasks.py")
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _code_files(root: Path) -> list[Path]:
    files = [root / f for f in CODE_FILES if (root / f).is_file()]
    for d in CODE_DIRS:
        for f in (root / d).rglob("*"):
            cached = _SKIP_DIRS & set(f.relative_to(root).parts) or f.suffix == ".pyc"
            if f.is_file() and not cached:
                files.append(f)
    return sorted(files, key=lambda f: f.relative_to(root).as_posix())


def code_fingerprint(root: Path = ROOT) -> str:
    """sha256 over (relative path, content) of the code, configs and lock file.
    CRLF is normalised to LF so a Windows and a WSL checkout of the same code agree."""
    h = hashlib.sha256()
    for f in _code_files(root):
        rel = f.relative_to(root).as_posix().encode()
        data = f.read_bytes().replace(b"\r\n", b"\n")
        h.update(len(rel).to_bytes(8, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def make_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H%M%SZ") + "_" + secrets.token_hex(3)


def _json_safe(x):
    """NaN/inf are not valid JSON: write them as null."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    return x


def log_run(log_path: Path, record: dict, run_id: str | None = None) -> str:
    """Append `record` (+ run_id, code_hash, timestamp) to the JSONL log; return run_id."""
    if record.get("phase") == "holdout":
        holdout_guard()  # P9 only: UNSEAL=I_ACCEPT and an unchanged pre-registration
    row = {
        "run_id": run_id or make_run_id(),
        "logged_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_fingerprint(),
        **record,
    }
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    return row["run_id"]


# ---------------------------------------------------------------- run_experiment (V5)

PHASES = ("dev", "holdout", "leaklab", "exploratory")


@dataclass(frozen=True)
class RuleModel:
    name: str
    params: dict
    score: Callable[[Universe], pd.DataFrame]  # date x asset, from data <= t


@dataclass(frozen=True)
class LearnedModel:
    name: str
    factory: Callable[[dict], Any]  # params -> object with fit(X, y, X_val, y_val), predict(X)
    configs: list[dict]  # the pre-declared grid (<= 24)
    features: Callable[[Universe], pd.DataFrame]  # long (date, asset) frame, data <= t
    params: dict = field(default_factory=dict)  # fixed settings outside the grid


def run_hash(cfg: ExperimentCfg, model: str, params: dict) -> str:
    """config_hash of (base config, model, params): equal hashes = an identical re-run."""
    from pydantic import BaseModel

    class _Run(BaseModel):
        base: ExperimentCfg
        model: str
        params: dict

    return config_hash(_Run(base=cfg, model=model, params=params))


def _long(wide: pd.DataFrame) -> pd.Series:
    return wide.rename_axis(index="date", columns="asset").stack(future_stack=True)


def _walk_forward_scores(model: LearnedModel, u: Universe, weekly, labels, cfg, period):
    days = u.returns.index
    X = model.features(u)
    X = X[X.index.get_level_values("date").isin(weekly)]
    X = X[_long(u.available).reindex(X.index).fillna(False).astype(bool).to_numpy()]
    lab = labels.reindex(X.index)
    y = lab - lab.groupby(level="date").transform("mean")  # demeaned target (PREREG §3)
    folds = walk_forward(weekly, days, cfg.split, period=period)
    preds, log = [], []
    for f in folds:
        res = tune(model.factory, X, y, f, days, cfg.split, model.configs)
        tr = X.index.get_level_values("date").isin(f.train_dates) & y.notna().to_numpy()
        te = X.index.get_level_values("date").isin(f.test_dates)
        m = model.factory(res.best_params)
        m.fit(X[tr], y[tr], None, None)  # refit on the whole (purged) training window
        preds.append(pd.Series(m.predict(X[te]), index=X.index[te]))
        log.append(
            {
                "test_start": f"{f.test_dates[0]:%Y-%m-%d}",
                "test_end": f"{f.test_dates[-1]:%Y-%m-%d}",
                "train_end": f"{f.train_dates[-1]:%Y-%m-%d}",
                "best_params": res.best_params,
                "trials": res.trials,
            }
        )
    pred = pd.concat(preds).rename("score")
    wide = pred.unstack("asset").reindex(columns=u.returns.columns)
    return wide, pred, y, folds, log


def run_experiment(
    cfg: ExperimentCfg,
    model: RuleModel | LearnedModel,
    *,
    panel_path: Path,
    log_path: Path,
    results_dir: Path,
    data_version: str,
    config_path: str,
    phase: str = "dev",
    pre_registered: bool = False,
    notes: str = "",
) -> dict:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    sp = cfg.split
    if phase == "holdout":
        holdout_guard()  # before any data is read or any file is written
        lo, hi, period = sp.holdout_start, sp.holdout_end, "holdout"
    else:
        lo, hi, period = sp.dev_test_start, sp.dev_test_end, "dev"
    set_all_seeds(cfg.seed)
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    u = load_universe(panel_path, cfg.universe, end=hi, holdout_start=sp.holdout_start)
    days, start = u.returns.index, pd.Timestamp(cfg.data.sample_start)
    weekly = rebalance_dates(days, cfg.rebalance_freq)
    weekly = weekly[weekly >= start]
    monthly = rebalance_dates(days, "M")
    monthly = monthly[monthly >= start]
    reb = weekly if cfg.label == "holding_period" else None
    labels = build_labels(u.returns, cfg.horizon, cfg.lag, rebalance=reb)
    cost, ev = CostModel.from_cfg(cfg.costs), slice(lo, hi)

    b0 = run_backtest(
        equal_weight_targets(u.available.loc[monthly]), u.returns, u.rf, cost, cfg.lag
    )
    if isinstance(model, RuleModel):
        scores = model.score(u).loc[weekly]
        pred = _long(scores.where(u.available.loc[weekly]))
        params, split, trials, r2 = model.params, {"scheme": "none (rule)"}, None, None
        n_trials = n_fits = 0
    else:
        scores, pred, y, folds, trials = _walk_forward_scores(model, u, weekly, labels, cfg, period)
        params = {**model.params, "grid": model.configs}
        split = {
            "scheme": sp.scheme,
            "refit_every": sp.refit_every,
            "purge_days": sp.purge_days,
            "embargo_weeks": sp.embargo_weeks,
            "inner_folds": sp.inner_folds,
            "inner_fold_years": sp.inner_fold_years,
            "n_folds": len(folds),
        }
        r2 = oos_r2(pred, y, pred * 0.0)  # benchmark: 0 = the cross-sectional mean
        n_trials, n_fits = len(model.configs), len(folds) * len(model.configs) * sp.inner_folds
    split["eval"] = [f"{lo:%Y-%m-%d}", f"{hi:%Y-%m-%d}"]

    w = construct_weights(scores, u.available.loc[scores.index], cfg.portfolio)
    bt = run_backtest(w, u.returns, u.rf, cost=cost, lag=cfg.lag)
    g, n, rf = bt.gross.loc[ev], bt.net.loc[ev], u.rf.loc[ev]
    m = summarize(n, rf, turnover=bt.turnover, benchmark=b0.net.loc[ev])
    ics = ic_summary(rank_ic(pred, labels).loc[ev], nw_lags=4)
    m = {
        "sharpe_net": m.pop("sharpe"),
        "sharpe_gross": sharpe(g, rf),
        "ann_return_net": m.pop("ann_return"),
        "ann_return_gross": ann_return(g),
        **m,
        "ic_mean": ics["ic_mean"],
        "ic_t_nw": ics["ic_t_nw"],
        "n_ic_weeks": ics["n"],
        "oos_r2": r2,
    }
    m["hit_rate_vs_b0"] = m.pop("hit_rate_vs_bench")

    run_id = make_run_id()
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame({"gross": bt.gross, "net": bt.net, "cost": bt.cost, "traded": bt.traded})
    daily.loc[:hi].to_parquet(out / "daily.parquet")
    scores.loc[:hi].to_parquet(out / "scores.parquet")
    if trials is not None:
        (out / "trials.json").write_text(json.dumps(_json_safe(trials), indent=1), "utf-8")
    record = {
        "config_hash": run_hash(cfg, model.name, params),
        "config_path": config_path,
        "data_manifest_hash": data_version,
        "phase": phase,
        "universe": cfg.universe,
        "model": model.name,
        "params": params,
        "split": split,
        "portfolio": {**cfg.portfolio.model_dump(mode="json"), "rebalance": cfg.rebalance_freq},
        "cost_bps": cost.per_side_bps,
        "lag": cfg.lag,
        "label": cfg.label,
        "borrow_bps": cost.borrow_bps,
        "seed": cfg.seed,
        "n_inner_trials": n_trials,
        "n_inner_fits": n_fits,
        "metrics": m,
        "ci": None,
        "results_path": f"results/{run_id}/",
        "pre_registered": pre_registered,
        "notes": notes,
    }
    log_run(log_path, record, run_id=run_id)
    return {"run_id": run_id, **record}
