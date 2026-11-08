"""B5: run baselines B0-B4 on the development period and log every run (PLAN §2.2, §7.1).

- Data is loaded only up to split.dev_test_end (2016-12-31); the loader refuses hold-out dates.
- Positions are built from data.sample_start; metrics are computed on the dev test window
  split.dev_test_start .. split.dev_test_end (1980-2016).
- Same engine, lag and cost model for every baseline (B1, the market index, has no trades).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from edge_autopsy.backtest import run_backtest
from edge_autopsy.config import ExperimentCfg, config_hash
from edge_autopsy.costs import CostModel
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.data.load import load_universe
from edge_autopsy.labels import build_labels
from edge_autopsy.metrics import ann_return, ic_summary, rank_ic, sharpe, summarize
from edge_autopsy.models.baselines import (
    equal_weight_targets,
    hist_mean_score,
    momentum_12_1_score,
    reversal_1w_score,
)
from edge_autopsy.portfolio import construct_weights
from edge_autopsy.runner import log_run, make_run_id

BASELINES = {
    "equal_weight_bh": "B0: equal weight, all available, monthly rebalance",
    "vw_market": "B1: French value-weighted market (Mkt-RF + RF), index, no trades",
    "hist_mean": "B2: expanding historical mean ranker, weekly top quintile",
    "momentum_12_1": "B3: 12-1 industry momentum, weekly top quintile",
    "reversal_1w": "B4: 1-week reversal, weekly top quintile",
}


def _run_hash(cfg: ExperimentCfg, model: str, params: dict) -> str:
    from pydantic import BaseModel

    class _Run(BaseModel):
        base: ExperimentCfg
        model: str
        params: dict

    return config_hash(_Run(base=cfg, model=model, params=params))


def run_baselines(
    cfg: ExperimentCfg,
    panel_path: Path,
    log_path: Path,
    results_dir: Path,
    data_version: str,
    config_path: str = "configs/base.yaml",
) -> list[dict]:
    sp, d = cfg.split, cfg.data
    dev_end = pd.Timestamp(sp.dev_test_end)
    u = load_universe(panel_path, cfg.universe, end=dev_end, holdout_start=sp.holdout_start)
    start = pd.Timestamp(d.sample_start)
    eval_slice = slice(pd.Timestamp(sp.dev_test_start), dev_end)
    cost = CostModel.from_cfg(cfg.costs)

    days = u.returns.index
    weekly = rebalance_dates(days, cfg.rebalance_freq)
    weekly = weekly[weekly >= start]
    monthly = rebalance_dates(days, "M")
    monthly = monthly[monthly >= start]
    reb = weekly if cfg.label == "holding_period" else None
    labels = build_labels(u.returns, cfg.horizon, cfg.lag, rebalance=reb)

    def score_weights(scores: pd.DataFrame) -> pd.DataFrame:
        return construct_weights(scores.loc[weekly], u.available.loc[weekly], cfg.portfolio)

    specs = {
        "equal_weight_bh": ("monthly", None, {"rebalance": "month_end"}),
        "vw_market": ("index", None, {}),
        "hist_mean": ("weekly", hist_mean_score(u.returns, 252), {"min_obs": 252}),
        "momentum_12_1": (
            "weekly",
            momentum_12_1_score(u.returns, *cfg.features.mom_12_1),
            {"lookback": cfg.features.mom_12_1[0], "skip": cfg.features.mom_12_1[1]},
        ),
        "reversal_1w": ("weekly", reversal_1w_score(u.returns, 5), {"days": 5}),
    }

    results, b0_net = [], None
    for model, (kind, scores, params) in specs.items():
        if kind == "index":
            gross = net = u.mkt
            turnover = None
            daily = pd.DataFrame({"gross": gross, "net": net})
        else:
            if kind == "monthly":
                w = equal_weight_targets(u.available.loc[monthly])
            else:
                w = score_weights(scores)
            bt = run_backtest(w, u.returns, u.rf, cost=cost, lag=cfg.lag)
            gross, net, turnover = bt.gross, bt.net, bt.turnover
            daily = pd.DataFrame(
                {"gross": bt.gross, "net": bt.net, "cost": bt.cost, "traded": bt.traded}
            )
        g, n = gross.loc[eval_slice], net.loc[eval_slice]
        rf = u.rf.loc[eval_slice]
        m = summarize(n, rf, turnover=turnover, benchmark=None if b0_net is None else b0_net)
        m = {
            "sharpe_net": m.pop("sharpe"),
            "sharpe_gross": sharpe(g, rf),
            "ann_return_net": m.pop("ann_return"),
            "ann_return_gross": ann_return(g),
            **m,
            "ic_mean": None,
            "ic_t_nw": None,
            "n_ic_weeks": None,
        }
        m["hit_rate_vs_b0"] = m.pop("hit_rate_vs_bench")
        if scores is not None:
            s = scores.loc[weekly].where(u.available.loc[weekly])
            s = s.rename_axis(index="date", columns="asset").stack(future_stack=True)
            ic = rank_ic(s, labels)
            ic = ic.loc[eval_slice]
            ics = ic_summary(ic, nw_lags=4)
            m.update(ic_mean=ics["ic_mean"], ic_t_nw=ics["ic_t_nw"], n_ic_weeks=ics["n"])
        if model == "equal_weight_bh":
            b0_net = n

        run_id = make_run_id()
        rel = Path("results") / run_id
        out_dir = Path(results_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        daily.loc[:dev_end].to_parquet(out_dir / "daily.parquet")
        record = {
            "config_hash": _run_hash(cfg, model, params),
            "config_path": config_path,
            "data_manifest_hash": data_version,
            "phase": "dev",
            "universe": cfg.universe,
            "model": model,
            "params": params,
            "split": {
                "scheme": "none (rule)",
                "eval": [str(sp.dev_test_start), str(dev_end.date())],
            },
            "portfolio": (
                {"type": "equal_weight_all", "rebalance": "monthly"}
                if model == "equal_weight_bh"
                else {"type": "index"}
                if model == "vw_market"
                else {**cfg.portfolio.model_dump(mode="json"), "rebalance": cfg.rebalance_freq}
            ),
            "cost_bps": cost.per_side_bps,
            "lag": cfg.lag,
            "label": cfg.label,
            "borrow_bps": cost.borrow_bps,
            "seed": cfg.seed,
            "n_inner_trials": 0,
            "metrics": m,
            "ci": None,
            "results_path": rel.as_posix() + "/",
            "pre_registered": False,
            "notes": BASELINES[model] + "; logged before PREREGISTRATION (V1).",
        }
        log_run(log_path, record, run_id=run_id)
        results.append({"run_id": run_id, "model": model, **m})
    return results
