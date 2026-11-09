"""Synthetic panel with a known answer (PLAN §9.2).

49 assets, daily returns. A latent signal f_i,k (AR(1) across weeks, rho = 0.9, unit variance)
is observed at the close of rebalance date t_k. The return each asset earns over the holding
window of week k (fill at close t_k+1 -> next fill, the label's exact window) gets a drift
beta_w * f_i,k spread evenly over its days, so the true weekly rank IC of f against the
demeaned label equals `ic`. Market drift and rf are 0, so a positive Sharpe can only come
from selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.data.validate import PANEL_COLUMNS
from edge_autopsy.features import cs_rank


def make_synthetic(
    n_assets=49, years=30, start="1987-01-02", ic=0.05, rho=0.9, idio_vol=0.01,
    mkt_vol=0.004, seed=0,
):  # fmt: skip
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(pd.bdate_range(start, periods=years * 252), name="date")
    weekly = rebalance_dates(days)
    n_t, n_k = len(days), len(weekly)
    f = np.empty((n_k, n_assets))
    f[0] = rng.normal(size=n_assets)
    for k in range(1, n_k):
        f[k] = rho * f[k - 1] + np.sqrt(1 - rho**2) * rng.normal(size=n_assets)
    pearson = 2 * np.sin(np.pi * ic / 6)  # Spearman ic -> Pearson for normal data
    beta_w = pearson / np.sqrt(1 - pearson**2) * np.sqrt(5) * idio_vol
    mkt = rng.normal(0.0, mkt_vol, n_t)
    r = mkt[:, None] + rng.normal(0.0, idio_vol, (n_t, n_assets))
    pos = days.get_indexer(weekly)
    for k in range(n_k - 1):
        span = np.arange(pos[k] + 2, min(pos[k + 1] + 2, n_t))
        if len(span):
            r[span] += beta_w / len(span) * f[k]
    assets = [f"S{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(r, index=days, columns=assets)
    signal = pd.DataFrame(f, index=weekly, columns=assets)
    return returns, pd.Series(mkt, index=days, name="mkt"), signal


def write_panel(returns: pd.DataFrame, mkt: pd.Series, path: Path, entry_days: int = 253):
    n = len(returns)
    frames = []
    for a in returns.columns:
        frames.append(
            pd.DataFrame(
                {
                    "date": returns.index,
                    "asset": a,
                    "universe": "ff49",
                    "ret": returns[a].to_numpy(),
                    "available": np.arange(n) >= entry_days - 1,
                    "rf": 0.0,
                    "mkt": mkt.to_numpy(),
                    "volume_usd": np.nan,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype("datetime64[ns]")
    for c in ("asset", "universe"):
        df[c] = df[c].astype("category")
    df[PANEL_COLUMNS].to_parquet(path, index=False)


def long_rank(wide: pd.DataFrame, name: str) -> pd.Series:
    """Cross-sectional rank in [-0.5, 0.5] as a long (date, asset) column."""
    out = cs_rank(wide).rename_axis(index="date", columns="asset").stack(future_stack=True)
    return out.rename(name)
