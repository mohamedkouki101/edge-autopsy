"""The 14 model features (task M1; PREREGISTRATION §4 + Amendment 1, PLAN §5.4; leaks K2, K4).

Every window ends at close t and includes t; nothing after t is used (tested by mutating all
data after t). 11 per-asset features are cross-sectional ranks in [-0.5, 0.5] on each date
(no information from other dates); 3 market features are z-scored with an expanding
mean/std over strictly past days, starting at `zscore_start` (the 1963-07-01 sample floor in
experiments; Amendment 1).

`raw_features` returns the un-normalised values (wide date x asset frames, or date Series
for the market features) so each definition can be checked against a direct computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from edge_autopsy.config import FeatureCfg
from edge_autopsy.models.baselines import momentum_12_1_score

ASSET_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1",
    "vol_21", "vol_63", "beta_252", "mdd_63", "skew_21",
]  # fmt: skip
MARKET_FEATURES = ["mkt_ret_21", "mkt_vol_21", "disp_21"]
FEATURES = ASSET_FEATURES + MARKET_FEATURES
Z_MIN_PAST_DAYS = 252  # market z-scores are missing until this many past days (Amendment 1)


def _compound(x, w: int):
    """Compounded return over the last w days; NaN unless all w are observed."""
    return np.expm1(np.log1p(x).rolling(w, min_periods=w).sum())


def _max_drawdown(returns: pd.DataFrame, w: int) -> pd.DataFrame:
    """Max drawdown (<= 0) of the path over the last w days, starting from 1 at the window
    start. NaN if any return in the window is missing."""
    logs = np.log1p(returns.to_numpy(dtype=float))
    out = np.full(logs.shape, np.nan)
    if len(logs) >= w:
        for j in range(logs.shape[1]):
            win = sliding_window_view(logs[:, j], w)  # (n - w + 1, w), row k ends at k + w - 1
            cum = np.cumsum(win, axis=1)
            peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)  # start value 1 = log 0
            out[w - 1 :, j] = np.expm1((cum - peak).min(axis=1))
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def raw_features(returns: pd.DataFrame, market: pd.Series, cfg: FeatureCfg) -> dict:
    market = market.reindex(returns.index)
    out: dict = {}
    for w in cfg.momentum_windows:
        out[f"ret_{w}"] = _compound(returns, w)
    out["mom_12_1"] = momentum_12_1_score(returns, *cfg.mom_12_1)
    for w in cfg.vol_windows:
        out[f"vol_{w}"] = returns.rolling(w, min_periods=w).std()
    b = cfg.beta_window
    cov = returns.rolling(b, min_periods=b).cov(market)
    out[f"beta_{b}"] = cov.div(market.rolling(b, min_periods=b).var(), axis=0)
    out[f"mdd_{cfg.mdd_window}"] = _max_drawdown(returns, cfg.mdd_window)
    out[f"skew_{cfg.skew_window}"] = returns.rolling(
        cfg.skew_window, min_periods=cfg.skew_window
    ).skew()
    w = cfg.market_window
    out[f"mkt_ret_{w}"] = _compound(market, w)
    out[f"mkt_vol_{w}"] = market.rolling(w, min_periods=w).std()
    out[f"disp_{w}"] = _compound(returns, w).std(axis=1, ddof=1)
    missing = set(FEATURES) - set(out)
    if missing:
        raise ValueError(f"feature config does not produce the pre-registered set: {missing}")
    return {k: out[k] for k in FEATURES}


def cs_rank(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-date rank among assets with a value (ties averaged) -> (rank-1)/(n-1) - 0.5."""
    rank = wide.rank(axis=1, method="average")
    n = wide.notna().sum(axis=1)
    scaled = rank.sub(1.0).div((n - 1).where(n > 1), axis=0) - 0.5
    return scaled.mask(rank.notna() & (n == 1).to_numpy()[:, None], 0.0)


def expanding_z(x: pd.Series, start=None, min_past: int = Z_MIN_PAST_DAYS) -> pd.Series:
    """(x_t - mean) / std of x over past days s < t with s >= start; NaN until min_past."""
    hist = x if start is None else x.where(x.index >= pd.Timestamp(start))
    past = hist.shift(1).expanding(min_periods=min_past)
    z = (x - past.mean()) / past.std(ddof=1)
    return z if start is None else z.where(x.index >= pd.Timestamp(start))


def build_features(
    returns: pd.DataFrame,
    market: pd.Series,
    cfg: FeatureCfg,
    dates=None,
    zscore_start=None,
) -> pd.DataFrame:
    """Long frame indexed by (date, asset) with the 14 FEATURES columns.

    Computed on every day of `returns` (rolling windows and z-score history need it), then
    restricted to `dates` (e.g. rebalance dates) if given.
    """
    raw = raw_features(returns, market, cfg)
    keep = returns.index if dates is None else pd.DatetimeIndex(dates)
    cols = {}
    for f in ASSET_FEATURES:
        wide = cs_rank(raw[f]).loc[keep].rename_axis(index="date", columns="asset")
        cols[f] = wide.stack(future_stack=True)
    idx = cols[ASSET_FEATURES[0]].index
    dlev = idx.get_level_values("date")
    for f in MARKET_FEATURES:
        z = expanding_z(raw[f], start=zscore_start)
        cols[f] = pd.Series(z.reindex(dlev).to_numpy(), index=idx)
    return pd.DataFrame(cols)[FEATURES]


def universe_features(u, cfg) -> pd.DataFrame:
    """Features for an experiment: on rebalance dates, market z-scores from the sample floor."""
    from edge_autopsy.data.calendar import rebalance_dates

    dates = rebalance_dates(u.returns.index, cfg.rebalance_freq)
    return build_features(
        u.returns, u.mkt, cfg.features, dates=dates, zscore_start=cfg.data.sample_floor
    )
