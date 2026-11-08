"""Rule-based baselines B0, B2-B4 (PLAN §2.2, §7.1). Every score at t uses data <= close t.

Windows are in trading days (rows of the daily returns frame). Higher score = more attractive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_targets(available: pd.DataFrame) -> pd.DataFrame:
    """B0: equal weight across available assets on each (rebalance) date."""
    a = available.astype(float)
    n = a.sum(axis=1).replace(0.0, np.nan)
    return a.div(n, axis=0).fillna(0.0)


def hist_mean_score(returns: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """B2: expanding mean of each asset's daily return up to t (the 'no information' ranker)."""
    return returns.expanding(min_periods=min_obs).mean()


def momentum_12_1_score(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """B3: compounded return over days t-(lookback-1) .. t-skip (12-1 momentum, N22).
    NaN unless every day in the window is observed."""
    window = lookback - skip
    logs = np.log1p(returns)
    return np.expm1(logs.rolling(window, min_periods=window).sum().shift(skip))


def reversal_1w_score(returns: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """B4: minus the compounded return of the last `days` trading days (buy last week's losers)."""
    logs = np.log1p(returns)
    return -np.expm1(logs.rolling(days, min_periods=days).sum())
