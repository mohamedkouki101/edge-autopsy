"""Training labels (PLAN §3.2, task B1).

Signal at close t, fill at close t+lag; holdings at close d-1 earn the return of day d.

Two label kinds:
- fixed horizon (`rebalance=None`): the compounded return of days t+lag+1 ... t+lag+horizon,
  on every date. Base case lag=1, horizon=5: days t+2..t+6. Used for daily-sampled
  diagnostics (e.g. leak lab L5).
- holding period (`rebalance=` the rebalance dates; the project default, decision
  2026-11-08): at each rebalance date t with next rebalance t', the compounded return of days
  t+lag+1 ... t'+lag, i.e. from this fill to the next fill. This is exactly what the
  strategy earns on the position, including 4-day holiday weeks (17% of dev weeks), where a
  fixed 5-day label would include one day the strategy does not hold. `horizon` becomes the
  maximum allowed holding period, so the 6-day purge (PLAN §5.5) still covers every label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fixed(returns: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    growth = None
    for k in range(lag + 1, lag + horizon + 1):
        g = 1.0 + returns.shift(-k)
        growth = g if growth is None else growth * g
    return growth - 1.0


def _holding_period(
    returns: pd.DataFrame, rebalance: pd.DatetimeIndex, horizon: int, lag: int
) -> pd.DataFrame:
    pos = returns.index.get_indexer(rebalance)
    if (pos < 0).any():
        bad = list(rebalance[pos < 0][:3])
        raise ValueError(f"rebalance dates must be trading days in `returns`; e.g. {bad}")
    spans = np.diff(pos)
    if (spans <= 0).any():
        raise ValueError("rebalance dates must be strictly increasing")
    if (spans > horizon).any():
        k = int(np.argmax(spans))
        raise ValueError(
            f"holding period of {spans[k]} days after {rebalance[k]:%Y-%m-%d} exceeds "
            f"horizon={horizon}; the purge assumes labels end within t+lag+horizon"
        )
    growth = 1.0 + returns.to_numpy(dtype=float)
    n = len(returns)
    out = np.full((len(rebalance), returns.shape[1]), np.nan)
    for k in range(len(rebalance) - 1):
        first, last = pos[k] + lag + 1, pos[k + 1] + lag  # inclusive day positions
        if last < n:
            out[k] = growth[first : last + 1].prod(axis=0) - 1.0  # NaN propagates
    return pd.DataFrame(out, index=rebalance, columns=returns.columns)


def build_labels(
    returns: pd.DataFrame,
    horizon: int = 5,
    lag: int = 1,
    rebalance: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Long Series indexed by (date, asset). NaN if any return in the window is missing or
    the window runs past the end of the data. `returns` must be on the full trading calendar
    (shifts are in rows = trading days)."""
    if horizon < 1 or lag < 0:
        raise ValueError("need horizon >= 1 and lag >= 0")
    if rebalance is None:
        y = _fixed(returns, horizon, lag)
    else:
        y = _holding_period(returns, pd.DatetimeIndex(rebalance), horizon, lag)
    y = y.rename_axis(index="date", columns="asset").stack(future_stack=True)
    return y.rename("label")
