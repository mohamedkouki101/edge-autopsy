"""Daily backtest engine (PLAN §3.2, §3.3, task B2).

Timing: target weights dated t (signal at close t) are filled at close t+lag. Holdings at the
close of day d-1 earn the returns of day d. With lag=1 a signal at t first earns day t+2.

Conventions (each covered by a test in tests/unit/test_backtest.py):
- Weights are fractions of NAV; the uninvested remainder 1 - sum(w) is cash and earns rf
  (so short proceeds earn rf). The portfolio starts in cash.
- Between fills, weights drift with returns. Traded notional on a fill day is
  sum|target - drifted|; one-sided turnover is half of that.
- A row of target weights replaces the whole portfolio (missing assets -> 0).
- A NaN return on a held asset is treated as 0 for that day and counted in n_missing_held.
- Costs (if a cost model is given) are charged on the fill day; borrow accrues daily on short
  notional held during the day. net = gross - cost - borrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class CostLike(Protocol):
    def trade_cost(self, traded: float) -> float: ...
    def borrow_cost(self, short_notional: float) -> float: ...


@dataclass(frozen=True)
class BacktestResult:
    gross: pd.Series  # daily return before costs and borrow
    net: pd.Series  # daily return after costs and borrow
    cost: pd.Series  # daily trading cost (fill days only non-zero)
    borrow: pd.Series  # daily borrow fee
    traded: pd.Series  # daily sum|dw| (fill days only non-zero)
    turnover: pd.Series  # one-sided turnover, fill days only
    weights: pd.DataFrame  # end-of-day weights (after any trade at that close)
    fill_dates: pd.DatetimeIndex
    n_missing_held: int


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    rf: pd.Series,
    cost: CostLike | None = None,
    lag: int = 1,
) -> BacktestResult:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    dates = returns.index
    if not weights.index.isin(dates).all():
        bad = weights.index[~weights.index.isin(dates)][:3]
        raise ValueError(f"weight dates must be trading days in `returns`; e.g. {list(bad)}")
    unknown = weights.columns.difference(returns.columns)
    if len(unknown):
        raise ValueError(f"weights reference assets not in returns: {list(unknown)}")

    n, m = returns.shape
    R = returns.to_numpy(dtype=float)
    RF = rf.reindex(dates).to_numpy(dtype=float)
    if np.isnan(RF).any():
        raise ValueError("rf must be defined on every trading day of `returns`")
    targets = weights.reindex(columns=returns.columns).fillna(0.0).to_numpy(dtype=float)
    fill_pos = dates.get_indexer(weights.index) + lag
    schedule = {int(p): targets[k] for k, p in enumerate(fill_pos) if p < n}

    gross = np.zeros(n)
    cst = np.zeros(n)
    brw = np.zeros(n)
    trd = np.zeros(n)
    W = np.zeros((n, m))
    w = np.zeros(m)  # holdings at the previous close, as fractions of NAV
    n_missing = 0
    for d in range(n):
        r = R[d]
        missing = np.isnan(r)
        held_missing = missing & (w != 0)
        n_missing += int(held_missing.sum())
        r = np.where(missing, 0.0, r)
        cash = 1.0 - w.sum()
        port = float(w @ r + cash * RF[d])
        gross[d] = port
        if cost is not None:
            brw[d] = cost.borrow_cost(float(-w[w < 0].sum()))
        w = w * (1.0 + r) / (1.0 + port)  # drift to this close
        if d in schedule:
            target = schedule[d]
            trd[d] = float(np.abs(target - w).sum())
            if cost is not None:
                cst[d] = cost.trade_cost(trd[d])
            w = target.copy()
        W[d] = w

    fills = pd.DatetimeIndex(sorted(dates[p] for p in schedule), name="date")
    s = lambda a, name: pd.Series(a, index=dates, name=name)  # noqa: E731
    traded = s(trd, "traded")
    return BacktestResult(
        gross=s(gross, "gross"),
        net=s(gross - cst - brw, "net"),
        cost=s(cst, "cost"),
        borrow=s(brw, "borrow"),
        traded=traded,
        turnover=(traded.loc[fills] / 2.0).rename("turnover"),
        weights=pd.DataFrame(W, index=dates, columns=returns.columns),
        fill_dates=fills,
        n_missing_held=n_missing,
    )
