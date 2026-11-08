"""Trading-day helpers. The calendar is whatever dates the data has (no exchange calendar)."""

from __future__ import annotations

import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DatetimeIndex:
    """Last trading day in `index` of each `freq` period (e.g. Thursday in a Good Friday week).

    A partial final week is kept: its last available day is a rebalance date. Callers that
    need a label horizon after the date must drop dates without enough future data.
    """
    idx = pd.DatetimeIndex(index).sort_values().unique()
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period(freq)).max()
    return pd.DatetimeIndex(last.to_numpy(), name="date")
