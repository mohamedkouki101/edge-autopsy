"""Universe rules and audit helpers (PLAN §5.3 rule 4, §5.5; task D3)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from edge_autopsy.data.calendar import rebalance_dates


def availability(returns: pd.DataFrame, entry_days: int) -> pd.DataFrame:
    """True where the return is observed and the asset has >= `entry_days` non-missing days
    up to and including that date. Uses only data <= t (cumulative count)."""
    observed = returns.notna()
    n_seen = observed.cumsum()
    return (observed & (n_seen >= entry_days)).astype(bool)


def first_sample_date(
    avail: pd.DataFrame,
    min_assets: int,
    freq: str = "W-FRI",
    floor: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """First rebalance date on or after `floor` with at least `min_assets` assets available."""
    rb = rebalance_dates(avail.index, freq)
    if floor is not None:
        rb = rb[rb >= pd.Timestamp(floor)]
    counts = avail.loc[rb].sum(axis=1)
    ok = counts[counts >= min_assets]
    if ok.empty:
        raise ValueError(f"no rebalance date has >= {min_assets} available assets")
    return pd.Timestamp(ok.index[0])


def top_abs_returns(returns: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n largest |returns| as rows (date, asset, ret), largest first."""
    long = (
        returns.rename_axis(index="date", columns="asset")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="ret")
        .dropna(subset=["ret"])
    )
    order = long["ret"].abs().sort_values(ascending=False, kind="stable").index[:n]
    return long.loc[order].reset_index(drop=True)


def write_sample_start(yaml_path: Path, date: pd.Timestamp) -> None:
    """Set `sample_start:` in base.yaml in place, keeping comments and layout."""
    p = Path(yaml_path)
    text = p.read_text(encoding="utf-8")
    pat = re.compile(r"^(\s*sample_start:)\s*\S+", re.MULTILINE)
    new, n = pat.subn(rf"\g<1> {date:%Y-%m-%d}", text)
    if n != 1:
        raise ValueError(f"expected one 'sample_start:' line in {p}, found {n}")
    p.write_text(new, encoding="utf-8")
