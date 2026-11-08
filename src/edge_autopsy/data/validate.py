"""validate_panel(): schema, range and calendar checks on panel.parquet (PLAN §5.2, task D6).

Raises DataError on anything that would silently corrupt results. Large returns
(|ret| >= flag_abs) are *flagged* in the returned report, not raised (PLAN §5.3 rule 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edge_autopsy.data.errors import DataError

PANEL_COLUMNS = ["date", "asset", "universe", "ret", "available", "rf", "mkt", "volume_usd"]
_DTYPES = {
    "date": "datetime64[ns]",
    "ret": "float64",
    "available": "bool",
    "rf": "float64",
    "mkt": "float64",
    "volume_usd": "float64",
}
UNIVERSES = {"ff49", "etf9"}


def validate_panel(
    df: pd.DataFrame,
    expected_assets: dict[str, set[str]],
    entry_days: int,
    sample_end: pd.Timestamp,
    flag_abs: float = 0.5,
) -> dict:
    if list(df.columns) != PANEL_COLUMNS:
        raise DataError(f"columns must be {PANEL_COLUMNS}, got {list(df.columns)}")
    for col, dt in _DTYPES.items():
        if str(df[col].dtype) != dt:
            raise DataError(f"dtype of {col!r} is {df[col].dtype}, expected {dt}")
    for col in ("asset", "universe"):
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            raise DataError(f"dtype of {col!r} must be category, got {df[col].dtype}")

    unis = set(df["universe"].astype(str).unique())
    if not unis <= UNIVERSES:
        raise DataError(f"unknown universe values: {unis - UNIVERSES}")
    if df.duplicated(["universe", "date", "asset"]).any():
        raise DataError("duplicate (universe, date, asset) rows")

    keys = ["universe", "asset"]
    grp = df.groupby(keys, observed=True, sort=False)
    if not grp["date"].apply(lambda s: s.is_monotonic_increasing).all():
        raise DataError("dates must be increasing within each (universe, asset)")

    for uni, want in expected_assets.items():
        got = set(df.loc[df["universe"] == uni, "asset"].astype(str).unique())
        if got != set(want):
            raise DataError(
                f"{uni} assets mismatch: missing {set(want) - got}, extra {got - set(want)}"
            )

    for col in ("ret", "rf", "mkt", "volume_usd"):
        v = df[col].to_numpy()
        if np.isinf(v).any():
            raise DataError(f"{col} has non-finite values (inf)")

    av = df["available"].to_numpy()
    if df.loc[av, "ret"].isna().any():
        raise DataError("available rows must have a non-missing return")
    for col in ("rf", "mkt"):
        if df.loc[av, col].isna().any():
            raise DataError(f"{col} missing on available rows")
    if (df["rf"].dropna() < 0).any():
        raise DataError("rf must be >= 0")

    n_seen = grp["ret"].transform(lambda s: s.notna().cumsum())
    if (df["available"] & (n_seen < entry_days)).any():
        raise DataError(f"available before the {entry_days}-day entry rule allows")

    for uni in unis:
        last = df.loc[df["universe"] == uni, "date"].max()
        if last != pd.Timestamp(sample_end):
            raise DataError(f"{uni} last date {last:%Y-%m-%d} != sample_end {sample_end:%Y-%m-%d}")

    flags = df.loc[df["ret"].abs() >= flag_abs, ["universe", "date", "asset", "ret"]]
    return {
        "n_rows": len(df),
        "n_available": int(av.sum()),
        "n_flags": len(flags),
        "flags": flags.reset_index(drop=True),
        "date_range": {
            u: (f"{d.min():%Y-%m-%d}", f"{d.max():%Y-%m-%d}")
            for u, d in df.groupby("universe", observed=True)["date"]
        },
    }
