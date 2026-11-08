"""Load one universe from panel.parquet as wide frames, never past `end` (task B5).

Before P9 the hold-out must be unreachable: asking for any date >= holdout_start raises.
V3 adds holdout_guard() (git tag + UNSEAL=I_ACCEPT) as the only way through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class HoldoutSealedError(PermissionError):
    """Raised when code asks for hold-out data before it is unsealed."""


@dataclass(frozen=True)
class Universe:
    returns: pd.DataFrame  # date x asset, simple daily returns (NaN = missing)
    available: pd.DataFrame  # date x asset, bool
    rf: pd.Series
    mkt: pd.Series
    volume_usd: pd.DataFrame


def load_universe(
    panel_path: Path, universe: str, end: pd.Timestamp, holdout_start: pd.Timestamp
) -> Universe:
    end, holdout_start = pd.Timestamp(end), pd.Timestamp(holdout_start)
    if end >= holdout_start:
        raise HoldoutSealedError(
            f"requested data up to {end:%Y-%m-%d}, but the hold-out starts "
            f"{holdout_start:%Y-%m-%d} and is sealed until P9"
        )
    df = pd.read_parquet(panel_path, filters=[("universe", "==", universe), ("date", "<=", end)])
    df = df[df["date"] <= end]  # belt and braces: never trust a filter alone
    piv = lambda col: df.pivot(index="date", columns="asset", values=col)  # noqa: E731
    ret = piv("ret")
    ret.columns = ret.columns.astype(str)
    av = piv("available").reindex(index=ret.index, columns=ret.columns).fillna(False)
    av = av.astype(bool)
    first = df.drop_duplicates("date").set_index("date").sort_index()
    return Universe(
        returns=ret,
        available=av,
        rf=first["rf"].reindex(ret.index),
        mkt=first["mkt"].reindex(ret.index),
        volume_usd=piv("volume_usd").reindex(index=ret.index, columns=ret.columns),
    )
