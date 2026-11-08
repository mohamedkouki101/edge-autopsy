"""Turn per-date scores into target weights (PLAN §5.6, task B6).

Rank 1 = highest score; ties broken by asset name (deterministic). Only assets that are
available and have a non-missing score on a date are eligible. n = max(1, round(q * n_eligible)).
- long_only: equal (or inverse-vol) weight on the top n; weights sum to 1.
- long_short: +1/n on the top n, -1/n on the bottom n (net 0, gross 2).
- band (long_only): keep a held asset while its rank <= exit; buy any asset ranked <= enter;
  hold the union (PLAN §5.6, S13). The number held can exceed `enter`.
A date with no eligible asset gives an all-zero (cash) row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edge_autopsy.config import PortfolioCfg


def _ranked(scores: pd.Series) -> list[str]:
    """Eligible assets ordered best first; ties by name."""
    df = pd.DataFrame({"s": scores.to_numpy(), "a": scores.index.astype(str)})
    df = df.sort_values(["s", "a"], ascending=[False, True], kind="stable")
    return df["a"].tolist()


def _sized(names: list[str], weighting: str, vol_row: pd.Series | None) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    if weighting == "equal":
        return pd.Series(1.0 / len(names), index=names)
    inv = 1.0 / vol_row.reindex(names)
    if inv.isna().any() or not np.isfinite(inv).all():
        raise ValueError(f"inverse_vol: missing/zero vol for {list(inv.index[~np.isfinite(inv)])}")
    return inv / inv.sum()


def construct_weights(
    scores: pd.DataFrame,
    available: pd.DataFrame,
    cfg: PortfolioCfg,
    vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cfg.weighting == "inverse_vol" and vol is None:
        raise ValueError("weighting='inverse_vol' needs a vol frame")
    if cfg.band is not None and cfg.kind != "long_only":
        raise NotImplementedError("buy/hold band is defined for long_only only")
    av = available.reindex(index=scores.index, columns=scores.columns).fillna(False)
    out = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    held: set[str] = set()
    for date in scores.index:
        row = scores.loc[date]
        row = row[av.loc[date].astype(bool) & row.notna()]
        order = _ranked(row)
        vol_row = vol.loc[date] if vol is not None else None
        if not order:
            held = set()
            continue
        n = max(1, round(cfg.quantile * len(order)))
        if cfg.kind == "long_short":
            longs = _sized(order[:n], cfg.weighting, vol_row)
            shorts = _sized(order[-n:], cfg.weighting, vol_row)
            out.loc[date, longs.index] = longs.to_numpy()
            out.loc[date, shorts.index] = -shorts.to_numpy()
            continue
        if cfg.band is None:
            pick = order[:n]
        else:
            enter, exit_ = cfg.band
            rank = {a: i + 1 for i, a in enumerate(order)}
            keep = {a for a in held if rank.get(a, np.inf) <= exit_}
            pick = [a for a in order if a in keep or rank[a] <= enter]
            held = set(pick)
        w = _sized(pick, cfg.weighting, vol_row)
        out.loc[date, w.index] = w.to_numpy()
    return out
