"""Purged walk-forward splits (task V2; PLAN §5.5, leak K3).

Samples are rebalance dates. A sample dated t uses features from closes <= t and a label that
ends at most `purge_days` (= lag + horizon = 6) trading days after t: fill at close t+1,
hold until the next fill, never more than `horizon` days (labels.py raises otherwise).

- `label_end(t)`: the trading day `purge_days` after t (NaT if past the calendar's end, i.e.
  the label is not observable yet).
- `purge`: drop training samples whose label window [t, label_end(t)] overlaps the test span
  [first test date, label_end(last test date)]. Before the test block the rule is
  label_end(t) < first test date (strict; K3).
- `embargo`: also drop training samples within `embargo_weeks` x 5 trading days after the
  test span. Walk-forward training never follows its test block, so here it removes nothing;
  it matters for inner folds/CPCV, where training can sit after the validation block.
- `walk_forward`: test dates in [dev_test_start, dev_test_end] are cut into blocks of
  `refit_every` rebalances (52 = annual refit); each block gets a model trained on everything
  before it (expanding) or on the last `rolling_years` (rolling), purged. `period="holdout"`
  tests 2017-01-02 -> instead, and only after `holdout_guard()` lets it through.
- `holdout_guard`: the hold-out opens only with UNSEAL=I_ACCEPT and a pre-registration that
  still matches PREREGISTRATION.lock (V3; git-free substitute for the `prereg-v1` tag).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from edge_autopsy.config import SplitCfg
from edge_autopsy.prereg import PreregError, verify

ROOT = Path(__file__).resolve().parents[2]
TRADING_DAYS_PER_WEEK = 5
UNSEAL_PHRASE = "I_ACCEPT"


class HoldoutSealedError(PermissionError):
    """Raised when code asks for hold-out data before it is unsealed."""


def holdout_guard(doc: Path | None = None, lock: Path | None = None) -> str:
    """Return the frozen pre-registration sha256, or raise HoldoutSealedError.

    Both are required: the exact env var UNSEAL=I_ACCEPT (a deliberate act, P9 only) and a
    pre-registration that is frozen and unchanged, so the rules can't move after a peek.
    """
    doc = ROOT / "PREREGISTRATION.md" if doc is None else Path(doc)
    lock = ROOT / "PREREGISTRATION.lock" if lock is None else Path(lock)
    if os.environ.get("UNSEAL") != UNSEAL_PHRASE:
        raise HoldoutSealedError(
            f"the hold-out is sealed until P9: set UNSEAL={UNSEAL_PHRASE} to open it, once, "
            "with the frozen config"
        )
    try:
        return verify(doc, lock)
    except PreregError as e:
        raise HoldoutSealedError(
            f"hold-out stays sealed, pre-registration check failed: {e}"
        ) from e


@dataclass(frozen=True, eq=False)
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purged_dates: pd.DatetimeIndex  # removed from train: label window reaches the test block
    embargoed_dates: pd.DatetimeIndex  # removed from train: just after the test block


def _index(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates).sort_values().unique()


def _positions(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> np.ndarray:
    pos = calendar.get_indexer(dates)
    if (pos < 0).any():
        bad = dates[pos < 0][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"sample dates must be trading days in the calendar; not found: {bad}")
    return pos


def label_end(dates, calendar, purge_days: int) -> pd.DatetimeIndex:
    """Last trading day a sample's label can use: `purge_days` trading days after it."""
    dates, calendar = pd.DatetimeIndex(dates), _index(calendar)
    target = _positions(dates, calendar) + purge_days
    ok = target < len(calendar)
    out = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    out[ok] = calendar[target[ok]].to_numpy()
    return pd.DatetimeIndex(out)


def _test_span(test, calendar, purge_days) -> tuple[pd.Timestamp, pd.Timestamp]:
    test = _index(test)
    end = label_end(test[-1:], calendar, purge_days)[0]
    return test[0], (pd.Timestamp.max if pd.isna(end) else end)


def purge(train, test, calendar, purge_days: int):
    """Split `train` into (kept, purged) around one contiguous test block."""
    train, calendar = _index(train), _index(calendar)
    first, span_end = _test_span(test, calendar, purge_days)
    last = _index(test)[-1]
    inside = (train >= first) & (train <= last)
    if inside.any():
        raise ValueError("training dates fall inside the test block")
    ends = label_end(train, calendar, purge_days)
    before = train < first
    overlap = np.where(before, ends.isna() | (ends >= first), train <= span_end)
    return train[~overlap], train[overlap]


def embargo(train, test, calendar, purge_days: int, embargo_weeks: int):
    """Split `train` into (kept, embargoed): samples just after the test labels end."""
    train, calendar = _index(train), _index(calendar)
    _, span_end = _test_span(test, calendar, purge_days)
    if span_end == pd.Timestamp.max:
        stop = pd.Timestamp.max
    else:
        p = calendar.get_loc(span_end) + embargo_weeks * TRADING_DAYS_PER_WEEK
        stop = calendar[p] if p < len(calendar) else pd.Timestamp.max
    hit = (train > _index(test)[-1]) & (train <= stop)
    return train[~hit], train[hit]


def walk_forward(
    dates, calendar, cfg: SplitCfg, period: Literal["dev", "holdout"] = "dev"
) -> list[Fold]:
    dates, calendar = _index(dates), _index(calendar)
    _positions(dates, calendar)
    if period == "holdout":
        holdout_guard()
        lo, hi = pd.Timestamp(cfg.holdout_start), pd.Timestamp(cfg.holdout_end)
    elif period == "dev":
        lo, hi = pd.Timestamp(cfg.dev_test_start), pd.Timestamp(cfg.dev_test_end)
    else:
        raise ValueError(f"period must be 'dev' or 'holdout', got {period!r}")
    tests = dates[(dates >= lo) & (dates <= hi)]
    if tests.empty:
        raise ValueError(f"no sample dates in the test window {lo:%Y-%m-%d}..{hi:%Y-%m-%d}")
    if cfg.scheme == "rolling" and cfg.rolling_years < cfg.min_train_years:
        raise ValueError(f"rolling_years={cfg.rolling_years} < min_train_years")
    need = tests[0] - pd.DateOffset(years=cfg.min_train_years)
    if dates[0] > need:
        raise ValueError(
            f"min_train_years={cfg.min_train_years}: first test date {tests[0]:%Y-%m-%d} needs "
            f"samples from {need:%Y-%m-%d}, but they start {dates[0]:%Y-%m-%d}"
        )
    folds = []
    for k in range(0, len(tests), cfg.refit_every):
        test = tests[k : k + cfg.refit_every]
        cand = dates[dates < test[0]]
        if cfg.scheme == "rolling":
            cand = cand[cand >= test[0] - pd.DateOffset(years=cfg.rolling_years)]
        kept, purged = purge(cand, test, calendar, cfg.purge_days)
        kept, emb = embargo(kept, test, calendar, cfg.purge_days, cfg.embargo_weeks)
        if kept.empty:
            raise ValueError(f"fold starting {test[0]:%Y-%m-%d} has no training samples")
        folds.append(Fold(kept, test, purged, emb))
    return folds


def inner_folds(train_dates, calendar, cfg: SplitCfg) -> list[Fold]:
    """Nested-tuning folds inside one outer training window (V4, PLAN §5.5).

    The last `inner_folds` x `inner_fold_years` years of `train_dates` become consecutive
    validation blocks; each is trained on the training dates before it, purged. Only
    `train_dates` are ever used, so the outer test block cannot reach tuning.
    """
    train, calendar = _index(train_dates), _index(calendar)
    if train.empty:
        raise ValueError("inner folds need a non-empty training window")
    end, k, yrs = train[-1], cfg.inner_folds, cfg.inner_fold_years
    bounds = [end - pd.DateOffset(years=yrs * (k - j)) for j in range(k)] + [end]
    folds = []
    for j in range(k):
        val = train[(train > bounds[j]) & (train <= bounds[j + 1])]
        kept, purged = purge(train[train <= bounds[j]], val, calendar, cfg.purge_days)
        if val.empty or kept.empty:
            raise ValueError(
                f"inner fold {j + 1}/{k} is empty: the training window "
                f"{train[0]:%Y-%m-%d}..{end:%Y-%m-%d} is too short for {k} x {yrs}-year folds"
            )
        folds.append(Fold(kept, val, purged, train[:0]))
    return folds


def split_diagram(folds: list[Fold]) -> str:
    """One line per fold: a bar with one character per year (= train, # test, x purged only)."""
    y0 = min(f.train_dates[0].year for f in folds)
    y1 = max(f.test_dates[-1].year for f in folds)
    years = range(y0, y1 + 1)
    axis = "".join(str(y)[-1] if y % 10 == 0 else ("'" if y % 5 == 0 else " ") for y in years)
    out = [f"years {y0}..{y1} (one char per year; '0' marks a decade)", f"         |{axis}|"]
    for i, f in enumerate(folds, 1):
        ty, tr, pu = (set(d.year for d in x) for x in (f.test_dates, f.train_dates, f.purged_dates))
        bar = "".join(
            "#" if y in ty else "=" if y in tr else "x" if y in pu else "." for y in years
        )
        out.append(
            f"fold {i:02d} |{bar}| train {f.train_dates[0]:%Y-%m-%d}..{f.train_dates[-1]:%Y-%m-%d}"
            f" n={len(f.train_dates)} | purged {len(f.purged_dates)}"
            f" | test {f.test_dates[0]:%Y-%m-%d}..{f.test_dates[-1]:%Y-%m-%d}"
            f" n={len(f.test_dates)}"
        )
    return "\n".join(out)
