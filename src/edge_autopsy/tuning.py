"""Nested grid search on inner folds only (task V4; PLAN §5.5, §6, leak K5).

For one outer fold: the rows of X/y outside `fold.train_dates` are dropped first, so the outer
test block never reaches a model. Each config in the pre-declared grid is fitted on every
inner fold's (purged) training dates and scored by the mean per-date rank IC on its
validation year; the score is the mean over inner folds. The best config is refit on the
whole training window by the caller (V5).

Model protocol: `factory(params)` returns an object with `fit(X, y, X_val, y_val)` (the val
fold is there for early stopping, XGB/LSTM) and `predict(X) -> Series` on X's index.
Every trial is returned for the trial count (DSR N = dev configs + inner trials, PLAN §7.4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edge_autopsy.config import SplitCfg
from edge_autopsy.metrics import rank_ic
from edge_autopsy.splits import Fold, inner_folds

MAX_CONFIGS = 24  # fairness rule: same tuning budget for every model (PLAN §6)


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    trials: list[dict]  # one per config: params, inner_ic (per fold), mean_ic


def grid(spec: dict[str, list]) -> list[dict]:
    """Cartesian product of the declared values, in declared order (ties go to the first)."""
    keys = list(spec)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*spec.values())]


def _rows(df, dates: pd.DatetimeIndex):
    return df[df.index.get_level_values("date").isin(dates)]


def tune(
    factory, X: pd.DataFrame, y: pd.Series, fold: Fold, calendar, cfg: SplitCfg, configs: list[dict]
) -> TuneResult:
    if not configs:
        raise ValueError("empty tuning grid")
    if len(configs) > MAX_CONFIGS:
        raise ValueError(f"{len(configs)} configs > budget of {MAX_CONFIGS} per model per refit")
    if fold.train_dates.isin(fold.test_dates).any():
        raise ValueError("training dates overlap the outer test block: refusing to tune on test")
    X, y = _rows(X, fold.train_dates), _rows(y.reindex(X.index), fold.train_dates)
    assert not X.index.get_level_values("date").isin(fold.test_dates).any()  # K5
    inner = inner_folds(fold.train_dates, calendar, cfg)

    def part(dates):
        yy = _rows(y, dates).dropna()
        return X.loc[yy.index], yy

    splits = [(part(f.train_dates), part(f.test_dates)) for f in inner]
    trials = []
    for params in configs:
        ics = []
        for (X_tr, y_tr), (X_va, y_va) in splits:
            model = factory(params)
            model.fit(X_tr, y_tr, X_va, y_va)
            ics.append(float(rank_ic(model.predict(X_va), y_va).mean()))
        trials.append({"params": params, "inner_ic": ics, "mean_ic": float(np.mean(ics))})
    scores = np.array([t["mean_ic"] for t in trials])
    if np.isnan(scores).all():
        raise ValueError("every config has an undefined inner IC (constant predictions?)")
    best = int(np.nanargmax(scores))  # first maximum = first in grid order
    return TuneResult(best_params=trials[best]["params"], trials=trials)
