"""V4 / K5: nested tuning only ever sees the outer training dates (PLAN §5.5).

A spy model records every (date) it is fitted or scored on. The outer test block must never
appear, and inside each inner fold the fit labels must end before the validation block starts.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.config import SplitCfg
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.splits import Fold, label_end, walk_forward
from edge_autopsy.tuning import MAX_CONFIGS, grid, tune


def _cfg() -> SplitCfg:
    return SplitCfg(
        scheme="expanding",
        dev_test_start=dt.date(2000, 1, 1),
        dev_test_end=dt.date(2001, 12, 31),
        holdout_start=dt.date(2002, 1, 2),
        holdout_end=dt.date(2002, 12, 31),
        min_train_years=4,
        rolling_years=4,
        refit_every=52,
        purge_days=6,
        embargo_weeks=1,
        inner_folds=3,
        inner_fold_years=1,
    )


@pytest.fixture(scope="module")
def data():
    """Weekly samples 1994-2001, 12 assets; y = 0.3 * signal + noise."""
    cal = pd.DatetimeIndex(pd.bdate_range("1994-01-03", "2001-12-31"), name="date")
    weekly = rebalance_dates(cal)
    idx = pd.MultiIndex.from_product(
        [weekly, [f"A{i}" for i in range(12)]], names=["date", "asset"]
    )
    rng = np.random.default_rng(0)
    signal = rng.normal(size=len(idx))
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(0.3 * signal + rng.normal(size=len(idx)), index=idx, name="label")
    folds = walk_forward(weekly, cal, _cfg())
    return cal, X, y, folds


class Spy:
    """Predicts one feature column; records the dates it is fitted and scored on."""

    log: list = []

    def __init__(self, params):
        self.col = params["feature"]

    def fit(self, X, y, X_val, y_val):
        assert X.index.equals(y.index) and X_val.index.equals(y_val.index)
        Spy.log.append(
            (
                X.index.get_level_values("date").unique(),
                X_val.index.get_level_values("date").unique(),
            )
        )
        return self

    def predict(self, X):
        return X[self.col]


def test_tuning_never_sees_test(data):
    """V4 acceptance (K5): no fit or validation row comes from the outer test block."""
    cal, X, y, folds = data
    for outer in folds:
        Spy.log = []
        tune(Spy, X, y, outer, cal, _cfg(), grid({"feature": ["signal", "noise"]}))
        assert len(Spy.log) == 2 * 3  # 2 configs x 3 inner folds
        for fit_dates, val_dates in Spy.log:
            assert fit_dates.isin(outer.train_dates).all()
            assert val_dates.isin(outer.train_dates).all()
            assert not fit_dates.isin(outer.test_dates).any()
            assert not val_dates.isin(outer.test_dates).any()
            assert label_end(fit_dates, cal, 6).max() < val_dates.min()  # purged inner fold
            assert label_end(val_dates, cal, 6).max() < outer.test_dates.min()


def test_tune_picks_the_best_mean_inner_ic_and_logs_every_trial(data):
    cal, X, y, folds = data
    res = tune(Spy, X, y, folds[0], cal, _cfg(), grid({"feature": ["noise", "signal"]}))
    assert res.best_params == {"feature": "signal"}
    assert [t["params"] for t in res.trials] == [{"feature": "noise"}, {"feature": "signal"}]
    for t in res.trials:
        assert len(t["inner_ic"]) == 3
        assert t["mean_ic"] == pytest.approx(np.mean(t["inner_ic"]))
    sig = res.trials[1]
    assert sig["mean_ic"] > 0.2 and abs(res.trials[0]["mean_ic"]) < 0.05


def test_tune_rejects_a_fold_whose_train_overlaps_its_test(data):
    """Leak lab L4 (tune on test) has to be built on purpose; tune() refuses it."""
    cal, X, y, folds = data
    f = folds[0]
    bad = Fold(
        f.train_dates.union(f.test_dates[:3]), f.test_dates, f.purged_dates, f.embargoed_dates
    )
    with pytest.raises(ValueError, match="test"):
        tune(Spy, X, y, bad, cal, _cfg(), grid({"feature": ["signal"]}))


def test_tune_ignores_rows_outside_train_even_if_passed_in(data):
    """Poison the outer test rows: results must not move."""
    cal, X, y, folds = data
    f = folds[0]
    configs = grid({"feature": ["signal", "noise"]})
    clean = tune(Spy, X, y, f, cal, _cfg(), configs)
    Xp, yp = X.copy(), y.copy()
    test_rows = X.index.get_level_values("date").isin(f.test_dates)
    Xp.loc[test_rows, "noise"] = yp[test_rows] * 100  # noise would look perfect on test
    poisoned = tune(Spy, Xp, yp, f, cal, _cfg(), configs)
    assert poisoned.trials == clean.trials and poisoned.best_params == clean.best_params


def test_budget_is_capped_at_24_configs(data):
    cal, X, y, folds = data
    assert MAX_CONFIGS == 24  # PLAN §6 fairness rule
    too_many = [{"feature": "signal", "k": k} for k in range(25)]
    with pytest.raises(ValueError, match="24"):
        tune(Spy, X, y, folds[0], cal, _cfg(), too_many)


def test_ties_go_to_the_first_config_and_nan_never_wins(data):
    cal, X, y, folds = data
    X2 = X.assign(const=0.0, signal2=X["signal"])
    configs = grid({"feature": ["const", "signal", "signal2"]})
    res = tune(Spy, X2, y, folds[0], cal, _cfg(), configs)
    assert np.isnan(res.trials[0]["mean_ic"])  # constant prediction: IC undefined
    assert res.best_params == {"feature": "signal"}  # tie with signal2 -> first in grid order


def test_grid_is_the_cartesian_product_in_declared_order():
    assert grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
