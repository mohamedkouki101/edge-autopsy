"""M3: Ridge and XGBoost forecasters (PREREGISTRATION §7 grids; U8 determinism)."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.config import load_config
from edge_autopsy.models.linear import RIDGE_GRID, RidgeForecaster, ridge_model
from edge_autopsy.models.xgb import XGB_GRID, XGBForecaster, xgb_model, xgb_refit_params
from edge_autopsy.tuning import grid

CFG = load_config("configs/base.yaml")


def _xy(n_dates=200, n_assets=20, seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [
            pd.bdate_range("2000-01-07", periods=n_dates, freq="W-FRI"),
            [f"A{i}" for i in range(n_assets)],
        ],
        names=["date", "asset"],
    )
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), 3)), index=idx, columns=["f", "g", "h"])
    y = pd.Series(beta * X["f"].to_numpy() + rng.normal(0, 0.3, len(idx)), index=idx)
    return X, y


def test_grids_are_the_preregistered_ones():
    assert grid(RIDGE_GRID) == [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)]
    xg = grid(XGB_GRID)
    assert len(xg) == 12
    assert {c["max_depth"] for c in xg} == {2, 3, 4}
    assert {c["learning_rate"] for c in xg} == {0.01, 0.05}
    assert {c["min_child_weight"] for c in xg} == {50, 200}
    spec = xgb_model(CFG)
    assert spec.configs == xg and spec.name == "xgb"
    assert ridge_model(CFG).configs == grid(RIDGE_GRID)


def test_ridge_learns_the_signal_and_treats_missing_as_the_median_rank():
    X, y = _xy()
    m = RidgeForecaster({"alpha": 1.0}).fit(X, y, None, None)
    assert m.coef_[0] == pytest.approx(0.5, abs=0.05)
    Xn = X.copy()
    Xn.iloc[0] = np.nan
    p = m.predict(Xn)
    assert p.index.equals(X.index) and np.isfinite(p).all()
    assert p.iloc[0] == pytest.approx(m.intercept_)  # all-missing row = all features at rank 0


def test_xgb_early_stops_on_the_validation_fold_and_records_the_round():
    X, y = _xy()
    tr = X.index.get_level_values("date") < "2003-01-01"
    m = XGBForecaster({"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 50}, seed=1)
    m.fit(X[tr], y[tr], X[~tr], y[~tr])
    n = m.fit_info_["best_iteration"]
    assert 0 <= n < 2000
    # the stopping curve is the validation fold's loss, not the training loss
    curve = m.model_.evals_result()["validation_0"]["rmse"]
    rmse_val = float(np.sqrt(np.mean((m.predict(X[~tr]) - y[~tr]) ** 2)))
    assert curve[n] == pytest.approx(rmse_val, rel=1e-5)
    assert int(np.argmin(curve)) == n and len(curve) <= n + 51  # stopped 50 rounds later
    assert m.model_.get_params()["n_estimators"] == 2000  # cap; stopped early at n


def test_xgb_final_refit_uses_the_mean_best_round_of_the_inner_folds():
    trial = {
        "params": {"max_depth": 3},
        "fit_info": [{"best_iteration": 9}, {"best_iteration": 19}, {"best_iteration": 30}],
    }
    # best_iteration is 0-based: trees = best_iteration + 1, averaged -> (10 + 20 + 31) / 3
    assert xgb_refit_params(trial) == {"max_depth": 3, "n_estimators": 20}
    X, y = _xy()
    m = XGBForecaster(
        {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 50, "n_estimators": 20}, seed=1
    )
    m.fit(X, y, None, None)
    assert m.model_.get_booster().num_boosted_rounds() == 20


def test_xgb_without_validation_needs_an_explicit_number_of_trees():
    X, y = _xy()
    with pytest.raises(ValueError, match="n_estimators"):
        XGBForecaster({"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 50}, seed=1).fit(
            X, y, None, None
        )


def test_xgb_is_deterministic_for_a_seed():
    """U8: two fits with the same seed give identical predictions."""
    X, y = _xy()
    tr = X.index.get_level_values("date") < "2003-01-01"
    p = {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 50}
    a = XGBForecaster(p, seed=7).fit(X[tr], y[tr], X[~tr], y[~tr]).predict(X)
    b = XGBForecaster(p, seed=7).fit(X[tr], y[tr], X[~tr], y[~tr]).predict(X)
    pd.testing.assert_series_equal(a, b, check_exact=True)
    c = XGBForecaster(p, seed=8).fit(X[tr], y[tr], X[~tr], y[~tr]).predict(X)
    assert not a.equals(c)  # subsample/colsample 0.8 make the seed matter


def test_tune_keeps_the_fit_info_so_the_refit_can_use_it():
    import datetime as dt

    from edge_autopsy.data.calendar import rebalance_dates
    from edge_autopsy.splits import walk_forward
    from edge_autopsy.tuning import tune

    cal = pd.DatetimeIndex(pd.bdate_range("1990-01-01", "2001-12-31"), name="date")
    weekly = rebalance_dates(cal)
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [weekly, [f"A{i}" for i in range(10)]], names=["date", "asset"]
    )
    X = pd.DataFrame({"f": rng.uniform(-0.5, 0.5, len(idx))}, index=idx)
    y = pd.Series(0.2 * X["f"].to_numpy() + rng.normal(0, 0.3, len(idx)), index=idx)
    sp = CFG.split.model_copy(
        update={
            "dev_test_start": dt.date(2000, 1, 1),
            "dev_test_end": dt.date(2001, 12, 31),
            "holdout_start": dt.date(2002, 1, 2),
            "holdout_end": dt.date(2002, 6, 28),
        }
    )
    fold = walk_forward(weekly, cal, sp)[0]
    res = tune(
        lambda p: XGBForecaster(p, seed=1),
        X,
        y,
        fold,
        cal,
        sp,
        [{"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 50}],
    )
    infos = res.trials[0]["fit_info"]
    assert len(infos) == 3 and all("best_iteration" in i for i in infos)
