"""B1: label at t = return from close t+lag to close t+lag+horizon (days t+2..t+6 by default)."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.labels import build_labels


def _returns():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10), name="date")
    a = [0.01, 0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, 0.01, 0.02]
    b = [0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, np.nan, 0.01, 0.01]
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_hand_example():
    r = _returns()
    y = build_labels(r, horizon=5, lag=1)
    # t = day 0: earns days 2..6 for A: (1-.01)(1.03)(1.00)(1.01)(.98) - 1, by hand
    expected = 0.99 * 1.03 * 1.00 * 1.01 * 0.98 - 1
    assert y.loc[(r.index[0], "A")] == pytest.approx(expected, abs=1e-15)
    assert y.index.names == ["date", "asset"]
    assert y.name == "label"


def test_fill_day_and_signal_day_returns_not_in_label():
    r = _returns()
    y0 = build_labels(r, 5, 1).loc[(r.index[0], "A")]
    r2 = r.copy()
    r2.iloc[0:2, 0] = [0.5, 0.9]  # signal day t and fill day t+1 change: label must not
    assert build_labels(r2, 5, 1).loc[(r.index[0], "A")] == y0
    r3 = r.copy()
    r3.iloc[6, 0] = 0.5  # day t+6 is inside the window: label must change
    assert build_labels(r3, 5, 1).loc[(r.index[0], "A")] != y0


def test_missing_return_in_window_gives_nan():
    r = _returns()
    y = build_labels(r, 5, 1)
    assert np.isnan(y.loc[(r.index[1], "B")])  # window days 3..7 includes NaN at day 7
    assert not np.isnan(y.loc[(r.index[0], "B")])  # days 2..6: complete


def test_window_past_end_gives_nan():
    r = _returns()
    y = build_labels(r, 5, 1)
    assert not np.isnan(y.loc[(r.index[3], "A")])  # days 5..9: last full window
    assert np.isnan(y.loc[(r.index[4], "A")])  # needs day 10: does not exist


def test_lag_zero_window():
    r = _returns()
    y = build_labels(r, horizon=2, lag=0)  # days t+1..t+2
    assert y.loc[(r.index[0], "A")] == pytest.approx(1.02 * 0.99 - 1, abs=1e-15)


# ---- holding-period labels (decision 2026-11-08, Option B): label at rebalance t = return
# from the fill at close t+lag to the next fill at close t'+lag (t' = next rebalance date).


def _returns12():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=12), name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, (12, 2)), index=idx, columns=["A", "B"])


def test_holding_period_hand_example():
    r = _returns12()
    reb = r.index[[0, 4, 9]]  # a 4-day "holiday week" then a 5-day week
    y = build_labels(r, horizon=5, lag=1, rebalance=reb)
    a = 1.0 + r["A"].to_numpy()
    assert y.loc[(reb[0], "A")] == pytest.approx(np.prod(a[2:6]) - 1, abs=1e-15)  # days 2..5
    assert y.loc[(reb[1], "A")] == pytest.approx(np.prod(a[6:11]) - 1, abs=1e-15)  # 6..10
    assert np.isnan(y.loc[(reb[2], "A")])  # last rebalance: no next fill
    assert list(y.index.get_level_values("date").unique()) == list(reb)
    assert y.index.names == ["date", "asset"] and y.name == "label"


def test_holding_period_equals_fixed_label_in_full_weeks():
    r = _returns12()
    reb = r.index[[0, 5]]
    hp = build_labels(r, 5, 1, rebalance=reb).loc[reb[0]]
    fixed = build_labels(r, 5, 1).loc[reb[0]]
    pd.testing.assert_series_equal(hp, fixed)


def test_holding_period_label_equals_what_the_backtest_earns():
    """The point of Option B: label == strategy's compounded gross return over the hold."""
    from edge_autopsy.backtest import run_backtest

    r = _returns12()
    reb = r.index[[0, 4, 9]]
    rf = pd.Series(0.0, index=r.index)
    y = build_labels(r, 5, 1, rebalance=reb)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=reb)
    bt = run_backtest(w, r, rf, lag=1)
    earned = np.prod(1.0 + bt.gross.iloc[2:6].to_numpy()) - 1  # held from fill 1 to fill 5
    assert earned == pytest.approx(y.loc[(reb[0], "A")], abs=1e-15)


def test_holding_period_span_longer_than_horizon_raises():
    r = _returns12()
    with pytest.raises(ValueError, match="horizon"):
        build_labels(r, 5, 1, rebalance=r.index[[0, 6]])  # 6-day hold > horizon 5


def test_holding_period_missing_return_gives_nan():
    r = _returns12()
    r.iloc[3, 1] = np.nan
    y = build_labels(r, 5, 1, rebalance=r.index[[0, 4, 9]])
    assert np.isnan(y.loc[(r.index[0], "B")])
    assert not np.isnan(y.loc[(r.index[0], "A")])


def test_holding_period_next_fill_past_end_gives_nan():
    r = _returns12()
    y = build_labels(r, 5, 1, rebalance=r.index[[6, 11]])  # next fill would be day 12
    assert np.isnan(y.loc[(r.index[6], "A")])


def test_holding_period_rebalance_not_trading_day_raises():
    r = _returns12()
    with pytest.raises(ValueError, match="trading days"):
        build_labels(r, 5, 1, rebalance=pd.DatetimeIndex(["2024-01-06"]))  # a Saturday
