"""B2: backtest engine timing, drift, turnover, cash (PLAN §3.2, §5.7 K1, §9.1)."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.backtest import run_backtest

IDX = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10), name="date")


def _flat(a=0.0, b=0.0):
    return pd.DataFrame({"A": a, "B": b}, index=IDX, dtype=float)


def _rf(v=0.0):
    return pd.Series(v, index=IDX, dtype=float)


def _w(date, **kw):
    return pd.DataFrame([kw], index=pd.DatetimeIndex([date], name="date"), dtype=float)


def test_backtest_timing():
    """K1: signal at close t, spike on t+1 is NOT earned with lag=1; t+2 spike is."""
    r = _flat()
    r.iloc[1, 0] = 1.00  # +100% on t+1
    r.iloc[2, 0] = 0.10  # +10% on t+2
    res = run_backtest(_w(IDX[0], A=1.0), r, _rf(), lag=1)
    assert res.gross.loc[IDX[1]] == 0.0  # fill day: still cash at its open
    assert res.gross.loc[IDX[2]] == pytest.approx(0.10)
    assert res.fill_dates.tolist() == [IDX[1]]


def test_lag_zero_earns_next_day():
    r = _flat()
    r.iloc[1, 0] = 1.00
    res = run_backtest(_w(IDX[0], A=1.0), r, _rf(), lag=0)
    assert res.gross.loc[IDX[1]] == pytest.approx(1.00)  # filled at close t, earns t+1


def test_signal_day_return_never_earned_even_lag_zero():
    r = _flat()
    r.iloc[0, 0] = 5.0  # return of the signal day itself
    res = run_backtest(_w(IDX[0], A=1.0), r, _rf(), lag=0)
    assert res.gross.loc[IDX[0]] == 0.0


def test_buy_and_hold_closed_form_with_drift():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.02, (10, 2)), index=IDX, columns=["A", "B"])
    res = run_backtest(_w(IDX[0], A=0.3, B=0.7), r, _rf(), lag=1)
    held = r.iloc[2:]
    expected = 0.3 * (1 + held["A"]).prod() + 0.7 * (1 + held["B"]).prod()
    assert (1 + res.gross).prod() == pytest.approx(expected, rel=1e-12)
    # end-of-day weights drift with returns
    w_end = res.weights.iloc[-1]
    va, vb = 0.3 * (1 + held["A"]).prod(), 0.7 * (1 + held["B"]).prod()
    assert w_end["A"] == pytest.approx(va / (va + vb), rel=1e-12)


def test_turnover_measured_against_drifted_weights():
    r = _flat()
    r.iloc[2, 0] = 0.20  # A up 20% on day 2 -> drifted weights 0.6/0.5 of 1.1
    w = pd.concat([_w(IDX[0], A=0.5, B=0.5), _w(IDX[2], A=0.5, B=0.5)])
    res = run_backtest(w, r, _rf(), lag=1)
    drifted_a = 0.6 / 1.1
    traded_second = abs(0.5 - drifted_a) + abs(0.5 - (1 - drifted_a))
    assert res.traded.loc[IDX[1]] == pytest.approx(1.0)  # entry from cash
    assert res.traded.loc[IDX[3]] == pytest.approx(traded_second)
    assert res.turnover.loc[IDX[3]] == pytest.approx(traded_second / 2)  # one-sided
    assert res.turnover.index.tolist() == [IDX[1], IDX[3]]  # fill days only


def test_cash_and_short_proceeds_earn_rf():
    r = _flat(a=0.01, b=-0.01)
    res = run_backtest(_w(IDX[0], A=1.0, B=-1.0), r, _rf(0.0002), lag=1)
    # long A +1%, short B gains +1%, cash (=NAV) earns rf
    assert res.gross.loc[IDX[2]] == pytest.approx(0.01 + 0.01 + 0.0002, rel=1e-12)
    assert res.gross.loc[IDX[0]] == pytest.approx(0.0002)  # all cash before first fill


def test_new_row_replaces_whole_portfolio():
    r = _flat(a=0.01, b=0.02)
    w = pd.concat([_w(IDX[0], A=1.0, B=0.0), _w(IDX[3], B=1.0)])  # second row omits A
    res = run_backtest(w.fillna(np.nan), r, _rf(), lag=1)
    assert res.weights.loc[IDX[4], "A"] == 0.0
    assert res.gross.loc[IDX[5]] == pytest.approx(0.02)


def test_fill_beyond_data_is_ignored():
    res = run_backtest(_w(IDX[-1], A=1.0), _flat(), _rf(), lag=1)
    assert res.fill_dates.empty and (res.gross == 0).all()


def test_missing_return_on_held_asset_counts_as_zero_and_is_reported():
    r = _flat(a=0.01)
    r.iloc[4, 0] = np.nan
    res = run_backtest(_w(IDX[0], A=1.0), r, _rf(), lag=1)
    assert res.gross.loc[IDX[4]] == 0.0
    assert res.n_missing_held == 1


def test_weight_dates_must_be_trading_days():
    with pytest.raises(ValueError, match="trading days"):
        run_backtest(_w(pd.Timestamp("2024-01-06"), A=1.0), _flat(), _rf(), lag=1)  # Saturday


def test_unknown_asset_raises():
    with pytest.raises(ValueError, match="assets"):
        run_backtest(_w(IDX[0], Z=1.0), _flat(), _rf(), lag=1)
