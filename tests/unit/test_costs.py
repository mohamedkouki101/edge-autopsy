"""B3: cost model hand examples (PLAN §5.6, §9.1) and engine integration."""

import pandas as pd
import pytest

from edge_autopsy.backtest import run_backtest
from edge_autopsy.config import load_config
from edge_autopsy.costs import CostModel

IDX = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=260), name="date")


def test_plan_hand_example_10bps():
    c = CostModel(spread_bps=5, slippage_bps=5, borrow_bps=25)
    assert c.per_side_bps == 10
    # cash -> 0.5/0.5: sum|dw| = 1.0 -> 2 x 0.5 x 10 bps = 10 bps
    assert c.trade_cost(abs(0.5 - 0) + abs(0.5 - 0)) == pytest.approx(0.0010, abs=1e-15)
    assert c.trade_cost(0.0) == 0.0


def test_borrow_over_252_days_equals_annual_fee():
    c = CostModel(spread_bps=0, slippage_bps=0, borrow_bps=25)
    assert sum(c.borrow_cost(1.0) for _ in range(252)) == pytest.approx(0.0025, abs=1e-15)
    assert c.borrow_cost(0.0) == 0.0


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        CostModel(spread_bps=-1, slippage_bps=0, borrow_bps=0)
    c = CostModel(5, 5, 25)
    with pytest.raises(ValueError):
        c.trade_cost(-0.1)
    with pytest.raises(ValueError):
        c.borrow_cost(-0.1)


def test_from_config_and_grid_override():
    c = CostModel.from_cfg(load_config("configs/base.yaml").costs)
    assert (c.per_side_bps, c.borrow_bps) == (10, 25)
    g = c.with_per_side(50)
    assert g.per_side_bps == 50 and g.borrow_bps == 25


def test_engine_charges_cost_on_fill_day():
    r = pd.DataFrame({"A": 0.0, "B": 0.0}, index=IDX)
    rf = pd.Series(0.0, index=IDX)
    w = pd.DataFrame([{"A": 0.5, "B": 0.5}], index=IDX[[0]])
    res = run_backtest(w, r, rf, cost=CostModel(5, 5, 25), lag=1)
    assert res.cost.loc[IDX[1]] == pytest.approx(0.0010)
    assert res.cost.drop(IDX[1]).abs().sum() == 0.0
    assert res.net.loc[IDX[1]] == pytest.approx(-0.0010)
    assert res.borrow.sum() == 0.0  # long-only pays no borrow


def test_engine_accrues_borrow_daily_on_shorts():
    r = pd.DataFrame({"A": 0.0, "B": 0.0}, index=IDX)
    rf = pd.Series(0.0, index=IDX)
    w = pd.DataFrame([{"A": 1.0, "B": -1.0}], index=IDX[[0]])
    res = run_backtest(w, r, rf, cost=CostModel(0, 0, 25), lag=1)
    held_days = len(IDX) - 2  # short held from close of day 1; accrues days 2..259
    assert res.borrow.sum() == pytest.approx(0.0025 * held_days / 252, rel=1e-12)
    assert res.borrow.loc[IDX[1]] == 0.0  # not yet short during the fill day
