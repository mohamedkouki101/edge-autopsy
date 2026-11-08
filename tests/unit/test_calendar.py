"""rebalance_dates: last trading day of each week (pulled forward from B1 for D3)."""

import pandas as pd

from edge_autopsy.data.calendar import rebalance_dates


def test_last_trading_day_of_each_week():
    idx = pd.bdate_range("2024-03-18", "2024-04-05")  # Mon..Fri x 3 weeks
    idx = idx.drop(pd.Timestamp("2024-03-29"))  # Good Friday 2024: market closed
    out = rebalance_dates(idx)
    assert list(out) == [
        pd.Timestamp("2024-03-22"),
        pd.Timestamp("2024-03-28"),  # Thursday, because Friday was a holiday
        pd.Timestamp("2024-04-05"),
    ]


def test_dates_are_subset_of_index_and_sorted():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    out = rebalance_dates(idx)
    assert out.isin(idx).all()
    assert out.is_monotonic_increasing and out.is_unique
    assert len(out) == 53  # 2020-01-03 ... 2020-12-31 (Thursday; partial last week kept)
