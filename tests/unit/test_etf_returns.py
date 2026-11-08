"""D5: ETF total returns = (Close + Dividend + CapGain) / prev Close - 1 (PLAN §5.3 rule 7)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.clean import etf_total_returns


def _bars(close, div, cg=None):
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=len(close)), name="date")
    return pd.DataFrame(
        {"Close": close, "Dividends": div, "Capital Gains": cg or [0.0] * len(close)}, index=idx
    )


def test_hand_example_dividend_on_exdate():
    r = etf_total_returns(_bars([100.0, 99.5, 101.0], [0.0, 1.0, 0.0]))
    assert np.isnan(r.iloc[0])  # no previous close: NaN, never 0
    assert r.iloc[1] == pytest.approx(0.005, abs=1e-12)  # (99.5 + 1) / 100 - 1
    assert r.iloc[2] == pytest.approx(101.0 / 99.5 - 1, abs=1e-12)  # dividend not repeated


def test_capital_gain_distribution_counts_as_income():
    r = etf_total_returns(_bars([50.0, 49.0], [0.0, 0.0], [0.0, 1.0]))
    assert r.iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_missing_close_gives_nan_not_fill():
    r = etf_total_returns(_bars([100.0, np.nan, 102.0], [0.0, 0.0, 0.0]))
    assert np.isnan(r.iloc[1]) and np.isnan(r.iloc[2])


@pytest.mark.data
def test_etf_total_return_on_exdate():
    """Real XLK data, 3 ex-dates: pipeline return == (close + div)/prev close within 1e-6.
    Also documents U5: yfinance Adj Close returns differ by more than 1e-6 on ex-dates."""
    from edge_autopsy.data.download import load_etf_raw

    raw = Path("data/raw")
    if not (raw / "etf" / "XLK.parquet").exists():
        pytest.skip("ETF raw files not downloaded")
    h = load_etf_raw(raw, "XLK")
    r = etf_total_returns(h)
    adj = h["Adj Close"].pct_change()
    for d in ["2006-09-15", "2016-03-18", "2026-09-21"]:
        d = pd.Timestamp(d)
        i = h.index.get_loc(d)
        assert h["Dividends"].iloc[i] > 0, d
        prev, close, div = h["Close"].iloc[i - 1], h["Close"].iloc[i], h["Dividends"].iloc[i]
        assert abs(r.loc[d] - ((close + div) / prev - 1)) < 1e-6
        assert abs(adj.loc[d] - close / (prev - div) + 1) < 5e-6  # Yahoo's convention
    ex = h.index[h["Dividends"] > 0]
    assert (r[ex] - adj[ex]).abs().max() > 1e-5  # so Adj Close fails the 1e-6 test (U5)
    assert r.loc["2025-12-05"] == pytest.approx(0.00732, abs=1e-4)  # 2:1 split day: no jump
