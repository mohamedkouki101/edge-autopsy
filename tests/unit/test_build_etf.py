"""D5 output stage: raw ETF bars -> etf9_daily.parquet (long)."""

import pandas as pd
import pytest

from edge_autopsy.data.build import build_etf


def _write(raw, t, close, div, vol):
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=len(close)), name="date")
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": vol,
            "Dividends": div,
            "Stock Splits": 0.0,
            "Capital Gains": 0.0,
        },
        index=idx,
    )
    (raw / "etf").mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw / "etf" / f"{t}.parquet")


def test_build_etf_long_frame(tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    _write(raw, "AAA", [10.0, 10.5, 10.0], [0.0, 0.0, 0.5], [100, 200, 300])
    _write(raw, "SPY", [100.0, 101.0, 102.0], [0.0, 0.0, 0.0], [1, 1, 1])
    path = build_etf(raw, proc, ["AAA"], "SPY")
    df = pd.read_parquet(path)
    assert list(df.columns) == ["date", "asset", "ret", "volume_usd"]
    a = df[df.asset == "AAA"].set_index("date")
    assert a["ret"].iloc[2] == pytest.approx((10.0 + 0.5) / 10.5 - 1)
    assert a["volume_usd"].iloc[1] == pytest.approx(10.5 * 200)
    assert set(df.asset) == {"AAA", "SPY"}
