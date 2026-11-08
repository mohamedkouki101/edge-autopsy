"""D4: ETF download (yfinance injected), manifest entries, first-date check."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.download import (
    ETF_COLUMNS,
    check_etf_first_dates,
    download_etfs,
    load_etf_raw,
    load_manifest,
)
from edge_autopsy.data.errors import DataError


def _fake_history(first: str):
    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        idx = pd.bdate_range(max(first, start), "1999-02-26", tz="America/New_York")
        n = len(idx)
        return pd.DataFrame(
            {
                "Open": 10.0,
                "High": 10.5,
                "Low": 9.5,
                "Close": np.linspace(10, 11, n),
                "Adj Close": np.linspace(8, 9, n),
                "Volume": 1000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
                "Capital Gains": 0.0,
            },
            index=idx.rename("Date"),
        )

    return fetch


def test_download_writes_parquet_and_manifest(tmp_path):
    paths = download_etfs(
        ["AAA", "BBB"], "1998-12-01", "1999-03-01", tmp_path, fetch=_fake_history("1998-12-22")
    )
    assert set(paths) == {"AAA", "BBB"}
    df = load_etf_raw(tmp_path, "AAA")
    assert list(df.columns) == ETF_COLUMNS
    assert df.index.tz is None and df.index.name == "date"  # exchange-local dates, tz dropped
    assert df.index[0] == pd.Timestamp("1998-12-22")
    m = load_manifest(tmp_path)["files"]["etf_AAA"]
    assert m["first_date"] == "1998-12-22" and len(m["sha256"]) == 64
    assert m["file"] == "etf/AAA.parquet"


def test_first_dates_ok_and_late_ticker_raises(tmp_path):
    download_etfs(["OLD"], "1998-12-01", "1999-03-01", tmp_path, fetch=_fake_history("1998-12-22"))
    download_etfs(["NEW"], "1998-12-01", "1999-03-01", tmp_path, fetch=_fake_history("1999-01-15"))
    check_etf_first_dates(tmp_path, ["OLD"], pd.Timestamp("1999-01-04"))
    with pytest.raises(DataError, match="NEW"):
        check_etf_first_dates(tmp_path, ["OLD", "NEW"], pd.Timestamp("1999-01-04"))


def test_offline_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="XLK"):
        download_etfs(["XLK"], "1998-12-01", "1999-03-01", tmp_path, offline=True)


def test_manual_csv_is_accepted(tmp_path):
    df = _fake_history("1998-12-22")("CCC", "1998-12-01", "1999-03-01")
    (tmp_path / "etf").mkdir()
    df.tz_localize(None).to_csv(tmp_path / "etf" / "CCC.csv")
    download_etfs(["CCC"], "1998-12-01", "1999-03-01", tmp_path, offline=True)
    assert load_etf_raw(tmp_path, "CCC").index[0] == pd.Timestamp("1998-12-22")


@pytest.mark.data
def test_etf_first_dates():
    """Real data: each original Select Sector SPDR (and SPY) trades by 1999-01-04 (N18)."""
    from edge_autopsy.config import load_config

    raw = Path("data/raw")
    cfg = load_config("configs/base.yaml")
    tickers = [*cfg.data.etf_tickers, cfg.data.etf_benchmark]
    if not all((raw / "etf" / f"{t}.parquet").exists() for t in tickers):
        pytest.skip("ETF raw files not downloaded (run tasks.py data)")
    check_etf_first_dates(raw, tickers, pd.Timestamp(cfg.data.etf_first_date_max))
