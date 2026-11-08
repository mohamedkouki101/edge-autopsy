"""Parse Ken French CSV files (PLAN §5.3 rules 1-3, 5; task D2).

Each function takes the CSV *text* (what's inside the zip) and returns a wide float64 frame
indexed by `date`. Rules:
1. Select a block by its header text; assert the column count.
2. -99.99 and -999 -> NaN (French's missing codes), then divide by 100 (percent -> decimal).
3. Dates YYYYMMDD -> datetime; strictly increasing and unique, else DataError.
5. No forward-filling anywhere.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from edge_autopsy.data.errors import DataError

__all__ = [
    "DataError",
    "MISSING_CODES",
    "clean_french_ff3",
    "clean_french_ind49",
    "clean_french_mom",
    "etf_total_returns",
    "parse_french_block",
    "read_zip_text",
]

MISSING_CODES = (-99.99, -999.0)
VW_DAILY_HEADER = "Average Value Weighted Returns -- Daily"
_DATE_ROW = re.compile(r"^\s*\d{8}\s*,")


def read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        (name,) = zf.namelist()
        return zf.read(name).decode("latin-1")


def _find_line(lines: list[str], needle: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise DataError(f"expected exactly one line containing {needle!r}, found {len(hits)}")
    return hits[0]


def parse_french_block(text: str, header_line: int, expected_cols: int) -> pd.DataFrame:
    """Parse the data rows that follow the column-header line `header_line`."""
    lines = text.splitlines()
    header = [c.strip() for c in lines[header_line].split(",")][1:]
    if len(header) != expected_cols:
        raise DataError(f"expected {expected_cols} columns, header has {len(header)}: {header}")
    body = []
    for line in lines[header_line + 1 :]:
        if not _DATE_ROW.match(line):
            break  # block ends at the first non-data line (blank line or next title)
        body.append(line)
    if not body:
        raise DataError("block has no data rows")
    df = pd.read_csv(
        io.StringIO("\n".join(body)),
        header=None,
        names=["date", *header],
        dtype={"date": str},
        skipinitialspace=True,
    )
    if df.shape[1] != expected_cols + 1:
        raise DataError(f"expected {expected_cols} data columns, parsed {df.shape[1] - 1}")
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")
    df = df.set_index("date").astype("float64")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise DataError("dates must be strictly increasing and unique")
    vals = df.to_numpy()
    missing = np.zeros(vals.shape, dtype=bool)
    for code in MISSING_CODES:
        # tolerance, not ==: never depend on the float parser being correctly rounded
        missing |= np.isclose(vals, code, rtol=0.0, atol=1e-9)
    df = df.mask(missing)
    return df / 100.0


def clean_french_ind49(text: str) -> pd.DataFrame:
    """Value-weighted daily block of '49 Industry Portfolios [Daily]' (49 columns)."""
    lines = text.splitlines()
    title = _find_line(lines, VW_DAILY_HEADER)
    return parse_french_block(text, header_line=title + 1, expected_cols=49)


def clean_french_ff3(text: str) -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RF as decimals, plus mkt = (Mkt-RF) + RF."""
    lines = text.splitlines()
    head = _find_line(lines, ",Mkt-RF,SMB,HML,RF")
    df = parse_french_block(text, header_line=head, expected_cols=4)
    df.columns = ["mkt_rf", "smb", "hml", "rf"]
    df["mkt"] = df["mkt_rf"] + df["rf"]
    return df


def clean_french_mom(text: str) -> pd.DataFrame:
    """Daily momentum factor (UMD) as a decimal."""
    lines = text.splitlines()
    head = _find_line(lines, ",Mom")
    df = parse_french_block(text, header_line=head, expected_cols=1)
    df.columns = ["mom"]
    return df


def etf_total_returns(bars: pd.DataFrame) -> pd.Series:
    """Daily total return (Close_t + Div_t + CapGain_t) / Close_{t-1} - 1 (PLAN §5.3 rule 7).

    yfinance `Close` (auto_adjust=False) is split-adjusted but not dividend-adjusted. We do not
    use `Adj Close`: Yahoo's multiplicative adjustment gives Close_t / (Close_{t-1} - Div) - 1
    on ex-dates, which differs from this CRSP-style definition by up to ~1e-4 (U5, D5 test).
    A missing close gives NaN for that day and the next (no filling).
    """
    close = bars["Close"]
    income = bars["Dividends"].fillna(0.0)
    if "Capital Gains" in bars:
        income = income + bars["Capital Gains"].fillna(0.0)
    r = (close + income) / close.shift(1) - 1.0
    return r.rename("ret")
