"""D6: validate_panel accepts a clean panel and raises DataError on each corruption."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.errors import DataError
from edge_autopsy.data.validate import PANEL_COLUMNS, validate_panel


def _panel(n_days=300):
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    rows = []
    for uni, assets in [("ff49", ["A", "B"]), ("etf9", ["X"])]:
        for a in assets:
            ret = np.full(n_days, 0.001)
            avail = np.arange(n_days) >= 252  # 253rd non-missing day onward
            rows.append(
                pd.DataFrame(
                    {
                        "date": idx,
                        "asset": a,
                        "universe": uni,
                        "ret": ret,
                        "available": avail,
                        "rf": 0.0001,
                        "mkt": 0.0005,
                        "volume_usd": 1e6 if uni == "etf9" else np.nan,
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)
    df["date"] = df["date"].astype("datetime64[ns]")
    df["asset"] = df["asset"].astype("category")
    df["universe"] = df["universe"].astype("category")
    return df[PANEL_COLUMNS]


KW = dict(
    expected_assets={"ff49": {"A", "B"}, "etf9": {"X"}},
    entry_days=253,
    sample_end=pd.Timestamp("2001-02-23"),
)


def test_clean_panel_passes():
    report = validate_panel(_panel(), **KW)
    assert report["n_rows"] == 900 and report["n_flags"] == 0


def test_large_return_is_flagged_not_raised():
    df = _panel()
    df.loc[5, "ret"] = 0.7
    assert validate_panel(df, **KW)["n_flags"] == 1


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda d: d.assign(rf=d.rf.where(d.index != 3, -0.0001)), "rf"),
        (lambda d: d.assign(ret=d.ret.where(d.index != 290, np.nan)), "available"),
        (lambda d: d.assign(ret=d.ret.where(d.index != 10, np.inf)), "finite"),
        (
            lambda d: d.assign(
                universe=d.universe.cat.add_categories("x").where(d.index != 0, "x")
            ),
            "universe",
        ),
        (lambda d: d.assign(date=d.date.astype(str)), "dtype"),
        (lambda d: d.drop(columns="volume_usd"), "columns"),
        (lambda d: d.assign(available=d.available | (d.index == 0)), "entry"),
        (lambda d: d.assign(mkt=d.mkt.where(d.index != 280, np.nan)), "mkt"),
        (lambda d: d.iloc[::-1].reset_index(drop=True), "increasing"),
        (lambda d: d[d.asset != "B"].reset_index(drop=True), "assets"),
        (lambda d: d[d.date < "2001-01-01"].reset_index(drop=True), "sample_end"),
    ],
)
def test_corruptions_raise(corrupt, match):
    with pytest.raises(DataError, match=match):
        validate_panel(corrupt(_panel()), **KW)
