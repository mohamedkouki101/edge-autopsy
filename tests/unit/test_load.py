"""B5: the dev loader can never return hold-out dates."""

import pandas as pd
import pytest

from edge_autopsy.data.load import HoldoutSealedError, load_universe
from edge_autopsy.data.validate import PANEL_COLUMNS


def _panel(tmp_path):
    idx = pd.bdate_range("2016-12-26", "2017-01-06")
    df = pd.DataFrame(
        {
            "date": idx,
            "asset": "A",
            "universe": "ff49",
            "ret": 0.01,
            "available": True,
            "rf": 0.0001,
            "mkt": 0.002,
            "volume_usd": float("nan"),
        }
    )
    df["date"] = df["date"].astype("datetime64[ns]")
    for c in ("asset", "universe"):
        df[c] = df[c].astype("category")
    p = tmp_path / "panel.parquet"
    df[PANEL_COLUMNS].to_parquet(p, index=False)
    return p


def test_dev_loader_truncates_at_end(tmp_path):
    u = load_universe(
        _panel(tmp_path),
        "ff49",
        end=pd.Timestamp("2016-12-31"),
        holdout_start=pd.Timestamp("2017-01-02"),
    )
    for frame in (u.returns, u.available, u.rf, u.mkt):
        assert frame.index.max() == pd.Timestamp("2016-12-30")
    assert u.returns.loc["2016-12-30", "A"] == 0.01


def test_loader_refuses_holdout_dates(tmp_path):
    with pytest.raises(HoldoutSealedError):
        load_universe(
            _panel(tmp_path),
            "ff49",
            end=pd.Timestamp("2017-01-03"),
            holdout_start=pd.Timestamp("2017-01-02"),
        )
