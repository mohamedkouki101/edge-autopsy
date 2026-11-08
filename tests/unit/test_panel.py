"""D6: build_panel assembles the long schema from processed parts; revision diff."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.build import build_panel, french_revision_diff
from edge_autopsy.data.validate import PANEL_COLUMNS
from tests.unit.french_fixtures import ind49_csv, zip_bytes


def _processed(tmp_path, n=300):
    idx = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=n), name="date")
    ind = pd.DataFrame(0.001, index=idx, columns=[f"I{i}" for i in range(49)])
    ind.iloc[:10, 0] = np.nan
    ind.to_parquet(tmp_path / "ff49_vw_daily.parquet")
    fac = pd.DataFrame(
        {"mkt_rf": 0.0004, "smb": 0.0, "hml": 0.0, "rf": 0.0001, "mkt": 0.0005, "mom": 0.0},
        index=idx,
    )
    fac.to_parquet(tmp_path / "ff_factors_daily.parquet")
    etf = []
    for t in ["E1", "E2", "SPY"]:
        r = np.full(n, 0.002 if t == "SPY" else 0.001)
        r[0] = np.nan
        etf.append(pd.DataFrame({"date": idx, "asset": t, "ret": r, "volume_usd": 5e6}))
    pd.concat(etf).to_parquet(tmp_path / "etf9_daily.parquet", index=False)
    return idx


def test_build_panel_schema_and_rules(tmp_path):
    idx = _processed(tmp_path)
    df = pd.read_parquet(
        build_panel(tmp_path, ["E1", "E2"], "SPY", entry_days=253, sample_end=idx[-1])
    )
    assert list(df.columns) == PANEL_COLUMNS
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert isinstance(df["asset"].dtype, pd.CategoricalDtype)
    assert set(df["universe"]) == {"ff49", "etf9"}
    assert "SPY" not in set(df["asset"])  # benchmark is the etf9 `mkt`, not an asset
    e1 = df[df.asset == "E1"].set_index("date")
    assert e1["mkt"].iloc[5] == 0.002 and e1["rf"].iloc[5] == 0.0001
    assert not e1["available"].iloc[252] and e1["available"].iloc[253]  # 1st ret NaN
    i0 = df[df.asset == "I0"].set_index("date")
    assert i0["available"].idxmax() == idx[262]  # 10 leading NaNs
    assert i0["volume_usd"].isna().all()
    assert df.groupby("universe", observed=True)["asset"].nunique().to_dict() == {
        "etf9": 2,
        "ff49": 49,
    }


def test_build_panel_rejects_etf_date_without_rf(tmp_path):
    idx = _processed(tmp_path)
    fac = pd.read_parquet(tmp_path / "ff_factors_daily.parquet").drop(idx[100])
    fac.to_parquet(tmp_path / "ff_factors_daily.parquet")
    from edge_autopsy.data.errors import DataError

    with pytest.raises(DataError, match="rf"):
        build_panel(tmp_path, ["E1", "E2"], "SPY", entry_days=253, sample_end=idx[-1])


def test_french_revision_diff(tmp_path):
    base = {"20260928": [0.1] * 49, "20260929": [0.2] * 49}
    new = {"20260928": [0.1] * 49, "20260929": [0.2] * 48 + [0.5], "20260930": [0.3] * 49}
    old_zip, new_zip = tmp_path / "old.zip", tmp_path / "new.zip"
    old_zip.write_bytes(zip_bytes("x.csv", ind49_csv(base, base)))
    new_zip.write_bytes(zip_bytes("x.csv", ind49_csv(new, new)))
    d = french_revision_diff(old_zip, new_zip)
    assert d["changed_cells"] == 1
    assert d["max_abs_change"] == pytest.approx(0.003)
    assert d["new_dates"] == 1 and d["dropped_dates"] == 0
