"""D2 output stage: raw French zips -> processed parquet files."""

import pandas as pd

from edge_autopsy.data.build import build_french
from tests.unit.french_fixtures import default_ind49_zip, ff3_csv, mom_csv, zip_bytes


def _place_raw(raw):
    d = raw / "french"
    d.mkdir(parents=True)
    (d / "49_Industry_Portfolios_daily_CSV.zip").write_bytes(default_ind49_zip())
    rows = {
        "20260928": [0.1, 0, 0, 0.01],
        "20260929": [0.2, 0, 0, 0.01],
        "20260930": [0.3, 0, 0, 0.01],
    }
    (d / "F-F_Research_Data_Factors_daily_CSV.zip").write_bytes(
        zip_bytes("F-F_Research_Data_Factors_daily.csv", ff3_csv(rows))
    )
    (d / "F-F_Momentum_Factor_daily_CSV.zip").write_bytes(
        zip_bytes("F-F_Momentum_Factor_daily.csv", mom_csv({"20260929": 0.5, "20260930": 0.3}))
    )


FILES = {
    "ind49_daily": "49_Industry_Portfolios_daily_CSV.zip",
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}


def test_build_french_writes_parquet(tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    _place_raw(raw)
    out = build_french(raw, proc, FILES)
    ind = pd.read_parquet(out["ff49_vw_daily"])
    fac = pd.read_parquet(out["ff_factors_daily"])
    assert ind.shape == (3, 49)
    assert ind.iloc[2, 0] == 0.003
    assert list(fac.columns) == ["mkt_rf", "smb", "hml", "rf", "mkt", "mom"]
    assert pd.isna(fac.loc["2026-09-28", "mom"])  # momentum starts later: NaN, not ffilled
    assert fac.index.equals(ind.index)
