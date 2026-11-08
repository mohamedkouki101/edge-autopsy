"""D3 audit stage on a synthetic panel: files written, flags, sample start honoured."""

import numpy as np
import pandas as pd

from edge_autopsy.data.build import audit_french


def test_audit_writes_report_flags_and_start(tmp_path):
    idx = pd.bdate_range("1960-01-01", "1966-12-30")
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=["A", "B", "C"])
    r.index.name = "date"
    r.loc[:"1963-12-31", "C"] = np.nan  # C enters 1964 -> available ~1965
    r.loc["1961-03-01", "A"] = 0.8  # pre-floor extreme: flagged
    r.loc["1965-03-01", "B"] = -0.6  # in-sample extreme: flagged
    r.to_parquet(tmp_path / "ff49_vw_daily.parquet")
    out = audit_french(tmp_path, entry_days=253, min_assets=2, floor=pd.Timestamp("1963-07-01"))
    assert out["sample_start"] == pd.Timestamp("1963-07-05")
    flags = pd.read_csv(tmp_path / "flags.csv")
    assert len(flags) == 2 and set(flags["asset"]) == {"A", "B"}
    report = (tmp_path / "data_audit.md").read_text(encoding="utf-8")
    assert "1963-07-05" in report and "1965-03-01" in report
    by_year = pd.read_csv(tmp_path / "nonmissing_by_year.csv", index_col=0)
    assert by_year.loc[1963, "C"] == 0 and by_year.loc[1964, "C"] > 250
    av = pd.read_parquet(tmp_path / "ff49_available.parquet")
    assert av.shape == r.shape and av.dtypes.unique().tolist() == [np.dtype(bool)]
