"""D3: availability (253-day entry rule), sample start, extremes, base.yaml update."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.universe import (
    availability,
    first_sample_date,
    top_abs_returns,
    write_sample_start,
)


def _frame(n=300):
    idx = pd.bdate_range("2000-01-03", periods=n)
    a = pd.Series(0.001, index=idx)
    a.iloc[:2] = np.nan  # A enters 2 days late
    a.iloc[260] = np.nan  # and has one missing day later
    b = pd.Series(0.002, index=idx)
    return pd.DataFrame({"A": a, "B": b})


def test_available_from_253rd_non_missing_day():
    df = _frame()
    av = availability(df, entry_days=253)
    assert av["B"].idxmax() == df.index[252]  # B's 253rd day is position 252
    assert not av["B"].iloc[251]
    assert av["A"].idxmax() == df.index[254]  # 2 leading NaNs shift entry by 2
    assert not av["A"].iloc[260]  # a missing return is never available (no ffill)
    assert av["A"].iloc[261]  # count is cumulative: back the next day
    assert av.dtypes.unique().tolist() == [np.dtype(bool)]


def test_availability_uses_only_past_data():
    df = _frame()
    t = 255
    base = availability(df, 253).iloc[: t + 1]
    mutated = df.copy()
    mutated.iloc[t + 1 :] = np.nan  # wipe the future
    assert availability(mutated, 253).iloc[: t + 1].equals(base)


def test_first_sample_date_is_a_rebalance_date():
    idx = pd.bdate_range("2000-01-03", periods=20)
    av = pd.DataFrame(False, index=idx, columns=list("abc"))
    av.loc[idx[7] :, ["a", "b"]] = True  # 2 available from Wed 2000-01-12
    d = first_sample_date(av, min_assets=2)
    assert d == pd.Timestamp("2000-01-14")  # that week's last trading day (Friday)
    with pytest.raises(ValueError):
        first_sample_date(av, min_assets=3)


def test_top_abs_returns_sorted_by_magnitude():
    idx = pd.bdate_range("2000-01-03", periods=3)
    df = pd.DataFrame({"x": [0.01, -0.30, 0.02], "y": [0.20, np.nan, -0.05]}, index=idx)
    top = top_abs_returns(df, 2)
    assert list(top["ret"]) == [-0.30, 0.20]
    assert list(top["asset"]) == ["x", "y"]


def test_write_sample_start_keeps_comments(tmp_path):
    p = tmp_path / "base.yaml"
    p.write_text("data:\n  sample_start: null      # written by D3\n  sample_end: 2026-09-30\n")
    write_sample_start(p, pd.Timestamp("1963-07-05"))
    text = p.read_text()
    assert "  sample_start: 1963-07-05      # written by D3\n" in text
    assert "sample_end: 2026-09-30" in text


def test_first_sample_date_respects_floor():
    idx = pd.bdate_range("2000-01-03", periods=20)
    av = pd.DataFrame(True, index=idx, columns=list("ab"))
    assert first_sample_date(av, min_assets=2) == pd.Timestamp("2000-01-07")
    floor = pd.Timestamp("2000-01-12")  # a Wednesday
    assert first_sample_date(av, min_assets=2, floor=floor) == pd.Timestamp("2000-01-14")
