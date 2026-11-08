"""D2 acceptance: the VW daily block is selected, missing codes become NaN, % -> decimal."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.data.clean import (
    DataError,
    clean_french_ff3,
    clean_french_ind49,
    clean_french_mom,
    parse_french_block,
)
from tests.unit.french_fixtures import IND_NAMES, ff3_csv, ind49_csv, mom_csv


def _vw_ew():
    vw = {
        "19260701": [0.56, -99.99] + [0.10] * 47,
        "19260702": [-999.0, 1.50] + [0.20] * 47,
        "19260706": [-0.07, 0.00] + [0.30] * 47,
    }
    ew = {d: [9.0] * 49 for d in vw}  # EW values are distinct: picking EW would be caught
    return vw, ew


def test_clean_missing_codes():
    df = clean_french_ind49(ind49_csv(*_vw_ew()))
    assert np.isnan(df.iloc[0, 1])  # -99.99 -> NaN
    assert np.isnan(df.iloc[1, 0])  # -999 -> NaN
    assert df.iloc[0, 0] == pytest.approx(0.0056)  # percent -> decimal
    assert df.iloc[2, 1] == 0.0  # a true zero return is kept, not treated as missing
    assert df.isna().sum().sum() == 2


def test_vw_block_selected():
    df = clean_french_ind49(ind49_csv(*_vw_ew()))
    assert df.shape == (3, 49)
    assert list(df.columns) == IND_NAMES
    assert df.index[0] == pd.Timestamp("1926-07-01")  # first VW date from the file
    assert df.index.name == "date"
    assert (df.iloc[:, 2:] < 0.05).all().all()  # no EW value (9% -> 0.09) leaked in
    assert df.dtypes.unique().tolist() == [np.dtype("float64")]


def test_missing_block_raises():
    text = ind49_csv(*_vw_ew()).replace("Average Value Weighted Returns", "Something Else")
    with pytest.raises(DataError, match="Average Value Weighted Returns"):
        clean_french_ind49(text)


def test_wrong_column_count_raises():
    text = ind49_csv(*_vw_ew()).replace(",Ind49", "", 1)  # VW header loses one industry
    with pytest.raises(DataError, match="columns"):
        clean_french_ind49(text)


def test_dates_must_be_strictly_increasing():
    vw = {"19260702": [0.1] * 49, "19260701": [0.2] * 49}
    with pytest.raises(DataError, match="increasing"):
        clean_french_ind49(ind49_csv(vw, vw))


def test_duplicate_dates_raise():
    text = "\n".join([",A,B", "19260701, 1.0, 2.0", "19260701, 1.0, 2.0", ""])
    with pytest.raises(DataError, match="increasing"):
        parse_french_block(text, header_line=0, expected_cols=2)


def test_ff3_columns_and_market():
    df = clean_french_ff3(ff3_csv({"20260930": [-0.26, 0.32, -0.88, 0.01]}))
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rf", "mkt"]
    row = df.loc["2026-09-30"]
    assert row["rf"] == pytest.approx(0.0001)
    assert row["mkt"] == pytest.approx(-0.0026 + 0.0001)  # mkt = (Mkt-RF) + RF


def test_mom_column():
    df = clean_french_mom(mom_csv({"19261103": 0.55, "19261104": -99.99}))
    assert list(df.columns) == ["mom"]
    assert df.iloc[0, 0] == pytest.approx(0.0055)
    assert np.isnan(df.iloc[1, 0])
