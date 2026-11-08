"""B6: weights from scores (PLAN §5.6)."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.backtest import run_backtest
from edge_autopsy.config import PortfolioCfg
from edge_autopsy.portfolio import construct_weights

ASSETS = [f"I{i:02d}" for i in range(49)]


def _cfg(**kw):
    base = dict(kind="long_only", quantile=0.2, weighting="equal", band=None)
    base.update(kw)
    return PortfolioCfg(**base)


def _scores(n_dates=1, seed=0, cols=ASSETS):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-05", periods=n_dates, freq="W-FRI"), name="date")
    return pd.DataFrame(rng.normal(size=(n_dates, len(cols))), index=idx, columns=cols)


def _avail(s):
    return pd.DataFrame(True, index=s.index, columns=s.columns)


def test_long_only_top_quintile_equal_weight():
    s = _scores()
    w = construct_weights(s, _avail(s), _cfg())
    top = s.iloc[0].sort_values(ascending=False).index[:10]
    assert w.iloc[0][top].to_numpy() == pytest.approx([0.1] * 10)
    assert w.iloc[0].drop(top).abs().sum() == 0.0
    assert w.iloc[0].sum() == pytest.approx(1.0)


def test_nine_etfs_gives_two():
    cols = list("ABCDEFGHI")
    s = _scores(cols=cols)
    w = construct_weights(s, _avail(s), _cfg())
    assert (w.iloc[0] > 0).sum() == 2 and w.iloc[0].sum() == pytest.approx(1.0)


def test_unavailable_or_nan_score_never_held():
    s = _scores()
    best = s.iloc[0].idxmax()
    av = _avail(s)
    av.loc[:, best] = False
    s.iloc[0, 1] = np.nan
    w = construct_weights(s, av, _cfg())
    assert w.iloc[0][best] == 0.0 and w.iloc[0].iloc[1] == 0.0
    assert (w.iloc[0] > 0).sum() == round(0.2 * 47)  # 47 eligible -> 9


def test_long_short_dollar_neutral():
    s = _scores()
    w = construct_weights(s, _avail(s), _cfg(kind="long_short"))
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0.0, abs=1e-15)
    assert row.abs().sum() == pytest.approx(2.0)
    assert (row > 0).sum() == 10 and (row < 0).sum() == 10
    assert row.idxmin() in s.iloc[0].sort_values().index[:10]


def test_ties_broken_by_asset_name():
    s = pd.DataFrame([[1.0] * 49], index=_scores().index, columns=ASSETS)
    w = construct_weights(s, _avail(s), _cfg())
    assert list(w.columns[w.iloc[0] > 0]) == ASSETS[:10]


def test_inverse_vol_weights():
    s = _scores()
    vol = pd.DataFrame([np.linspace(0.1, 0.5, 49)], index=s.index, columns=ASSETS)
    w = construct_weights(s, _avail(s), _cfg(weighting="inverse_vol"), vol=vol)
    sel = w.iloc[0][w.iloc[0] > 0]
    inv = 1 / vol.iloc[0][sel.index]
    assert sel.to_numpy() == pytest.approx((inv / inv.sum()).to_numpy())
    with pytest.raises(ValueError, match="vol"):
        construct_weights(s, _avail(s), _cfg(weighting="inverse_vol"))


def test_band_keeps_holdings_until_rank_exceeds_exit():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-05", periods=2, freq="W-FRI"), name="date")
    first = np.arange(49, 0, -1.0)  # I00 best ... I48 worst
    second = first.copy()
    second[[0, 1]] = [first[12], first[20]]  # I00 drops to rank 13 (kept), I01 to 21 (sold)
    second[12], second[20] = first[0], first[1]
    s = pd.DataFrame([first, second], index=idx, columns=ASSETS)
    w = construct_weights(s, _avail(s), _cfg(band=(10, 15)))
    held2 = set(w.columns[w.iloc[1] > 0])
    assert "I00" in held2 and "I01" not in held2
    assert {"I12", "I20"} <= held2  # new top-ranked names enter
    assert w.iloc[1].sum() == pytest.approx(1.0)


def test_band_reduces_turnover_on_jittery_ranks():
    rng = np.random.default_rng(7)
    n = 104
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n * 5 + 10), name="date")
    reb = days[::5][:n]
    base = rng.normal(size=49)
    s = pd.DataFrame(base + rng.normal(0, 0.5, (n, 49)), index=reb, columns=ASSETS)
    r = pd.DataFrame(0.0, index=days, columns=ASSETS)
    rf = pd.Series(0.0, index=days)
    t_plain = run_backtest(construct_weights(s, _avail(s), _cfg()), r, rf).turnover.sum()
    t_band = run_backtest(
        construct_weights(s, _avail(s), _cfg(band=(10, 15))), r, rf
    ).turnover.sum()
    # Acceptance (ROADMAP B6): the band reduces turnover. Measured on this example
    # (2026-11-08): 32.9 vs 42.8 one-sided, i.e. -23%. The literal PLAN §5.6 band varies the
    # number held, and every count change re-weights all names, which limits the saving.
    assert t_band < t_plain


def test_no_eligible_assets_gives_cash_row():
    s = _scores()
    w = construct_weights(s, _avail(s) & False, _cfg())
    assert (w.iloc[0] == 0).all()
