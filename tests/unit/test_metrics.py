"""B4: metrics match closed form (PLAN §2.1) and QuantStats within 1e-6 (S40)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.metrics import (
    ann_return,
    ann_vol,
    annual_turnover,
    drawdown_duration,
    hit_rate_vs,
    ic_summary,
    max_drawdown,
    oos_r2,
    rank_ic,
    sharpe,
    sign_accuracy,
    sortino,
    summarize,
)

IDX = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=2520), name="date")


def _series(seed=0, mu=0.0004, sd=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, len(IDX)), index=IDX)


def test_sharpe_closed_form():
    r, rf = _series(), pd.Series(0.0001, index=IDX)
    ex = (r - rf).to_numpy()
    assert sharpe(r, rf) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252), rel=1e-12)


def test_sortino_closed_form():
    r, rf = _series(1), pd.Series(0.0, index=IDX)
    x = r.to_numpy()
    dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))  # RMS over ALL days, target 0
    assert sortino(r, rf) == pytest.approx(x.mean() / dd * np.sqrt(252), rel=1e-12)


def test_ann_return_geometric_and_vol():
    r = pd.Series(0.001, index=IDX[:252])
    assert ann_return(r) == pytest.approx(1.001**252 - 1, rel=1e-12)
    r2 = _series(2)
    assert ann_vol(r2) == pytest.approx(r2.std(ddof=1) * np.sqrt(252), rel=1e-12)


def test_max_drawdown_counts_loss_from_initial_wealth():
    r = pd.Series([-0.10, 0.05, 0.02, -0.03, 0.10], index=IDX[:5])
    assert max_drawdown(r) == pytest.approx(-0.10, abs=1e-15)
    r = pd.Series([0.10, -0.50, 0.20], index=IDX[:3])
    assert max_drawdown(r) == pytest.approx(-0.50, abs=1e-15)


def test_drawdown_duration_in_trading_days():
    # wealth: 1.1, 0.99, 1.089, 1.1979 -> peak at day 0, recovered on day 3 => 3 days
    r = pd.Series([0.10, -0.10, 0.10, 0.10], index=IDX[:4])
    assert drawdown_duration(r) == 3
    r = pd.Series([0.10, -0.10, 0.01], index=IDX[:3])  # never recovers: counts to the end
    assert drawdown_duration(r) == 2
    assert drawdown_duration(pd.Series([0.01, 0.01], index=IDX[:2])) == 0


def test_matches_quantstats():
    qs = pytest.importorskip("quantstats")
    r, rf = _series(3), pd.Series(0.00008, index=IDX)
    ex = r - rf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sharpe(r, rf) == pytest.approx(qs.stats.sharpe(ex, rf=0.0, periods=252), abs=1e-6)
        assert sortino(r, rf) == pytest.approx(qs.stats.sortino(ex, rf=0.0, periods=252), abs=1e-6)
        assert ann_vol(r) == pytest.approx(qs.stats.volatility(r, periods=252), abs=1e-6)
        assert max_drawdown(r) == pytest.approx(qs.stats.max_drawdown(r), abs=1e-6)


def test_annual_turnover():
    t = pd.Series(0.1, index=IDX[::5][:50])  # 50 rebalances of 10% one-sided
    assert annual_turnover(t, n_days=252) == pytest.approx(5.0 * 252 / 252)
    assert annual_turnover(t, n_days=504) == pytest.approx(2.5)


def test_hit_rate_vs_benchmark_weekly():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10), name="date")  # 2 weeks
    s = pd.Series([0.01] * 5 + [-0.01] * 5, index=idx)
    b = pd.Series(0.0, index=idx)
    assert hit_rate_vs(s, b) == 0.5


def _panel(pred_vals, real_vals):
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-12"], name="date")
    mi = pd.MultiIndex.from_product([dates, list("abcd")], names=["date", "asset"])
    return pd.Series(pred_vals, index=mi), pd.Series(real_vals, index=mi)


def test_rank_ic_per_date():
    p, y = _panel([1, 2, 3, 4, 1, 2, 3, 4], [10, 20, 30, 40, 40, 30, 20, 10])
    ic = rank_ic(p, y)
    assert ic.tolist() == pytest.approx([1.0, -1.0])
    assert ic.index.name == "date"


def test_rank_ic_ignores_nan_and_needs_3_assets():
    p, y = _panel([1, 2, 3, np.nan, 1, 2, np.nan, np.nan], [1, 2, 3, 4, 1, 2, 3, 4])
    ic = rank_ic(p, y)
    assert ic.iloc[0] == pytest.approx(1.0) and np.isnan(ic.iloc[1])


def test_ic_summary_newey_west():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(5)
    ic = pd.Series(0.02 + rng.normal(0, 0.1, 500))
    out = ic_summary(ic, nw_lags=4)
    fit = sm.OLS(ic.to_numpy(), np.ones(500)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    assert out["ic_mean"] == pytest.approx(ic.mean())
    assert out["ic_t_nw"] == pytest.approx(float(fit.tvalues[0]), rel=1e-10)
    assert out["n"] == 500


def test_sign_accuracy_cross_sectional():
    # demeaned per date: pred signs (-,-,+,+) vs realised (-,+,-,+) -> 2/4; date 2: 4/4
    p, y = _panel([1, 2, 3, 4, 1, 2, 3, 4], [1, 3, 2, 4, 1, 2, 3, 4])
    assert sign_accuracy(p, y) == pytest.approx(0.75)


def test_oos_r2_vs_benchmark_forecast():
    y = pd.Series([1.0, 2.0, 3.0])
    assert oos_r2(y, y, pd.Series([2.0, 2.0, 2.0])) == 1.0
    assert oos_r2(pd.Series([2.0] * 3), y, pd.Series([2.0] * 3)) == 0.0  # same as benchmark
    assert oos_r2(pd.Series([3.0, 3.0, 3.0]), y, pd.Series([2.0] * 3)) < 0  # worse


def test_summarize_keys_and_consistency():
    r, rf = _series(4), pd.Series(0.0001, index=IDX)
    t = pd.Series(0.05, index=IDX[::5])
    s = summarize(r, rf, turnover=t, benchmark=r * 0.5)
    for k in [
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "mdd",
        "mdd_duration_days",
        "turnover_ann",
        "hit_rate_vs_bench",
        "n_days",
        "start",
        "end",
    ]:
        assert k in s, k
    assert s["sharpe"] == sharpe(r, rf) and s["n_days"] == len(IDX)


def test_summarize_rejects_nan_returns():
    r = _series(6)
    r.iloc[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        summarize(r, pd.Series(0.0, index=IDX))


def test_rank_ic_index_is_sorted_by_date():
    dates = pd.DatetimeIndex(["2024-01-12", "2024-01-05"], name="date")  # out of order
    mi = pd.MultiIndex.from_product([dates, list("abc")], names=["date", "asset"])
    p = pd.Series([1, 2, 3, 3, 2, 1], index=mi, dtype=float)
    ic = rank_ic(p, p)
    assert ic.index.is_monotonic_increasing
    assert ic.loc["2024-01-01":"2024-01-31"].tolist() == pytest.approx([1.0, 1.0])


def test_sortino_without_downside_is_nan_not_inf():
    r = pd.Series(0.001, index=IDX[:10])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero warning either
        assert np.isnan(sortino(r, pd.Series(0.0, index=IDX[:10])))
