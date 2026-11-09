"""M1 / K2, K4: the 14 features (PREREGISTRATION §4 + Amendment 1) use data <= close t only.

Acceptance: `test_features_no_lookahead` - mutate every return (assets and market) after t;
every feature on dates <= t must be unchanged, bit for bit (hypothesis).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edge_autopsy.config import load_config
from edge_autopsy.features import (
    ASSET_FEATURES,
    FEATURES,
    MARKET_FEATURES,
    build_features,
    raw_features,
)

CFG = load_config("configs/base.yaml").features


def _data(n=420, m=6, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=n), name="date")
    mkt = pd.Series(rng.normal(0.0004, 0.01, n), index=idx, name="mkt")
    beta = rng.uniform(0.5, 1.5, m)
    r = mkt.to_numpy()[:, None] * beta + rng.normal(0, 0.01, (n, m))
    rets = pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(m)])
    if nan_frac:
        rets = rets.mask(rng.random(rets.shape) < nan_frac)
    return rets, mkt


# ---------------------------------------------------------------- acceptance: no lookahead


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    t=st.integers(0, 419),
    seed=st.integers(0, 10_000),
    nan_frac=st.sampled_from([0.0, 0.0, 0.01]),
    scale=st.sampled_from([0.0, 0.05, 10.0]),
)
def test_features_no_lookahead(t, seed, nan_frac, scale):
    rets, mkt = _data(seed=seed % 7, nan_frac=nan_frac)
    before = build_features(rets, mkt, CFG)
    rng = np.random.default_rng(seed)
    r2, m2 = rets.copy(), mkt.copy()
    # any future path, but still a valid simple return (> -100%)
    r2.iloc[t + 1 :] = np.maximum(rng.normal(0, scale, r2.iloc[t + 1 :].shape), -0.95)
    m2.iloc[t + 1 :] = np.maximum(rng.normal(0, scale, len(m2) - t - 1), -0.95)
    after = build_features(r2, m2, CFG)
    cut = rets.index[t]
    pd.testing.assert_frame_equal(before.loc[:cut], after.loc[:cut], check_exact=True)


def test_mutating_the_past_does_change_features():
    """Guard against a vacuous no-lookahead test: features do depend on data <= t."""
    rets, mkt = _data()
    a = build_features(rets, mkt, CFG)
    r2 = rets.copy()
    r2.iloc[300, 0] += 0.2
    b = build_features(r2, mkt, CFG)
    assert not a.loc[rets.index[300]].equals(b.loc[rets.index[300]])


# ---------------------------------------------------------------- shape


def test_output_is_long_with_the_14_preregistered_columns():
    rets, mkt = _data()
    X = build_features(rets, mkt, CFG)
    assert list(X.columns) == FEATURES and len(FEATURES) == 14
    assert len(ASSET_FEATURES) == 11 and len(MARKET_FEATURES) == 3
    assert X.index.names == ["date", "asset"] and len(X) == rets.size


def test_dates_argument_only_subsets_rows():
    rets, mkt = _data()
    full = build_features(rets, mkt, CFG)
    some = rets.index[[260, 300, 419]]
    sub = build_features(rets, mkt, CFG, dates=some)
    pd.testing.assert_frame_equal(sub, full.loc[some], check_exact=True)


# ---------------------------------------------------------------- definitions (Amendment 1)


def _comp(x):
    return float(np.prod(1 + np.asarray(x)) - 1)


def test_raw_definitions_match_direct_computation():
    rets, mkt = _data()
    raw = raw_features(rets, mkt, CFG)
    t, a = 400, "A2"
    r, m = rets[a].to_numpy(), mkt.to_numpy()
    for w in (5, 21, 63, 126, 252):
        assert raw[f"ret_{w}"].iloc[t][a] == pytest.approx(_comp(r[t - w + 1 : t + 1]), rel=1e-12)
    assert raw["mom_12_1"].iloc[t][a] == pytest.approx(_comp(r[t - 251 : t - 20]), rel=1e-12)
    for w in (21, 63):
        assert raw[f"vol_{w}"].iloc[t][a] == pytest.approx(np.std(r[t - w + 1 : t + 1], ddof=1))
    rw, mw = r[t - 251 : t + 1], m[t - 251 : t + 1]
    assert raw["beta_252"].iloc[t][a] == pytest.approx(np.cov(rw, mw)[0, 1] / np.var(mw, ddof=1))
    path = np.concatenate([[1.0], np.cumprod(1 + r[t - 62 : t + 1])])
    mdd = float(np.min(path / np.maximum.accumulate(path)) - 1)
    assert raw["mdd_63"].iloc[t][a] == pytest.approx(mdd, rel=1e-10)
    assert raw["skew_21"].iloc[t][a] == pytest.approx(pd.Series(r[t - 20 : t + 1]).skew())
    assert raw["mkt_ret_21"].iloc[t] == pytest.approx(_comp(m[t - 20 : t + 1]), rel=1e-12)
    assert raw["mkt_vol_21"].iloc[t] == pytest.approx(np.std(m[t - 20 : t + 1], ddof=1))
    r21 = [_comp(rets[c].to_numpy()[t - 20 : t + 1]) for c in rets.columns]
    assert raw["disp_21"].iloc[t] == pytest.approx(np.std(r21, ddof=1))


def test_max_drawdown_counts_a_fall_from_the_window_start():
    rets, mkt = _data(n=300)
    rets.iloc[:, 0] = 0.0
    rets.iloc[-63, 0] = -0.10  # first day of the last 63-day window
    rets.iloc[-30, 0] = 0.05
    raw = raw_features(rets, mkt, CFG)
    assert raw["mdd_63"].iloc[-1]["A0"] == pytest.approx(-0.10)
    rets.iloc[-63, 0] = 0.0
    assert raw_features(rets, mkt, CFG)["mdd_63"].iloc[-1]["A0"] == 0.0


def test_a_missing_return_in_the_window_makes_the_feature_missing():
    rets, mkt = _data()
    rets.iloc[395, 1] = np.nan
    raw = raw_features(rets, mkt, CFG)
    t = 400
    for f in ("ret_21", "ret_252", "vol_21", "beta_252", "mdd_63", "skew_21"):
        assert np.isnan(raw[f].iloc[t]["A1"]), f
        assert not np.isnan(raw[f].iloc[t]["A0"]), f
    assert not np.isnan(raw["ret_5"].iloc[t]["A1"])  # 395 is outside the 5-day window


# ---------------------------------------------------------------- normalisation


def test_asset_features_are_cross_sectional_ranks_in_minus_half_to_half():
    rets, mkt = _data(nan_frac=0.02)
    X = build_features(rets, mkt, CFG)
    vals = X[ASSET_FEATURES].stack().dropna()
    assert len(vals) > 0 and vals.between(-0.5, 0.5).all()
    day = X.xs(rets.index[400], level="date")["ret_21"].dropna()
    n = len(day)
    assert sorted(day) == pytest.approx(sorted(np.arange(n) / (n - 1) - 0.5))


def test_rank_ties_are_averaged_and_a_single_asset_gets_zero():
    from edge_autopsy.features import cs_rank

    wide = pd.DataFrame([[1.0, 1.0, 3.0], [np.nan, 2.0, np.nan]], columns=list("abc"))
    out = cs_rank(wide)
    assert out.iloc[0].tolist() == [-0.25, -0.25, 0.5]
    assert np.isnan(out.iloc[1]["a"]) and out.iloc[1]["b"] == 0.0


def test_market_features_are_z_scored_on_strictly_past_days_from_the_floor():
    rets, mkt = _data(n=700)
    floor = rets.index[100]
    raw = raw_features(rets, mkt, CFG)
    X = build_features(rets, mkt, CFG, zscore_start=floor)
    x = raw["mkt_vol_21"]
    t = 600
    past = x.iloc[100:t].dropna()
    expect = (x.iloc[t] - past.mean()) / past.std(ddof=1)
    got = X.xs(rets.index[t], level="date")["mkt_vol_21"]
    assert (got == got.iloc[0]).all()  # same value for every asset on a date
    assert got.iloc[0] == pytest.approx(expect, rel=1e-10)
    # missing until 252 past days from the floor exist
    z = X.xs("A0", level="asset")["mkt_vol_21"]
    first = z.first_valid_index()
    assert first == x.loc[floor:].dropna().index[252]


def test_market_z_ignores_everything_before_the_floor():
    rets, mkt = _data(n=700)
    floor = rets.index[300]
    a = build_features(rets, mkt, CFG, zscore_start=floor)
    m2 = mkt.copy()
    m2.iloc[:250] += 0.5  # wild pre-floor history (but outside every window after day 299)
    b = build_features(rets, m2, CFG, zscore_start=floor)
    later = rets.index[600:]
    # pandas rolling sums are incremental, so far-past values leave ~1e-16 rounding residue:
    # equal to 1e-12, not bit-exact (past data only; the lookahead test above stays exact)
    pd.testing.assert_frame_equal(
        a.loc[later, MARKET_FEATURES],
        b.loc[later, MARKET_FEATURES],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_universe_features_uses_rebalance_dates_and_the_sample_floor():
    from edge_autopsy.data.calendar import rebalance_dates
    from edge_autopsy.data.load import Universe
    from edge_autopsy.features import universe_features

    cfg = load_config("configs/base.yaml")
    rets, mkt = _data(n=700)
    av = rets.notna()
    u = Universe(returns=rets, available=av, rf=mkt * 0, mkt=mkt, volume_usd=rets * np.nan)
    floor = rets.index[100]
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"sample_floor": floor.date()})}
    )
    X = universe_features(u, cfg)
    weekly = rebalance_dates(rets.index, cfg.rebalance_freq)
    assert X.index.get_level_values("date").unique().equals(weekly)
    full = build_features(rets, mkt, cfg.features, zscore_start=floor)
    pd.testing.assert_frame_equal(X, full.loc[weekly], check_exact=True)
