"""B5: baseline scores use data <= t only; hand examples."""

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.models.baselines import (
    equal_weight_targets,
    hist_mean_score,
    momentum_12_1_score,
    reversal_1w_score,
)

IDX = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=400), name="date")


def _r(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (400, 3)), index=IDX, columns=list("ABC"))


@pytest.mark.parametrize("fn", [hist_mean_score, momentum_12_1_score, reversal_1w_score])
def test_scores_use_only_past_data(fn):
    r = _r()
    t = 300
    base = fn(r).iloc[: t + 1]
    m = r.copy()
    m.iloc[t + 1 :] = 0.5  # wreck the future
    assert fn(m).iloc[: t + 1].equals(base)


def test_momentum_12_1_hand_example():
    r = _r(1)
    t = 299
    window = r["A"].iloc[t - 251 : t - 20]  # days t-251 .. t-21 inclusive (231 days)
    expected = float(np.prod(1 + window.to_numpy()) - 1)
    assert momentum_12_1_score(r)["A"].iloc[t] == pytest.approx(expected, rel=1e-10)
    m = r.copy()
    m.iloc[t - 20 :, 0] = 0.3  # the skipped last month must not matter
    assert momentum_12_1_score(m)["A"].iloc[t] == pytest.approx(expected, rel=1e-10)
    assert np.isnan(momentum_12_1_score(r)["A"].iloc[250])  # < 252 days of history


def test_reversal_hand_example():
    r = _r(2)
    t = 10
    expected = -(float(np.prod(1 + r["B"].iloc[t - 4 : t + 1].to_numpy())) - 1)
    assert reversal_1w_score(r)["B"].iloc[t] == pytest.approx(expected, rel=1e-10)


def test_hist_mean_is_expanding_mean_with_min_history():
    r = _r(3)
    s = hist_mean_score(r, min_obs=252)
    assert np.isnan(s["C"].iloc[250])
    assert s["C"].iloc[260] == pytest.approx(r["C"].iloc[:261].mean(), rel=1e-12)


def test_equal_weight_targets_over_available():
    idx = pd.DatetimeIndex(["2000-01-31", "2000-02-29"], name="date")
    av = pd.DataFrame([[True, True, False], [True, True, True]], index=idx, columns=list("ABC"))
    w = equal_weight_targets(av)
    assert w.iloc[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert w.iloc[1].tolist() == pytest.approx([1 / 3] * 3)
