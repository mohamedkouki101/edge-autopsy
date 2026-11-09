"""M2: the pipeline must find a signal that exists, find none in noise, and a planted leak
must both inflate IC and be caught by the no-lookahead check (PLAN §9.2).

Same code path as real experiments: run_experiment(LearnedModel) with the 14 features plus,
here, one extra column (the observed signal or the leak). Pre-registered grids.
"""

import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.config import load_config
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.features import universe_features
from edge_autopsy.labels import build_labels
from edge_autopsy.metrics import ic_summary, rank_ic
from edge_autopsy.models.linear import ridge_model
from edge_autopsy.models.xgb import xgb_model
from edge_autopsy.runner import run_experiment
from tests.synthetic.generator import long_rank, make_synthetic, write_panel

START = "1987-01-02"  # 30 years of business days -> 2015-12


def _cfg(last_day):
    cfg = load_config("configs/base.yaml")
    data = cfg.data.model_copy(
        update={"sample_floor": dt.date(1988, 1, 1), "sample_start": dt.date(1988, 1, 8)}
    )
    split = cfg.split.model_copy(
        update={
            "dev_test_start": dt.date(2006, 1, 1),  # 10 test years, >= 18 training years
            "dev_test_end": last_day.date(),
            "holdout_start": (last_day + pd.Timedelta(days=1)).date(),
            "holdout_end": dt.date(2030, 1, 1),
        }
    )
    return cfg.model_copy(update={"data": data, "split": split})


class World:
    def __init__(self, tmp, ic, seed):
        self.returns, self.mkt, self.signal = make_synthetic(ic=ic, seed=seed)
        self.panel = tmp / "panel.parquet"
        write_panel(self.returns, self.mkt, self.panel)
        self.cfg = _cfg(self.returns.index[-1])
        self.tmp = tmp

    def run(self, model, extra):
        """extra: universe -> long Series added to the 14 features."""
        base_features = model.features
        spec = dataclasses.replace(model, features=lambda u: base_features(u).join(extra(u)))
        row = run_experiment(
            self.cfg, spec, panel_path=self.panel, log_path=self.tmp / "e.jsonl",
            results_dir=self.tmp / "results", data_version="synthetic",
            config_path="synthetic", phase="exploratory",
        )  # fmt: skip
        return row["metrics"]


@pytest.fixture(scope="module")
def signal_world(tmp_path_factory):
    return World(tmp_path_factory.mktemp("signal"), ic=0.05, seed=11)


@pytest.fixture(scope="module")
def noise_world(tmp_path_factory):
    return World(tmp_path_factory.mktemp("noise"), ic=0.0, seed=12)


def _models(cfg):
    return {"ridge": ridge_model(cfg), "xgb": xgb_model(cfg)}


def test_generator_true_ic_is_the_planted_one(signal_world):
    """Calibration of the harness itself: the oracle (f itself) has rank IC ~0.05."""
    w = signal_world
    labels = build_labels(w.returns, 5, 1, rebalance=rebalance_dates(w.returns.index))
    ic = rank_ic(long_rank(w.signal, "f"), labels)
    s = ic_summary(ic.dropna())
    assert 0.045 <= s["ic_mean"] <= 0.055, s


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ridge", "xgb"])
def test_known_signal_is_recovered(signal_world, name):
    w = signal_world
    m = w.run(_models(w.cfg)[name], lambda u: long_rank(w.signal, "signal"))
    print(name, json.dumps({k: m[k] for k in ("ic_mean", "ic_t_nw", "sharpe_gross", "sharpe_net")}))
    assert 0.03 <= m["ic_mean"] <= 0.07
    assert m["sharpe_gross"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ridge", "xgb"])
def test_pure_noise_gives_an_ic_indistinguishable_from_zero(noise_world, name):
    w = noise_world
    m = w.run(_models(w.cfg)[name], lambda u: long_rank(w.signal, "signal"))
    print(name, json.dumps({k: m[k] for k in ("ic_mean", "ic_t_nw", "sharpe_gross")}))
    assert abs(m["ic_t_nw"]) < 1.96  # 95% CI of the mean IC contains 0
    # DSR < 0.5 and PBO ~ 0.5 (PLAN §9.2) are added when the stats layer exists (P5).


def _leak(lead):
    """Return of the day `lead` days after t, ranked: lead=2 is the first day the position
    earns (lag 1); lead=1 is the literal PLAN L2 `ret.shift(-1)`."""

    def extra(u):
        dates = rebalance_dates(u.returns.index)
        return long_rank(u.returns.shift(-lead).loc[dates], "leak")

    return extra


def _has_lookahead(feature_fn, w, t=5000):
    """The K2 check (test_features_no_lookahead) applied to a feature function."""
    from edge_autopsy.data.load import Universe

    def feats(rets):
        u = Universe(rets, rets.notna(), w.mkt * 0, w.mkt, rets * np.nan)
        return feature_fn(u)

    before = feats(w.returns)
    r2 = w.returns.copy()
    r2.iloc[t + 1 :] = np.random.default_rng(0).normal(0, 0.01, r2.iloc[t + 1 :].shape)
    after = feats(r2)
    cut = w.returns.index[t]
    return not before.loc[:cut].equals(after.loc[:cut])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ridge", "xgb"])
def test_planted_leak_inflates_ic_and_is_caught(noise_world, name):
    w = noise_world
    m = w.run(_models(w.cfg)[name], _leak(2))
    print(name, json.dumps({k: m[k] for k in ("ic_mean", "ic_t_nw", "sharpe_gross")}))
    assert m["ic_mean"] > 0.1  # a fake edge appears ...
    assert _has_lookahead(_leak(2), w)  # ... and the K2 check catches the feature
    assert not _has_lookahead(lambda u: universe_features(u, w.cfg), w)  # the real ones pass


@pytest.mark.slow
def test_literal_l2_next_day_return_does_not_reach_the_label(noise_world):
    """With a 1-day execution lag, the return from close t to t+1 is never earned, so the
    literal L2 feature `ret.shift(-1)` can't inflate IC. (Input for the P6 leak-lab design.)"""
    w = noise_world
    m = w.run(_models(w.cfg)["ridge"], _leak(1))
    print("ridge L2-literal", json.dumps({k: m[k] for k in ("ic_mean", "ic_t_nw")}))
    assert abs(m["ic_t_nw"]) < 1.96
    assert _has_lookahead(_leak(1), w)  # still a lookahead bug, and still caught by K2
