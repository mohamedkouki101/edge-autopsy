"""V5: run_experiment() = one pipeline for every model (rule or learned), one JSONL row per run
(PLAN §7.4) + a results folder. Synthetic panel 1960-2017 that *contains* hold-out dates."""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.baselines_dev import run_baselines
from edge_autopsy.config import load_config
from edge_autopsy.models.baselines import momentum_12_1_score
from edge_autopsy.runner import LearnedModel, RuleModel, run_experiment
from edge_autopsy.splits import HoldoutSealedError, label_end
from tests.unit.test_baselines_dev import _synthetic_panel

PLAN_7_4 = {
    "run_id", "code_hash", "config_hash", "config_path", "data_manifest_hash", "phase",
    "universe", "model", "params", "split", "portfolio", "cost_bps", "lag", "label",
    "borrow_bps", "seed", "n_inner_trials", "n_inner_fits", "metrics", "ci", "results_path",
    "pre_registered", "notes",
}  # fmt: skip
METRICS = {
    "sharpe_net", "sharpe_gross", "ic_mean", "ic_t_nw", "oos_r2", "mdd", "turnover_ann",
    "hit_rate_vs_b0", "sortino", "ann_return_net", "ann_vol",
}  # fmt: skip


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    p = tmp_path_factory.mktemp("panel") / "panel.parquet"
    _synthetic_panel(p)
    return p


def _cfg(test_start=dt.date(2012, 1, 1)):
    cfg = load_config("configs/base.yaml")
    return cfg.model_copy(
        update={"split": cfg.split.model_copy(update={"dev_test_start": test_start})}
    )


MOMENTUM = RuleModel(
    "momentum_12_1", {"lookback": 252, "skip": 21}, lambda u: momentum_12_1_score(u.returns)
)


def _run(model, panel, tmp_path, cfg=None, **kw):
    return run_experiment(
        cfg or _cfg(),
        model,
        panel_path=panel,
        log_path=tmp_path / "e.jsonl",
        results_dir=tmp_path / "results",
        data_version="test",
        config_path="configs/base.yaml",
        **kw,
    )


def _rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]


def test_running_a_baseline_twice_appends_two_identical_rows(panel, tmp_path):
    """V5 acceptance."""
    _run(MOMENTUM, panel, tmp_path)
    _run(MOMENTUM, panel, tmp_path)
    a, b = _rows(tmp_path)
    assert a["run_id"] != b["run_id"]
    assert a["metrics"] == b["metrics"] and a["config_hash"] == b["config_hash"]
    for r in (a, b):
        out = tmp_path / r["results_path"]
        assert (out / "daily.parquet").exists() and (out / "scores.parquet").exists()


def test_row_has_the_plan_7_4_fields(panel, tmp_path):
    _run(MOMENTUM, panel, tmp_path, pre_registered=True, notes="B3 via run_experiment")
    (r,) = _rows(tmp_path)
    assert PLAN_7_4 <= set(r)
    assert METRICS <= set(r["metrics"])
    assert r["phase"] == "dev" and r["pre_registered"] is True
    assert r["metrics"]["start"] >= "2012-01-01" and r["metrics"]["end"] <= "2016-12-31"
    assert r["n_inner_trials"] == 0 and r["split"]["scheme"] == "none (rule)"


@pytest.mark.slow
def test_rule_model_reproduces_the_b5_baseline_numbers_exactly(panel, tmp_path):
    """Same engine as B5 (whose numbers are frozen in PREREGISTRATION §12)."""
    cfg = load_config("configs/base.yaml")
    b5 = run_baselines(cfg, panel, tmp_path / "b5.jsonl", tmp_path / "b5", data_version="test")
    mom = next(r for r in b5 if r["model"] == "momentum_12_1")
    _run(MOMENTUM, panel, tmp_path, cfg=cfg)
    (r,) = _rows(tmp_path)
    for k in ("sharpe_net", "sharpe_gross", "ic_mean", "ic_t_nw", "mdd", "turnover_ann"):
        assert r["metrics"][k] == mom[k], k
    assert r["metrics"]["hit_rate_vs_b0"] == mom["hit_rate_vs_b0"]


# ---------------------------------------------------------------- learned models


def _features(u):
    """Two features from data <= t: 5-day return and a fixed-seed noise column."""
    mom5 = (1 + u.returns.fillna(0)).rolling(5).apply(np.prod, raw=True) - 1
    X = mom5.rename_axis(index="date", columns="asset").stack(future_stack=True).to_frame("mom5")
    X["noise"] = np.random.default_rng(1).normal(size=len(X))
    return X


class OLS:
    calls: list = []
    y_means: list = []

    def __init__(self, params):
        self.cols = list(params["cols"])

    def fit(self, X, y, X_val=None, y_val=None):
        OLS.calls.append(("fit", X.index.get_level_values("date").unique(), X_val is None))
        OLS.y_means.append(float(y.groupby(level="date").mean().abs().max()))
        A = X[self.cols].fillna(0).to_numpy()
        self.beta = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
        return self

    def predict(self, X):
        OLS.calls.append(("predict", X.index.get_level_values("date").unique(), None))
        return pd.Series(X[self.cols].fillna(0).to_numpy() @ self.beta, index=X.index)


LEARN_START = dt.date(2015, 1, 1)
LEARNED = LearnedModel(
    "ols_toy",
    OLS,
    [{"cols": ["mom5"]}, {"cols": ["noise"]}, {"cols": ["mom5", "noise"]}],
    _features,
)


def test_learned_model_runs_walk_forward_deterministically(panel, tmp_path):
    _run(LEARNED, panel, tmp_path, cfg=_cfg(LEARN_START))
    _run(LEARNED, panel, tmp_path, cfg=_cfg(LEARN_START))
    a, b = _rows(tmp_path)
    assert a["metrics"] == b["metrics"]
    assert a["split"]["scheme"] == "expanding" and a["split"]["purge_days"] == 6
    n_folds = a["split"]["n_folds"]
    assert n_folds == 3  # 105 weekly test dates 2015-2016 = 52 + 52 + a 1-week stub
    assert a["n_inner_trials"] == 3
    assert a["n_inner_fits"] == n_folds * 3 * 3  # folds x configs x inner folds
    assert a["metrics"]["oos_r2"] is not None and a["metrics"]["ic_mean"] is not None
    trials = json.loads((tmp_path / a["results_path"] / "trials.json").read_text())
    assert len(trials) == n_folds and all(len(t["trials"]) == 3 for t in trials)
    scores = pd.read_parquet(tmp_path / a["results_path"] / "scores.parquet")
    assert scores.index.min() >= pd.Timestamp("2015-01-01")
    assert scores.index.max() <= pd.Timestamp("2016-12-31")


def test_final_refit_trains_only_on_labels_known_before_the_test_block(panel, tmp_path):
    OLS.calls, OLS.y_means = [], []
    cfg = _cfg(LEARN_START)
    _run(LEARNED, panel, tmp_path, cfg=cfg)
    cal = pd.bdate_range("1960-01-01", "2016-12-31")  # the synthetic calendar, dev part
    refits = [(i, c) for i, c in enumerate(OLS.calls) if c[0] == "fit" and c[2]]
    assert len(refits) == 3
    for i, (_, fit_dates, _) in refits:
        _, test_dates, _ = OLS.calls[i + 1]  # the refit model's predict call
        assert OLS.calls[i + 1][0] == "predict"
        assert label_end(fit_dates, cal, cfg.split.purge_days).max() < test_dates.min()
    # target = label minus the cross-sectional mean on each date (PREREGISTRATION §3)
    assert max(OLS.y_means) < 1e-12


def test_holdout_phase_is_sealed_and_leaves_no_trace(panel, tmp_path, monkeypatch):
    monkeypatch.delenv("UNSEAL", raising=False)
    with pytest.raises(HoldoutSealedError):
        _run(MOMENTUM, panel, tmp_path, phase="holdout")
    assert not (tmp_path / "e.jsonl").exists()
    assert not (tmp_path / "results").exists()


def test_unknown_phase_is_rejected(panel, tmp_path):
    with pytest.raises(ValueError, match="phase"):
        _run(MOMENTUM, panel, tmp_path, phase="final_final")
