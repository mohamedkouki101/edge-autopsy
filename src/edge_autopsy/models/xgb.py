"""XGBoost forecaster (primary model; PREREGISTRATION §7, PLAN §7.1).

Grid: max_depth {2,3,4} x learning_rate {0.01, 0.05} x min_child_weight {50, 200} (12
configs); subsample = colsample_bytree = 0.8; up to 2000 trees with early stopping (50 rounds)
on the inner validation fold. `n_jobs=1` and a fixed seed make fits reproducible (U8).

Final refit on the whole training window has no validation fold, so it uses the mean number
of trees chosen by early stopping in the inner folds of the selected config
(`xgb_refit_params`). Missing values are handled by XGBoost natively.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

XGB_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.01, 0.05], "min_child_weight": [50, 200]}
XGB_FIXED = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
}
MAX_TREES, EARLY_STOP = 2000, 50


class XGBForecaster:
    def __init__(self, params: dict, seed: int):
        self.params, self.seed = dict(params), seed

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        self.columns_ = list(X.columns)
        kw = {**XGB_FIXED, **self.params, "random_state": self.seed}
        if X_val is not None:
            kw.setdefault("n_estimators", MAX_TREES)
            self.model_ = XGBRegressor(**kw, early_stopping_rounds=EARLY_STOP)
            self.model_.fit(X, y, eval_set=[(X_val[self.columns_], y_val)], verbose=False)
            self.fit_info_ = {"best_iteration": int(self.model_.best_iteration)}
        else:
            if "n_estimators" not in self.params:
                raise ValueError("a fit without a validation fold needs an explicit n_estimators")
            self.model_ = XGBRegressor(**kw)
            self.model_.fit(X, y, verbose=False)
            self.fit_info_ = {"n_estimators": int(self.params["n_estimators"])}
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model_.predict(X[self.columns_]), index=X.index, name="pred")


def xgb_refit_params(trial: dict) -> dict:
    """Selected config + the mean early-stopping tree count over its inner folds."""
    trees = [i["best_iteration"] + 1 for i in trial["fit_info"]]
    return {**trial["params"], "n_estimators": int(round(float(np.mean(trees))))}


def xgb_model(cfg):
    from edge_autopsy.features import universe_features
    from edge_autopsy.runner import LearnedModel
    from edge_autopsy.tuning import grid

    return LearnedModel(
        "xgb",
        partial(XGBForecaster, seed=cfg.seed),
        grid(XGB_GRID),
        lambda u: universe_features(u, cfg),
        params={**XGB_FIXED, "max_trees": MAX_TREES, "early_stopping_rounds": EARLY_STOP,
                "refit_trees": "mean inner best_iteration + 1"},
        refit_params=xgb_refit_params,
    )  # fmt: skip
