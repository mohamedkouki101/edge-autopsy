"""Ridge regression on the 14 features (rung B5; PREREGISTRATION §7: alpha in {0.1 ... 1000}).

Missing feature values are set to 0: the median rank for asset features and the long-run
mean for market z-scores. Deterministic (closed form).
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

RIDGE_GRID = {"alpha": [0.1, 1, 10, 100, 1000]}


class RidgeForecaster:
    def __init__(self, params: dict):
        self.params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        self.columns_ = list(X.columns)
        self.model_ = Ridge(alpha=self.params["alpha"], fit_intercept=True)
        self.model_.fit(X.fillna(0.0).to_numpy(), y.to_numpy())
        self.coef_, self.intercept_ = self.model_.coef_, float(self.model_.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        p = self.model_.predict(X[self.columns_].fillna(0.0).to_numpy())
        return pd.Series(p, index=X.index, name="pred")


def ridge_model(cfg):
    from edge_autopsy.features import universe_features
    from edge_autopsy.runner import LearnedModel
    from edge_autopsy.tuning import grid

    return LearnedModel(
        "ridge", RidgeForecaster, grid(RIDGE_GRID), lambda u: universe_features(u, cfg),
        params={"missing": "fill 0", "fit_intercept": True},
    )  # fmt: skip
