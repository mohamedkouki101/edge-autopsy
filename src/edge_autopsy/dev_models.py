"""M3/M4: pre-registered model runs on the development period (PREREGISTRATION §7).

Each (model, scheme) pair is one dev trial: ridge and XGBoost with the 14 features, purged
walk-forward (expanding = primary, rolling 10y = pre-registered comparison A1), nested
tuning, refit, test 1980-2016. Hold-out data is unreachable (loader + guard).
"""

from __future__ import annotations

from pathlib import Path

from edge_autopsy.config import ExperimentCfg
from edge_autopsy.models.linear import ridge_model
from edge_autopsy.models.xgb import xgb_model
from edge_autopsy.runner import run_experiment

MODELS = (ridge_model, xgb_model)


def run_dev_models(
    cfg: ExperimentCfg,
    panel_path: Path,
    log_path: Path,
    results_dir: Path,
    data_version: str,
    models=MODELS,
    schemes=("expanding", "rolling"),
    config_path: str = "configs/base.yaml",
) -> list[dict]:
    rows = []
    for build in models:
        for scheme in schemes:
            c = cfg.model_copy(update={"split": cfg.split.model_copy(update={"scheme": scheme})})
            spec = build(c)
            rows.append(
                run_experiment(
                    c,
                    spec,
                    panel_path=panel_path,
                    log_path=log_path,
                    results_dir=results_dir,
                    data_version=data_version,
                    config_path=config_path,
                    phase="dev",
                    pre_registered=True,
                    notes=f"{spec.name} {scheme}: PREREG §7 grid, 14 features (Amendment 1)",
                )  # fmt: skip
            )
    return rows
