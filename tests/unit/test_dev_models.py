"""M3/M4: pre-registered model runs on the dev period, one row per (model, scheme)."""

import datetime as dt
import json

from edge_autopsy.config import load_config
from edge_autopsy.dev_models import run_dev_models
from edge_autopsy.models.linear import ridge_model
from tests.unit.test_baselines_dev import _synthetic_panel


def test_run_dev_models_logs_one_preregistered_dev_row_per_scheme(tmp_path):
    panel = tmp_path / "panel.parquet"
    _synthetic_panel(panel)
    cfg = load_config("configs/base.yaml")
    cfg = cfg.model_copy(
        update={"split": cfg.split.model_copy(update={"dev_test_start": dt.date(2015, 1, 1)})}
    )
    out = run_dev_models(
        cfg, panel, tmp_path / "e.jsonl", tmp_path / "results", data_version="test",
        models=[ridge_model], schemes=("expanding", "rolling"),
    )  # fmt: skip
    rows = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert [(r["model"], r["split"]["scheme"]) for r in rows] == [
        ("ridge", "expanding"),
        ("ridge", "rolling"),
    ]
    assert len(out) == 2 and rows[0]["config_hash"] != rows[1]["config_hash"]
    for r in rows:
        assert r["phase"] == "dev" and r["pre_registered"] is True
        assert r["n_inner_trials"] == 5 and r["metrics"]["end"] <= "2016-12-31"
