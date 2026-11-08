"""B5 orchestration on a synthetic panel that *contains* hold-out dates: outputs never do."""

import json

import numpy as np
import pandas as pd

from edge_autopsy.baselines_dev import BASELINES, run_baselines
from edge_autopsy.config import load_config
from edge_autopsy.data.validate import PANEL_COLUMNS


def _synthetic_panel(path):
    idx = pd.bdate_range("1960-01-01", "2017-06-30")
    rng = np.random.default_rng(0)
    rows = []
    for k in range(10):
        r = rng.normal(0.0003, 0.01, len(idx))
        rows.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "asset": f"I{k}",
                    "universe": "ff49",
                    "ret": r,
                    "available": np.arange(len(idx)) >= 252,
                    "rf": 0.0001,
                    "mkt": 0.0004,
                    "volume_usd": np.nan,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["date"] = df["date"].astype("datetime64[ns]")
    for c in ("asset", "universe"):
        df[c] = df[c].astype("category")
    df[PANEL_COLUMNS].to_parquet(path, index=False)


def test_run_baselines_logs_all_and_never_touches_holdout(tmp_path):
    panel, log, res = tmp_path / "panel.parquet", tmp_path / "e.jsonl", tmp_path / "results"
    _synthetic_panel(panel)
    cfg = load_config("configs/base.yaml")
    out = run_baselines(cfg, panel, log, res, data_version="test")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["model"] for r in rows] == list(BASELINES) and len(out) == 5
    for r in rows:
        assert r["phase"] == "dev" and r["metrics"]["end"] <= "2016-12-31"
        assert r["label"] == "holding_period"
        assert r["metrics"]["start"] >= "1980-01-01"
        daily = pd.read_parquet(tmp_path / r["results_path"] / "daily.parquet")
        assert daily.index.max() <= pd.Timestamp("2016-12-31")
    mom = next(r for r in rows if r["model"] == "momentum_12_1")
    assert mom["metrics"]["sharpe_gross"] >= mom["metrics"]["sharpe_net"]
    assert mom["metrics"]["turnover_ann"] > 0 and mom["metrics"]["ic_mean"] is not None
    b0 = next(r for r in rows if r["model"] == "equal_weight_bh")
    assert b0["metrics"]["turnover_ann"] < mom["metrics"]["turnover_ann"]
