"""M4: the dev-OOS comparison table is generated from the run log + saved daily results
(never typed by hand). Net Sharpe at each cost is rebuilt exactly: costs are linear and the
gross path does not depend on them."""

import json

import numpy as np
import pandas as pd
import pytest

from edge_autopsy.metrics import sharpe
from edge_autopsy.reports.tables import cost_grid_table, to_markdown

IDX = pd.bdate_range("2000-01-03", periods=600, name="date")
RF = pd.Series(0.0001, index=IDX)


def _run(root, rid, model, scheme, seed, cost_bps=10.0, phase="dev", chash=None):
    rng = np.random.default_rng(seed)
    gross = pd.Series(rng.normal(0.0006, 0.01, len(IDX)), index=IDX)
    traded = pd.Series(np.where(np.arange(len(IDX)) % 5 == 0, 0.4, 0.0), index=IDX)
    cost = traded * cost_bps / 1e4
    out = root / "results" / rid
    out.mkdir(parents=True)
    pd.DataFrame({"gross": gross, "net": gross - cost, "cost": cost, "traded": traded}).to_parquet(
        out / "daily.parquet"
    )
    ev = slice("2000-06-01", "2002-03-29")
    row = {
        "run_id": rid, "config_hash": chash or rid, "phase": phase, "model": model,
        "split": {"scheme": scheme, "eval": ["2000-06-01", "2002-03-29"]},
        "cost_bps": cost_bps, "n_inner_trials": 0, "results_path": f"results/{rid}/",
        "metrics": {"sharpe_net": sharpe((gross - cost).loc[ev], RF.loc[ev]),
                    "sharpe_gross": sharpe(gross.loc[ev], RF.loc[ev]), "ic_mean": 0.01,
                    "ic_t_nw": 1.0, "oos_r2": None, "turnover_ann": 20.0},
    }  # fmt: skip
    with (root / "e.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return gross, traded


def test_net_sharpe_on_the_cost_grid_is_rebuilt_exactly(tmp_path):
    gross, traded = _run(tmp_path, "r1", "xgb", "expanding", 0)
    t = cost_grid_table(tmp_path / "e.jsonl", tmp_path, RF, [0, 5, 25])
    ev = slice("2000-06-01", "2002-03-29")
    for c in (0, 5, 25):
        expect = sharpe((gross - traded * c / 1e4).loc[ev], RF.loc[ev])
        assert t.loc[0, f"sharpe_net_{c}bps"] == pytest.approx(expect, rel=1e-12)
    assert t.loc[0, "sharpe_net_0bps"] == pytest.approx(t.loc[0, "sharpe_gross"])


def test_break_even_cost_sets_net_excess_return_to_zero(tmp_path):
    gross, traded = _run(tmp_path, "r1", "xgb", "expanding", 0)
    t = cost_grid_table(tmp_path / "e.jsonl", tmp_path, RF, [10])
    c = t.loc[0, "break_even_bps"]
    ev = slice("2000-06-01", "2002-03-29")
    assert sharpe((gross - traded * c / 1e4).loc[ev], RF.loc[ev]) == pytest.approx(0, abs=1e-12)


def test_rows_identified_by_model_and_scheme_and_reruns_deduplicated(tmp_path):
    _run(tmp_path, "a", "ridge", "expanding", 1, chash="h1")
    _run(tmp_path, "b", "ridge", "expanding", 1, chash="h1")  # identical re-run
    _run(tmp_path, "c", "ridge", "rolling", 2)
    _run(tmp_path, "d", "momentum_12_1", "none (rule)", 3)
    _run(tmp_path, "e", "xgb", "expanding", 4, phase="exploratory")  # not a dev result
    t = cost_grid_table(tmp_path / "e.jsonl", tmp_path, RF, [10])
    assert t["label"].tolist() == ["ridge (expanding)", "ridge (rolling)", "momentum_12_1"]
    assert t.loc[0, "run_id"] == "b"  # latest of the identical runs


def test_table_refuses_a_log_that_does_not_match_its_daily_file(tmp_path):
    _run(tmp_path, "r1", "xgb", "expanding", 0)
    rows = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    rows[0]["metrics"]["sharpe_net"] += 0.01  # a hand-edited number
    (tmp_path / "e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="does not match"):
        cost_grid_table(tmp_path / "e.jsonl", tmp_path, RF, [10])


def test_markdown_has_one_line_per_row_and_shows_missing_as_a_dash(tmp_path):
    _run(tmp_path, "r1", "xgb", "expanding", 0)
    md = to_markdown(cost_grid_table(tmp_path / "e.jsonl", tmp_path, RF, [10]))
    lines = md.strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("| ") and "—" in lines[2]
