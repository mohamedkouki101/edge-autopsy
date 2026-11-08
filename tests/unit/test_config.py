"""P0.5 acceptance: config hash is stable and sensitive; configs are strictly validated."""

from pathlib import Path

import pytest
import yaml

from edge_autopsy.config import ExperimentCfg, config_hash, load_config

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "base.yaml"


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def test_config_hash_stable():
    assert config_hash(load_config(BASE)) == config_hash(load_config(BASE))


def test_config_hash_ignores_key_order(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    reordered = dict(reversed(list(raw.items())))
    p = _write(tmp_path, "reordered.yaml", reordered)
    assert config_hash(load_config(p)) == config_hash(load_config(BASE))


def test_config_hash_changes_with_one_field(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["costs"]["spread_bps"] = raw["costs"]["spread_bps"] + 1
    p = _write(tmp_path, "changed.yaml", raw)
    assert config_hash(load_config(p)) != config_hash(load_config(BASE))


def test_experiment_inherits_base_and_overrides_only_what_it_changes(tmp_path):
    exp = _write(
        tmp_path, "exp.yaml", {"inherits": str(BASE), "name": "x", "costs": {"spread_bps": 0}}
    )
    cfg = load_config(exp)
    base = load_config(BASE)
    assert cfg.costs.spread_bps == 0
    assert cfg.costs.slippage_bps == base.costs.slippage_bps  # untouched sibling kept
    assert cfg.seed == base.seed
    assert config_hash(cfg) != config_hash(base)


def test_unknown_key_rejected(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["costs"]["sprad_bps"] = 3  # typo must not be silently ignored
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "typo.yaml", raw))


def test_holdout_must_start_after_dev_period(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["split"]["holdout_start"] = "2010-01-04"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "bad.yaml", raw))


def test_base_matches_plan_values():
    c = load_config(BASE)
    assert isinstance(c, ExperimentCfg)
    assert c.seed == 20261106
    assert (c.horizon, c.lag) == (5, 1)
    assert c.label == "holding_period"  # decision 2026-11-08 (Option B)
    assert str(c.split.holdout_start) == "2017-01-02"
    assert (c.split.purge_days, c.split.embargo_weeks, c.split.refit_every) == (6, 1, 52)
    assert c.costs.spread_bps + c.costs.slippage_bps == 10
    assert c.costs.borrow_bps == 25
    assert c.costs.grid_bps == [0, 2, 5, 10, 25, 50]


def test_label_kind_is_validated(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["label"] = "five_day_ish"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "bad_label.yaml", raw))


def test_purge_must_cover_the_whole_label_window(tmp_path):
    """K3: a label ends up to lag + horizon trading days after t; a shorter purge leaks."""
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["horizon"] = 10  # purge_days stays 6 < 1 + 10
    with pytest.raises(ValueError, match="purge_days"):
        load_config(_write(tmp_path, "short_purge.yaml", raw))
