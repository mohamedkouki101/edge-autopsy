"""Typed configuration (PLAN §4 "Config system").

`configs/base.yaml` holds every global choice. Experiment YAMLs set `inherits: <path>` and
override only what they change (deep merge). Every model is strict (`extra="forbid"`) so a
typo raises instead of silently falling back to a default. `config_hash()` is the sha256 of
the canonical JSON of the validated config, so key order and YAML formatting don't matter.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataCfg(_Strict):
    french_base_url: str
    french_files: dict[str, str]  # logical name -> zip file name on the French site
    etf_tickers: list[str]
    etf_benchmark: str
    etf_start: dt.date
    etf_first_date_max: dt.date  # each ETF must trade on or before this date (D4)
    sample_floor: dt.date  # sample may not start before this (D3 decision, 2026-11-07)
    sample_start: dt.date | None  # first rebalance >= floor with >= min_industries (D3)
    sample_end: dt.date
    min_industries: int = Field(ge=1)
    entry_days: int = Field(ge=1)  # asset available from its Nth non-missing day (PLAN §5.3.4)
    max_abs_ret_flag: float = Field(gt=0)


class FeatureCfg(_Strict):
    momentum_windows: list[int]
    mom_12_1: tuple[int, int]  # (lookback, skip)
    vol_windows: list[int]
    beta_window: int
    mdd_window: int
    skew_window: int
    market_window: int


class SplitCfg(_Strict):
    scheme: Literal["expanding", "rolling"]
    dev_test_start: dt.date
    dev_test_end: dt.date
    holdout_start: dt.date
    holdout_end: dt.date
    min_train_years: int
    rolling_years: int
    refit_every: int
    purge_days: int
    embargo_weeks: int
    inner_folds: int
    inner_fold_years: int

    @model_validator(mode="after")
    def _ordered(self) -> SplitCfg:
        if not (self.dev_test_start < self.dev_test_end < self.holdout_start <= self.holdout_end):
            raise ValueError(
                "need dev_test_start < dev_test_end < holdout_start <= holdout_end; got "
                f"{self.dev_test_start}, {self.dev_test_end}, {self.holdout_start}, "
                f"{self.holdout_end}"
            )
        return self


class PortfolioCfg(_Strict):
    kind: Literal["long_only", "long_short"]
    quantile: float = Field(gt=0, le=0.5)
    weighting: Literal["equal", "inverse_vol"]
    band: tuple[int, int] | None  # (enter rank <=, exit rank >)


class CostCfg(_Strict):
    spread_bps: float = Field(ge=0)  # half-spread per side
    slippage_bps: float = Field(ge=0)
    borrow_bps: float = Field(ge=0)  # annual fee on short notional
    grid_bps: list[float]


class ExperimentCfg(_Strict):
    name: str
    universe: Literal["ff49", "etf9"]
    seed: int
    rebalance_freq: str
    horizon: int = Field(ge=1)
    lag: int = Field(ge=0)
    # holding_period: fill-to-next-fill return (max `horizon` days); fixed_horizon: always
    # `horizon` days after the fill. Decision 2026-11-08 (docs/DECISIONS.md).
    label: Literal["holding_period", "fixed_horizon"]
    data: DataCfg
    features: FeatureCfg
    split: SplitCfg
    portfolio: PortfolioCfg
    costs: CostCfg


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = Path(path).parent / parent_path
    return _deep_merge(_load_raw(parent_path), raw)


def load_config(path: str | Path) -> ExperimentCfg:
    """Load a YAML config (following `inherits:` chains) and validate it."""
    return ExperimentCfg.model_validate(_load_raw(Path(path)))


def config_hash(cfg: BaseModel) -> str:
    """sha256 of the canonical JSON of a validated config (key-order independent)."""
    canon = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
