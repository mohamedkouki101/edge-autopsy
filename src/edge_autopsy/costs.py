"""Linear cost model (PLAN §5.6, task B3).

- Trading cost per side = half-spread + slippage, applied to traded notional sum|dw|.
- Borrow fee on short notional accrues daily at fee / 252.
Market impact beyond slippage is not modelled (limitation, PLAN §8.5).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from edge_autopsy.config import CostCfg

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    spread_bps: float  # half-spread, per side
    slippage_bps: float  # per side
    borrow_bps: float  # annual fee on short notional

    def __post_init__(self) -> None:
        if min(self.spread_bps, self.slippage_bps, self.borrow_bps) < 0:
            raise ValueError("cost parameters must be >= 0")

    @property
    def per_side_bps(self) -> float:
        return self.spread_bps + self.slippage_bps

    def trade_cost(self, traded: float) -> float:
        """Cost as a fraction of NAV for traded notional sum|dw| (fraction of NAV)."""
        if traded < 0:
            raise ValueError("traded notional must be >= 0")
        return traded * self.per_side_bps / 1e4

    def borrow_cost(self, short_notional: float) -> float:
        """One day's borrow fee for short notional (fraction of NAV, positive number)."""
        if short_notional < 0:
            raise ValueError("short notional must be >= 0")
        return short_notional * self.borrow_bps / 1e4 / TRADING_DAYS

    def with_per_side(self, bps: float) -> CostModel:
        """Same borrow fee, total per-side cost `bps` (used for the cost grid)."""
        return replace(self, spread_bps=bps, slippage_bps=0.0)

    @classmethod
    def from_cfg(cls, cfg: CostCfg) -> CostModel:
        return cls(cfg.spread_bps, cfg.slippage_bps, cfg.borrow_bps)
