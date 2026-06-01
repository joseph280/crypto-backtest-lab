"""Realistic cost model: fees, slippage, and funding (briefing Sections 6 and 9).

A backtest that ignores both-leg slippage and funding is fiction. The numbers
here all come from config.backtest (fee_taker_pct 0.045, fee_maker_pct 0.015,
slippage_bps, funding_settlement hourly), never hardcoded.

Two of the three (fees, slippage) are modelled by the vectorbt engine natively.
Funding is not something vectorbt models, so it is computed here and layered onto
the engine's per-bar returns. Funding is the structural edge and a cost at once:
holding the unpopular side collects it; holding the crowded side pays it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from cbt_lab.config import Settings, load_settings


@dataclass
class CostModel:
    fee_frac: float          # per-side trade fee as a fraction of notional
    slippage_frac: float     # per-side slippage as a fraction of notional
    funding_settlement: str  # 'hourly' on Hyperliquid

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None, prefer_maker: Optional[bool] = None) -> "CostModel":
        settings = settings or load_settings()
        bt = settings.backtest
        use_maker = settings.risk.prefer_maker if prefer_maker is None else prefer_maker
        fee = bt.fee_maker_frac if use_maker else bt.fee_taker_frac
        return cls(fee_frac=fee, slippage_frac=bt.slippage_frac, funding_settlement=bt.funding_settlement)


def turnover(position: pd.Series) -> pd.Series:
    """Absolute change in target position per bar (units of notional traded)."""
    pos = position.astype("float64")
    change = pos.diff()
    change.iloc[0] = pos.iloc[0]  # opening the first position is a trade
    return change.abs()


def trade_cost_returns(position: pd.Series, cost: CostModel) -> pd.Series:
    """Per-bar return drag from fees and slippage at the bars where the position
    changes. Provided for a transparent manual accounting path; the vectorbt
    engine applies fees and slippage itself, so do not add this on top of it."""
    return turnover(position) * (cost.fee_frac + cost.slippage_frac)


def funding_return(
    position: pd.Series,
    funding_rate: pd.Series,
    interval_seconds: float,
    settlement: str = "hourly",
) -> pd.Series:
    """Per-bar funding PnL as a fraction of notional.

    A long pays funding when the rate is positive (longs pay shorts), so the
    funding return to a position is -position * rate. The hourly rate is prorated
    to the bar length so that, for example, four 15m bars accrue one hourly
    settlement rather than four.
    """
    pos = position.astype("float64").fillna(0.0)
    rate = funding_rate.reindex(pos.index).astype("float64").fillna(0.0)
    factor = (interval_seconds / 3600.0) if settlement == "hourly" else 1.0
    return -(pos * rate * factor)
