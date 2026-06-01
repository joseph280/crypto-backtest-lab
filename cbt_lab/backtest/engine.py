"""Vectorbt backtest engine with realistic costs (briefing Section 6).

The engine runs a target-position series (one of -1, 0, +1 per bar, decided at
that bar's close from causal features) through vectorbt for fee and slippage
realism, then layers hourly funding on top via backtest/costs.py. The same
per-bar net return series it produces is exactly what the validation harness
consumes, so backtest and validation share one definition of "return".

This is the engine only. It does not decide positions; strategy modules do that
later. Stage 0 feeds it sample positions purely to prove the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from cbt_lab.backtest.costs import CostModel, funding_return
from cbt_lab.config import Settings, load_settings
from cbt_lab.data.collectors import interval_to_ms


def position_to_signals(position: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Convert a target-position series in {-1, 0, +1} into the four vectorbt
    signal arrays (long entries/exits, short entries/exits)."""
    pos = position.astype("float64").fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    long_entries = (pos == 1) & (prev != 1)
    long_exits = (prev == 1) & (pos != 1)
    short_entries = (pos == -1) & (prev != -1)
    short_exits = (prev == -1) & (pos != -1)
    return long_entries, long_exits, short_entries, short_exits


@dataclass
class BacktestResult:
    net_returns: pd.Series       # per-bar return net of fees, slippage, and funding
    market_returns: pd.Series    # vectorbt returns (fees + slippage, no funding)
    funding_returns: pd.Series   # funding contribution per bar
    n_trades: int
    total_return: float
    portfolio: object            # the vectorbt Portfolio, for further inspection

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "total_return": self.total_return,
            "mean_bar_return": float(self.net_returns.mean()),
            "funding_total": float(self.funding_returns.sum()),
        }


def run_backtest(
    close: pd.Series,
    position: pd.Series,
    interval: str = "1h",
    funding_rate: Optional[pd.Series] = None,
    settings: Optional[Settings] = None,
    prefer_maker: Optional[bool] = None,
    init_cash: float = 100.0,
) -> BacktestResult:
    """Run one target-position series through vectorbt with realistic costs.

    close, position (and funding_rate, if given) must share an index. The fee
    used is the maker fee when maker/post-only is preferred (config.risk), else
    the taker fee.
    """
    settings = settings or load_settings()
    cost = CostModel.from_settings(settings, prefer_maker=prefer_maker)
    close = close.astype("float64").reset_index(drop=True)
    position = position.reset_index(drop=True)

    le, lx, se, sx = position_to_signals(position)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        fees=cost.fee_frac,
        slippage=cost.slippage_frac,
        init_cash=init_cash,
        freq=interval,
    )
    market_returns = portfolio.returns().reset_index(drop=True)

    if funding_rate is not None:
        fr = funding_rate.reset_index(drop=True)
        interval_seconds = interval_to_ms(interval) / 1000.0
        funding_returns = funding_return(position, fr, interval_seconds, cost.funding_settlement)
    else:
        funding_returns = pd.Series(0.0, index=market_returns.index)

    net_returns = market_returns.add(funding_returns, fill_value=0.0)

    try:
        n_trades = int(portfolio.trades.count())
    except Exception:
        n_trades = int(((position.fillna(0).diff().abs()) > 0).sum())

    return BacktestResult(
        net_returns=net_returns,
        market_returns=market_returns,
        funding_returns=funding_returns,
        n_trades=n_trades,
        total_return=float((1.0 + net_returns).prod() - 1.0),
        portfolio=portfolio,
    )
