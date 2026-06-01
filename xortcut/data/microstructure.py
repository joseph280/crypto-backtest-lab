"""Perp microstructure features: the information edge (briefing Section 5, family B).

This is where the effort goes. These are the crypto-native, under-crowded signals
that justify the whole approach: funding and its trend, open-interest change, the
basis between perp and spot/mark, order-book imbalance, taker flow, and
liquidation activity. Most retail bots ignore these in favour of chart
indicators; Xortcut reads microstructure.

Causality is non-negotiable. Every function here is causal: the value at row t
uses only rows at or before t. In particular, all standardization is rolling
(never full-sample), because a full-sample mean or std would leak the future
into the present. This is exactly what tests/test_no_leakage.py checks.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _rolling_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score using only past and current observations.

    Full-sample standardization would leak future information, so we never use
    s.mean()/s.std() over the whole series here.
    """
    mp = min_periods if min_periods is not None else window
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


# ----------------------------------------------------------------- funding

def funding_features(
    funding_rate: pd.Series, short: int = 8, long: int = 24, z_window: int = 72
) -> pd.DataFrame:
    """Funding level, trend, accrual, and a rolling z-score.

    funding_rate is the per-settlement (hourly on Hyperliquid) rate aligned to
    each bar, as known at that bar's close. Extreme funding flags a one-sided,
    over-leveraged crowd, which often precedes reversals.
    """
    f = funding_rate.astype("float64")
    out = pd.DataFrame(index=f.index)
    out["funding_rate"] = f
    # Short and long trailing averages and their difference (the trend).
    out["funding_ma_short"] = f.rolling(short, min_periods=1).mean()
    out["funding_ma_long"] = f.rolling(long, min_periods=1).mean()
    out["funding_trend"] = out["funding_ma_short"] - out["funding_ma_long"]
    # Carry accrued over a trailing window (sum of rates paid/received).
    out["funding_cum_short"] = f.rolling(short, min_periods=1).sum()
    out["funding_cum_long"] = f.rolling(long, min_periods=1).sum()
    # How extreme is funding relative to its own recent distribution.
    out["funding_z"] = _rolling_z(f, z_window)
    # Sign persistence: how long has the crowd leaned one way recently.
    out["funding_sign_mean"] = np.sign(f).rolling(long, min_periods=1).mean()
    return out


# ----------------------------------------------------------------- open interest

def open_interest_features(open_interest: pd.Series, change_k: int = 1, z_window: int = 72) -> pd.DataFrame:
    """Open-interest level change and trend.

    Rising OI into a price move signals conviction; falling OI signals
    unwinding. Only available where forward OI snapshots have been collected
    (no historical REST endpoint), so this is empty when OI is absent.
    """
    oi = open_interest.astype("float64")
    out = pd.DataFrame(index=oi.index)
    out["oi"] = oi
    out["oi_change"] = oi.pct_change(change_k)
    out["oi_log_change"] = np.log(oi).diff(change_k)
    out["oi_z"] = _rolling_z(oi, z_window)
    return out


# ----------------------------------------------------------------- basis

def basis_features(
    mark_px: pd.Series, oracle_px: pd.Series, z_window: int = 72
) -> pd.DataFrame:
    """Basis between perp (mark) and spot/oracle, plus its trend.

    A positive basis means the perp trades rich to spot (crowd is long).
    """
    mark = mark_px.astype("float64")
    oracle = oracle_px.astype("float64")
    out = pd.DataFrame(index=mark.index)
    basis = (mark - oracle) / oracle.replace(0.0, np.nan)
    out["basis"] = basis
    out["basis_ma"] = basis.rolling(z_window, min_periods=1).mean()
    out["basis_z"] = _rolling_z(basis, z_window)
    return out


# ----------------------------------------------------------------- order book

def book_imbalance(bid_sizes: List[float], ask_sizes: List[float], depth: int = 5) -> float:
    """Top-of-book depth imbalance in [-1, 1] from one L2 snapshot.

    Positive means more bid depth than ask depth near the top of book. Computed
    per snapshot at decision time; forward-collected, not backfillable.
    """
    b = float(np.sum(bid_sizes[:depth]))
    a = float(np.sum(ask_sizes[:depth]))
    total = b + a
    if total <= 0:
        return 0.0
    return (b - a) / total


def book_imbalance_from_l2(l2_snapshot: dict, depth: int = 5) -> float:
    """Compute book imbalance from a hyperliquid l2Book snapshot dict."""
    levels = l2_snapshot.get("levels", [[], []])
    bids = [float(lvl["sz"]) for lvl in levels[0]]
    asks = [float(lvl["sz"]) for lvl in levels[1]]
    return book_imbalance(bids, asks, depth)


# ----------------------------------------------------------------- taker flow

def taker_flow_imbalance(buy_volume: pd.Series, sell_volume: pd.Series, window: int = 12) -> pd.DataFrame:
    """Aggressive buy versus aggressive sell volume imbalance.

    Needs a trades feed (taker side), forward-collected. Positive means buyers
    are lifting offers more than sellers are hitting bids.
    """
    buy = buy_volume.astype("float64")
    sell = sell_volume.astype("float64")
    out = pd.DataFrame(index=buy.index)
    total = (buy + sell).replace(0.0, np.nan)
    out["taker_imbalance"] = (buy - sell) / total
    out["taker_imbalance_ma"] = out["taker_imbalance"].rolling(window, min_periods=1).mean()
    return out


# ----------------------------------------------------------------- liquidations

def liquidation_features(liq_notional: pd.Series, window: int = 12, z_window: int = 72) -> pd.DataFrame:
    """Recent forced-liquidation intensity, which often marks local extremes.

    Needs a liquidation feed, forward-collected.
    """
    liq = liq_notional.astype("float64").fillna(0.0)
    out = pd.DataFrame(index=liq.index)
    out["liq_notional"] = liq
    out["liq_intensity"] = liq.rolling(window, min_periods=1).sum()
    out["liq_z"] = _rolling_z(liq, z_window)
    return out
