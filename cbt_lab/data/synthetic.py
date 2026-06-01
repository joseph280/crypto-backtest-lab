"""Deterministic synthetic data for offline development and harness proof.

The validation harness and backtest must be runnable without a live venue
connection, both for fast iteration and so the Stage 0 trust numbers can be
demonstrated on a known input. This module fabricates candles and funding in the
exact parquet schema the collectors produce.

This is clearly labelled synthetic. It is not market data and carries no edge.
It exists only to exercise the pipeline plumbing end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from cbt_lab.config import Settings, load_settings
from cbt_lab.data.collectors import candles_path, funding_path, interval_to_ms


def _coin_seed(coin: str) -> int:
    return abs(hash(coin)) % (2**31)


def generate_candles(
    coin: str, interval: str, n_bars: int, end_ms: int, start_price: float = 100.0, seed: int = 0
) -> pd.DataFrame:
    """A seeded geometric random walk with mild volatility clustering and a
    daily seasonal in volume. Reproducible for a given (coin, interval, seed).
    """
    rng = np.random.default_rng(seed + _coin_seed(coin) + interval_to_ms(interval))
    step = interval_to_ms(interval)
    open_time = end_ms - n_bars * step + np.arange(n_bars) * step

    # Volatility clustering via a slow-moving sigma, returns from it.
    base_sigma = 0.004
    vol_state = base_sigma * (1.0 + 0.5 * np.sin(np.linspace(0, 12 * np.pi, n_bars)))
    shocks = rng.standard_normal(n_bars)
    log_ret = vol_state * shocks
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]
    intrabar = np.abs(rng.standard_normal(n_bars)) * vol_state * close
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar

    hour = (pd.to_datetime(open_time, unit="ms", utc=True).hour).to_numpy()
    seasonal = 1.0 + 0.4 * np.cos((hour - 14) / 24.0 * 2 * np.pi)
    volume = np.abs(rng.gamma(2.0, 50.0, n_bars)) * seasonal
    trades = np.maximum(1, (volume / 5.0).astype("int64"))

    df = pd.DataFrame(
        {
            "open_time": open_time.astype("int64"),
            "close_time": (open_time + step - 1).astype("int64"),
            "timestamp": pd.to_datetime(open_time, unit="ms", utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trades": trades,
        }
    )
    return df


def generate_funding(coin: str, n_hours: int, end_ms: int, seed: int = 0) -> pd.DataFrame:
    """Hourly, mean-reverting funding around a small positive level."""
    rng = np.random.default_rng(seed + _coin_seed(coin) + 7)
    step = 3_600_000
    t = end_ms - n_hours * step + np.arange(n_hours) * step
    rate = np.zeros(n_hours)
    mean = 1.0e-5  # roughly 0.001 percent hourly, mildly positive
    theta, sigma = 0.05, 2.0e-5
    for i in range(1, n_hours):
        rate[i] = rate[i - 1] + theta * (mean - rate[i - 1]) + sigma * rng.standard_normal()
    df = pd.DataFrame(
        {
            "time": t.astype("int64"),
            "timestamp": pd.to_datetime(t, unit="ms", utc=True),
            "funding_rate": rate,
            "premium": rate * 0.5 + sigma * rng.standard_normal(n_hours),
        }
    )
    return df


def generate_lake(
    settings: Optional[Settings] = None,
    coins: Optional[List[str]] = None,
    days: Optional[int] = None,
    root: Optional[Path] = None,
    end_ms: Optional[int] = None,
    seed: int = 0,
) -> None:
    """Populate the parquet lake with synthetic candles and funding."""
    settings = settings or load_settings()
    coins = coins or settings.symbols.core
    days = days or settings.data.history_days
    # Fixed default end so runs are reproducible (2026-01-01 UTC) unless overridden.
    end = end_ms if end_ms is not None else 1_767_225_600_000
    start_prices = {"BTC": 72_000.0, "ETH": 2_000.0}

    for coin in coins:
        sp = start_prices.get(coin, 100.0)
        n_hours = days * 24
        fund = generate_funding(coin, n_hours, end, seed)
        fpath = funding_path(coin, settings, root)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fund.to_parquet(fpath, index=False)

        for interval in settings.data.intervals:
            step = interval_to_ms(interval)
            n_bars = max(2, days * 86_400_000 // step)
            cdf = generate_candles(coin, interval, int(n_bars), end, start_price=sp, seed=seed)
            cpath = candles_path(coin, interval, settings, root)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cdf.to_parquet(cpath, index=False)
