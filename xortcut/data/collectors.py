"""Parquet data-lake layout: paths and interval helpers.

This demo runs entirely on the synthetic lake (see data/synthetic.py), so the
only collector code it needs is the path layout and the interval-to-milliseconds
mapping. The lake is partitioned by symbol and interval and stored as parquet,
with all timestamps kept in a single timezone (UTC).

The live incremental collectors (venue REST pagination, gap checks, forward
open-interest snapshots) are part of the full project and are intentionally left
out of this offline demo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from xortcut.config import Settings

# Interval string to milliseconds. Covers the common perp candle intervals.
_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


def interval_to_ms(interval: str) -> int:
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}. Known: {sorted(_INTERVAL_MS)}")
    return _INTERVAL_MS[interval]


def now_ms() -> int:
    return int(time.time() * 1000)


# ----------------------------------------------------------------- paths


def candles_path(coin: str, interval: str, settings: Settings, root: Optional[Path] = None) -> Path:
    return settings.data.storage_dir(root) / "candles" / coin / f"{interval}.parquet"


def funding_path(coin: str, settings: Settings, root: Optional[Path] = None) -> Path:
    return settings.data.storage_dir(root) / "funding" / f"{coin}.parquet"


def asset_ctx_path(coin: str, settings: Settings, root: Optional[Path] = None) -> Path:
    return settings.data.storage_dir(root) / "asset_ctx" / f"{coin}.parquet"
