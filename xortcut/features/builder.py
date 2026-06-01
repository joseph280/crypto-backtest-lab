"""Assemble the feature matrix from the parquet lake (briefing Sections 4 and 5).

The matrix combines three families:
  A. Technical (commodity, table stakes) from features/indicators.py
  B. Microstructure (the differentiator, where effort goes) from data/microstructure.py
  C. Context: realized volatility, time-of-day and day-of-week, and BTC used as
     a feature when trading ETH.

Leakage discipline (briefing Section 6):
  - Microstructure series (funding, OI, basis) are aligned to each bar with a
    backward as-of join keyed on the bar's CLOSE time, so a bar only ever sees
    values known by its own close.
  - Every transform is causal (rolling/trailing, never centered or full-sample).
  - The result therefore satisfies tests/test_no_leakage.py: the feature row at
    time t is identical whether computed on data up to t or on the full series.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from xortcut.config import Settings, load_settings
from xortcut.data import microstructure as ms
from xortcut.data.collectors import asset_ctx_path, candles_path, funding_path
from xortcut.features.indicators import technical_features

# Columns that are raw market data or bookkeeping, not model features.
META_COLUMNS = ["open_time", "close_time", "timestamp", "open", "high", "low", "close", "volume", "trades"]


def feature_columns(df: pd.DataFrame) -> List[str]:
    """The model feature columns: everything that is not a META column."""
    return [c for c in df.columns if c not in META_COLUMNS]


# ----------------------------------------------------------------- loaders


def load_candles(coin: str, interval: str, settings: Settings, root: Optional[Path] = None) -> pd.DataFrame:
    path = candles_path(coin, interval, settings, root)
    if not path.exists():
        raise FileNotFoundError(f"No candles for {coin}/{interval} at {path}. Run scripts/pull_data.py first.")
    return pd.read_parquet(path).sort_values("open_time").reset_index(drop=True)


def _load_optional(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    return None


# ----------------------------------------------------------------- causal alignment


def _asof_align(candles: pd.DataFrame, source: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """Backward as-of join: for each bar, take the latest source row whose time
    is at or before the bar's CLOSE time. This is the causal alignment: a bar
    never sees a value timestamped after its close.
    """
    left = candles[["close_time"]].copy()
    right = source[["time"] + value_cols].copy()
    merged = pd.merge_asof(
        left.sort_values("close_time"),
        right.sort_values("time"),
        left_on="close_time",
        right_on="time",
        direction="backward",
    )
    return merged[value_cols].reset_index(drop=True)


# ----------------------------------------------------------------- context


def _context_features(candles: pd.DataFrame, ret_1: pd.Series, vol_window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=candles.index)
    # Realized volatility: trailing std of one-bar log returns.
    out["realized_vol"] = ret_1.rolling(vol_window, min_periods=max(2, vol_window // 2)).std(ddof=0)
    # Cyclical time-of-day and day-of-week (crypto has session patterns).
    ts = candles["timestamp"].dt
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


# ----------------------------------------------------------------- core builder


def build_features_from_frames(
    candles: pd.DataFrame,
    funding: Optional[pd.DataFrame] = None,
    asset_ctx: Optional[pd.DataFrame] = None,
    btc_candles: Optional[pd.DataFrame] = None,
    settings: Optional[Settings] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix from in-memory frames (pure, testable).

    Returns a frame indexed by integer position with META_COLUMNS (including
    close, for labeling and backtest) plus all feature columns.
    """
    settings = settings or load_settings()
    candles = candles.sort_values("open_time").reset_index(drop=True)
    vol_window = settings.labeling.vol_window

    parts: List[pd.DataFrame] = [candles[META_COLUMNS].copy()]

    # Family A: technical.
    if settings.features.technical:
        parts.append(technical_features(candles))

    one_bar_ret = np.log(candles["close"].astype("float64")).diff(1)

    # Family B: microstructure (weight of effort).
    if settings.features.microstructure:
        if funding is not None and not funding.empty:
            aligned = _asof_align(candles, funding, ["funding_rate"])
            parts.append(ms.funding_features(aligned["funding_rate"]))
        if asset_ctx is not None and not asset_ctx.empty:
            cols = [c for c in ["open_interest", "mark_px", "oracle_px"] if c in asset_ctx.columns]
            aligned = _asof_align(candles, asset_ctx, cols)
            if "open_interest" in aligned:
                parts.append(ms.open_interest_features(aligned["open_interest"]))
            if {"mark_px", "oracle_px"}.issubset(aligned.columns):
                parts.append(ms.basis_features(aligned["mark_px"], aligned["oracle_px"]))

    # Family C: context.
    if settings.features.context:
        parts.append(_context_features(candles, one_bar_ret, vol_window))
        # BTC as a feature for ETH (and any non-BTC core symbol).
        if btc_candles is not None and not btc_candles.empty:
            btc = btc_candles.sort_values("open_time").reset_index(drop=True)
            btc_ret = np.log(btc["close"].astype("float64")).diff(1)
            btc_frame = pd.DataFrame(
                {"open_time": btc["open_time"], "btc_ret_1": btc_ret,
                 "btc_vol": btc_ret.rolling(vol_window, min_periods=max(2, vol_window // 2)).std(ddof=0)}
            )
            merged = candles[["open_time"]].merge(btc_frame, on="open_time", how="left")
            parts.append(merged[["btc_ret_1", "btc_vol"]])

    matrix = pd.concat(parts, axis=1)
    if dropna:
        feat_cols = feature_columns(matrix)
        matrix = matrix.dropna(subset=feat_cols).reset_index(drop=True)
    return matrix


def build_features(
    coin: str,
    interval: str,
    settings: Optional[Settings] = None,
    root: Optional[Path] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix for one coin and interval from the parquet lake."""
    settings = settings or load_settings()
    candles = load_candles(coin, interval, settings, root)
    funding = _load_optional(funding_path(coin, settings, root))
    asset_ctx = _load_optional(asset_ctx_path(coin, settings, root))

    btc_candles = None
    if settings.features.context and coin != "BTC" and "BTC" in settings.symbols.core:
        btc_path = candles_path("BTC", interval, settings, root)
        if btc_path.exists():
            btc_candles = pd.read_parquet(btc_path)

    return build_features_from_frames(
        candles, funding=funding, asset_ctx=asset_ctx, btc_candles=btc_candles, settings=settings, dropna=dropna
    )
