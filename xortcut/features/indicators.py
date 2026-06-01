"""Technical indicators: feature family A (briefing Section 5).

These are table stakes, not the edge: returns over several lookbacks, momentum,
RSI, MACD, distance from moving averages, ATR, and Bollinger position. Everyone
has them. They are included so the model has the commodity context, but the
effort and the edge live in the microstructure family.

All indicators here are causal (rolling, trailing). We use the `ta` library
rather than pandas-ta; see docs/build-notes.md for why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

# Lookbacks are hyperparameters, not venue thresholds, so they live here as
# documented defaults rather than in settings.yaml.
RETURN_LOOKBACKS = (1, 2, 3, 6, 12, 24)
MA_WINDOWS = (12, 24, 48)


def technical_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Build the technical feature family from an OHLCV frame.

    candles must have columns: open, high, low, close, volume. The returned
    frame is indexed like candles and contains only past/current information at
    each row.
    """
    close = candles["close"].astype("float64")
    high = candles["high"].astype("float64")
    low = candles["low"].astype("float64")
    volume = candles["volume"].astype("float64")

    out = pd.DataFrame(index=candles.index)

    # Log returns over several lookbacks, plus simple momentum.
    log_close = np.log(close)
    for k in RETURN_LOOKBACKS:
        out[f"ret_{k}"] = log_close.diff(k)
        out[f"mom_{k}"] = close / close.shift(k) - 1.0

    # Distance from moving averages (trend context).
    for w in MA_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
        out[f"dist_sma_{w}"] = close / sma - 1.0
        out[f"dist_ema_{w}"] = close / ema - 1.0

    # RSI.
    out["rsi_14"] = RSIIndicator(close=close, window=14, fillna=False).rsi()

    # MACD (line, signal, and the histogram difference).
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # ATR normalized by price (scale-free volatility).
    atr = AverageTrueRange(high=high, low=low, close=close, window=14, fillna=False)
    out["atr_norm_14"] = atr.average_true_range() / close

    # Bollinger position (%B): where price sits inside its bands.
    bb = BollingerBands(close=close, window=20, window_dev=2, fillna=False)
    width = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0.0, np.nan)
    out["bb_pct_b"] = (close - bb.bollinger_lband()) / width
    out["bb_bandwidth"] = width / close

    # Volume context (trailing, scale-free).
    vol_ma = volume.rolling(24, min_periods=24).mean().replace(0.0, np.nan)
    out["vol_ratio_24"] = volume / vol_ma

    return out
