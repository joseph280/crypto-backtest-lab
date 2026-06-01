"""Volatility-scaled triple-barrier labeling (Marcos Lopez de Prado).

The model does not predict a price; it predicts a label. For each bar t0 we ask:
over the next horizon, does price hit an upper barrier (take-profit, label +1), a
lower barrier (stop-loss, label -1), or neither before a time limit (label 0)?
The barriers are scaled by recent volatility so a label means the same thing in
calm and wild markets. This matches how the bot actually trades (with a stop and
a target) and respects the path, not just the endpoint.

Reference pseudocode (briefing Section 5):

    def triple_barrier_label(prices, t0, up_mult, dn_mult, max_holding, vol):
        upper = prices[t0] * (1 + up_mult * vol[t0])
        lower = prices[t0] * (1 - dn_mult * vol[t0])
        t_end = t0 + max_holding
        for t in range(t0 + 1, min(t_end, len(prices))):
            if prices[t] >= upper: return +1   # take-profit hit first
            if prices[t] <= lower: return -1   # stop hit first
        return 0                                # time limit, no decisive move

This module implements that, vectorized over all t0, and additionally returns the
EXIT INDEX of each label (the bar where the barrier was touched or the time limit
reached). The exit index is the label's horizon end t1, which the validation
harness needs to purge overlapping samples (briefing Section 6).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from cbt_lab.config import Settings, load_settings


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Trailing per-bar volatility (std of one-bar log returns). Causal."""
    ret = np.log(close.astype("float64")).diff(1)
    return ret.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)


def triple_barrier_labels(
    close: pd.Series,
    vol: Optional[pd.Series] = None,
    up_mult: float = 1.5,
    dn_mult: float = 1.5,
    max_holding: int = 24,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    vol_window: int = 50,
    use_intrabar: bool = False,
) -> pd.DataFrame:
    """Label every bar with the triple-barrier method.

    Args:
        close: close price series.
        vol: per-bar volatility (fractional). If None, computed from close with
            a trailing window of vol_window.
        up_mult, dn_mult: barrier widths as multiples of vol.
        max_holding: vertical (time) barrier in bars.
        high, low: optional intrabar extremes; only used if use_intrabar=True.
        use_intrabar: if True, test barriers against the bar high/low (the bar
            can touch a barrier intrabar). If False (default, matching the
            briefing pseudocode), test against the close path only.

    Returns a frame aligned to close.index with columns:
        label: -1.0, 0.0, +1.0, or NaN where the forward window is incomplete.
        touch: 'up' | 'down' | 'time' | '' (empty where label is NaN).
        exit_index: integer position of the exit bar, or -1 if undefined.
        holding_bars: number of bars held to exit, or NaN.
        ret: log return from entry close to exit close, or NaN.
    """
    close = close.astype("float64")
    n = len(close)
    c = close.to_numpy()
    if vol is None:
        vol = realized_vol(close, vol_window)
    v = vol.to_numpy(dtype="float64")

    if use_intrabar and high is not None and low is not None:
        hi = high.astype("float64").to_numpy()
        lo = low.astype("float64").to_numpy()
    else:
        hi = c
        lo = c

    label = np.full(n, np.nan)
    touch = np.array([""] * n, dtype=object)
    exit_index = np.full(n, -1, dtype="int64")
    holding = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    for t0 in range(n):
        v0 = v[t0]
        # Cannot scale barriers without a volatility estimate.
        if not np.isfinite(v0) or v0 <= 0:
            continue
        upper = c[t0] * (1.0 + up_mult * v0)
        lower = c[t0] * (1.0 - dn_mult * v0)
        t_end = min(t0 + max_holding, n - 1)
        # Tail bars without a full forward window get no label (avoid optimistic
        # truncated labels leaking the end of the sample).
        if t0 + max_holding >= n:
            continue
        resolved = False
        for t in range(t0 + 1, t_end + 1):
            up_hit = hi[t] >= upper
            dn_hit = lo[t] <= lower
            if up_hit and dn_hit:
                # Both touched within the same bar: cannot tell order from OHLC,
                # so call it the time-barrier outcome (no decisive move). This is
                # the conservative choice and never invents a winner.
                label[t0], touch[t0] = 0.0, "time"
                exit_index[t0] = t
                resolved = True
                break
            if up_hit:
                label[t0], touch[t0] = 1.0, "up"
                exit_index[t0] = t
                resolved = True
                break
            if dn_hit:
                label[t0], touch[t0] = -1.0, "down"
                exit_index[t0] = t
                resolved = True
                break
        if not resolved:
            label[t0], touch[t0] = 0.0, "time"
            exit_index[t0] = t_end
        ei = exit_index[t0]
        holding[t0] = ei - t0
        ret[t0] = np.log(c[ei] / c[t0])

    return pd.DataFrame(
        {
            "label": label,
            "touch": touch,
            "exit_index": exit_index,
            "holding_bars": holding,
            "ret": ret,
        },
        index=close.index,
    )


def label_matrix(
    matrix: pd.DataFrame, settings: Optional[Settings] = None, use_intrabar: bool = False
) -> pd.DataFrame:
    """Label a feature matrix from features/builder using config parameters.

    Uses the matrix close prices and, if present, its realized_vol column;
    otherwise recomputes volatility. Barrier multipliers and the time barrier are
    read from config.labeling so there are no magic numbers here.
    """
    settings = settings or load_settings()
    lab = settings.labeling
    vol = matrix["realized_vol"] if "realized_vol" in matrix.columns else None
    return triple_barrier_labels(
        close=matrix["close"],
        vol=vol,
        up_mult=lab.up_mult,
        dn_mult=lab.dn_mult,
        max_holding=lab.max_holding_bars,
        high=matrix.get("high"),
        low=matrix.get("low"),
        vol_window=lab.vol_window,
        use_intrabar=use_intrabar,
    )
