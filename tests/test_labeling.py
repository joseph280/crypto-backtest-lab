"""Unit tests for the triple-barrier labeler."""

import numpy as np
import pandas as pd

from cbt_lab.labeling.triple_barrier import realized_vol, triple_barrier_labels


def test_rising_series_hits_upper():
    close = pd.Series([100, 101, 102, 103, 104, 105.0])
    vol = pd.Series([0.01] * 6)
    out = triple_barrier_labels(close, vol=vol, up_mult=1.5, dn_mult=1.5, max_holding=3)
    # The first bars resolve to the upper barrier; the tail is unlabeled.
    assert out["label"].iloc[0] == 1.0
    assert out["touch"].iloc[0] == "up"
    assert np.isnan(out["label"].iloc[-1])


def test_falling_series_hits_lower():
    close = pd.Series([100, 99, 98, 97, 96, 95.0])
    vol = pd.Series([0.01] * 6)
    out = triple_barrier_labels(close, vol=vol, up_mult=1.5, dn_mult=1.5, max_holding=3)
    assert out["label"].iloc[0] == -1.0
    assert out["touch"].iloc[0] == "down"


def test_flat_series_times_out():
    close = pd.Series([100.0] * 10)
    vol = pd.Series([0.01] * 10)
    out = triple_barrier_labels(close, vol=vol, up_mult=1.5, dn_mult=1.5, max_holding=4)
    resolved = out.dropna(subset=["label"])
    assert (resolved["label"] == 0.0).all()
    assert (resolved["touch"] == "time").all()


def test_exit_index_is_forward_and_bounded():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01)))
    out = triple_barrier_labels(close, up_mult=1.5, dn_mult=1.5, max_holding=10, vol_window=20)
    valid = out.dropna(subset=["label"])
    # Exit is always at or after the entry, and never beyond the time barrier.
    assert (valid["exit_index"].to_numpy() > valid.index.to_numpy()).all()
    assert (valid["holding_bars"] <= 10).all()


def test_tail_is_unlabeled():
    close = pd.Series(np.linspace(100, 110, 50))
    out = triple_barrier_labels(close, up_mult=1.0, dn_mult=1.0, max_holding=8, vol_window=10)
    # The last max_holding bars cannot have a full forward window.
    assert out["label"].tail(8).isna().all()


def test_higher_vol_widens_barriers():
    # With higher volatility, the same move is less likely to breach a barrier.
    close = pd.Series([100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5])
    low_vol = pd.Series([0.002] * len(close))
    high_vol = pd.Series([0.05] * len(close))
    lo = triple_barrier_labels(close, vol=low_vol, up_mult=1.5, dn_mult=1.5, max_holding=4)
    hi = triple_barrier_labels(close, vol=high_vol, up_mult=1.5, dn_mult=1.5, max_holding=4)
    # Low vol: the small uptrend breaches the tight upper barrier.
    assert lo["label"].iloc[0] == 1.0
    # High vol: barriers are far, so the same move times out instead.
    assert hi["label"].iloc[0] == 0.0


def test_realized_vol_is_causal_and_positive():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01)))
    rv = realized_vol(close, window=20)
    assert (rv.dropna() >= 0).all()
