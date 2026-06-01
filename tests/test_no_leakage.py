"""Leakage test: fail if any feature uses information from at or after the moment
being predicted (briefing Section 6, the anti-leakage rule).

Method (the rigorous one): a causal feature value at time t depends only on data
up to and including t. So if we recompute the feature matrix on the data
truncated at t, the row at t must be identical to the row at t in the matrix
computed on the full series. If a feature peeks into the future, adding future
rows changes its value at t, and the values diverge. We assert they do not.

The final test is a self-check: it injects a deliberately leaky feature
(tomorrow's price) and asserts that the very same comparison flags it, proving
the detector actually detects.
"""

import numpy as np
import pandas as pd
import pytest

from xortcut.config import load_settings
from xortcut.data.synthetic import generate_candles, generate_funding
from xortcut.features.builder import build_features_from_frames, feature_columns

TOL = 1e-9


def _frames(n_bars=400):
    end = 1_767_225_600_000
    candles = generate_candles("BTC", "1h", n_bars, end, start_price=100.0, seed=3)
    funding = generate_funding("BTC", n_bars, end, seed=3)
    return candles, funding


def _full_matrix():
    candles, funding = _frames()
    settings = load_settings()
    full = build_features_from_frames(candles, funding=funding, settings=settings, dropna=False)
    return candles, funding, settings, full


def _recompute_diffs(candles, funding, settings, full, check_points):
    """Max abs diff per feature between the full matrix and a prefix recompute,
    evaluated at each check point and aligned by open_time."""
    feat_cols = feature_columns(full)
    full_indexed = full.set_index("open_time")
    diffs = {c: 0.0 for c in feat_cols}
    for t in check_points:
        ot = int(candles["open_time"].iloc[t])
        prefix = build_features_from_frames(
            candles.iloc[: t + 1], funding=funding, settings=settings, dropna=False
        )
        prefix_row = prefix.set_index("open_time").loc[ot]
        full_row = full_indexed.loc[ot]
        for c in feat_cols:
            a, b = full_row[c], prefix_row[c]
            if pd.isna(a) and pd.isna(b):
                continue
            d = abs(float(a) - float(b))
            diffs[c] = max(diffs[c], d if np.isfinite(d) else np.inf)
    return diffs


def test_all_features_are_causal():
    candles, funding, settings, full = _full_matrix()
    n = len(candles)
    # Check across the series, past the warmup region, including near the end.
    check_points = [120, 200, 280, 350, n - 30]
    diffs = _recompute_diffs(candles, funding, settings, full, check_points)
    leaky = {c: d for c, d in diffs.items() if d > TOL}
    assert not leaky, f"Features changed when future data was removed (leakage): {leaky}"


def test_leak_detector_catches_injected_leak():
    """Inject tomorrow's return as a feature and confirm the same comparison
    flags it. This proves the causality check above is not vacuous."""
    candles, funding, settings, full = _full_matrix()

    def with_leak(df):
        m = build_features_from_frames(df, funding=funding, settings=settings, dropna=False)
        # A blatantly forward-looking feature: next bar's close.
        m["LEAKY_next_close"] = df["close"].shift(-1).to_numpy()
        return m

    full_leak = with_leak(candles)
    t = 200
    ot = int(candles["open_time"].iloc[t])
    prefix_leak = with_leak(candles.iloc[: t + 1])
    full_val = full_leak.set_index("open_time").loc[ot, "LEAKY_next_close"]
    prefix_val = prefix_leak.set_index("open_time").loc[ot, "LEAKY_next_close"]
    # On the full series the leaky feature sees tomorrow; on the prefix it cannot
    # (it is the last row, so shift(-1) is NaN). The values must differ.
    assert not (pd.isna(full_val) and pd.isna(prefix_val))
    assert pd.isna(prefix_val) or abs(float(full_val) - float(prefix_val)) > TOL


def test_funding_alignment_is_backward_only():
    """Even with the full (future-bearing) funding series present, a bar's
    funding feature must not change when later candles are removed."""
    candles, funding, settings, full = _full_matrix()
    diffs = _recompute_diffs(candles, funding, settings, full, [150, 300])
    funding_cols = [c for c in feature_columns(full) if c.startswith("funding")]
    assert funding_cols, "expected funding features to be present"
    for c in funding_cols:
        assert diffs[c] <= TOL, f"funding feature {c} leaked future information (diff {diffs[c]})"
