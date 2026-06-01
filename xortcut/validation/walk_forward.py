"""Walk-forward validation with purging and embargo (briefing Section 6).

Walk-forward is the simpler discipline you stand up first: train on a rolling
window, test on the block that immediately follows, advance, repeat. Naive
k-fold leaks because training observations adjacent in time to the test block
share information with it. Two defenses, both applied here:

  - Purge: drop any training sample whose label horizon [t, t1] overlaps the
    test block. The label of a sample reaches into the future (the triple
    barrier resolves at t1), so a sample near the test boundary would otherwise
    be trained on information that overlaps the test period.
  - Embargo: drop training samples in a small window immediately after the test
    block, to kill residual serial correlation.

t1 is the per-sample exit index from the triple-barrier labeler.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from xortcut.config import Settings, load_settings
from xortcut.validation.metrics import bars_per_year

Split = Tuple[np.ndarray, np.ndarray]


def bars_per_day(interval: str) -> float:
    return bars_per_year(interval) / 365.0


def apply_purge_embargo(
    train_idx: np.ndarray, test_idx: np.ndarray, t1: np.ndarray, embargo: int
) -> np.ndarray:
    """Remove training samples that overlap the test block (purge) or fall in
    the embargo window just after it. Vectorized via searchsorted.

    train_idx, test_idx: positional indices. t1: array (length n_samples) of
    label exit positions. embargo: number of bars to embargo after the test set.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        return train_idx
    t1 = np.asarray(t1, dtype=np.int64)
    # A missing/short horizon defaults to the sample itself.
    horizon = np.maximum(t1[train_idx], train_idx)
    test_sorted = np.sort(test_idx)

    # Purge: a test position exists within [i, t1[i]].
    lo = np.searchsorted(test_sorted, train_idx, side="left")
    hi = np.searchsorted(test_sorted, horizon, side="right")
    overlap = hi > lo

    # Embargo: a test position exists in [i - embargo, i - 1] (test just before).
    if embargo > 0:
        e_lo = np.searchsorted(test_sorted, train_idx - embargo, side="left")
        e_hi = np.searchsorted(test_sorted, train_idx - 1, side="right")
        embargoed = e_hi > e_lo
    else:
        embargoed = np.zeros_like(train_idx, dtype=bool)

    return train_idx[~(overlap | embargoed)]


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    t1: Optional[np.ndarray] = None,
    embargo: int = 0,
    step: Optional[int] = None,
    anchored: bool = False,
) -> List[Split]:
    """Generate purged, embargoed walk-forward (train_idx, test_idx) splits.

    anchored=True keeps the train start fixed at 0 (expanding window); otherwise
    the train window rolls. step defaults to test_size (non-overlapping tests).
    """
    if t1 is None:
        t1 = np.arange(n_samples, dtype=np.int64)
    step = step or test_size
    splits: List[Split] = []
    train_start = 0
    test_start = train_size
    while test_start + test_size <= n_samples:
        tr_start = 0 if anchored else train_start
        train_idx = np.arange(tr_start, test_start, dtype=np.int64)
        test_idx = np.arange(test_start, test_start + test_size, dtype=np.int64)
        train_idx = apply_purge_embargo(train_idx, test_idx, t1, embargo)
        if train_idx.size > 0:
            splits.append((train_idx, test_idx))
        train_start += step
        test_start += step
    return splits


def walk_forward_from_config(
    n_samples: int,
    interval: str,
    t1: Optional[np.ndarray] = None,
    settings: Optional[Settings] = None,
    anchored: bool = False,
) -> List[Split]:
    """Build walk-forward splits from config.validation.walk_forward and the
    candle interval (train_days/test_days converted to bars; embargo in bars)."""
    settings = settings or load_settings()
    wf = settings.validation.walk_forward
    bpd = bars_per_day(interval)
    train_size = max(2, int(round(wf.train_days * bpd)))
    test_size = max(1, int(round(wf.test_days * bpd)))
    embargo = settings.validation.cpcv.embargo_bars if settings.validation.purge else 0
    return walk_forward_splits(
        n_samples, train_size, test_size, t1=t1, embargo=embargo, anchored=anchored
    )
