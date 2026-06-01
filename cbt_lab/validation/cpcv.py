"""Combinatorial Purged Cross-Validation (briefing Section 6).

Walk-forward gives one path through history and one Sharpe. CPCV is the rigorous
follow-up: partition the samples into N contiguous groups, hold out every
combination of k groups as the test set, and train on the rest, applying the
same purge and embargo as walk-forward. This produces a DISTRIBUTION of
out-of-sample results across many train/test paths rather than one lucky number,
which is what the PBO and deflated-Sharpe statistics need.

Number of distinct backtest paths reconstructable from the splits is
C(N, k) * k / N (Lopez de Prado).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import List, NamedTuple, Optional

import numpy as np

from cbt_lab.config import Settings, load_settings
from cbt_lab.validation.walk_forward import apply_purge_embargo


class CpcvSplit(NamedTuple):
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple  # which group ids form the test set


def make_groups(n_samples: int, n_groups: int) -> List[np.ndarray]:
    """Partition positions [0, n_samples) into n_groups contiguous groups."""
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    return [g for g in np.array_split(np.arange(n_samples, dtype=np.int64), n_groups) if g.size > 0]


def n_paths(n_groups: int, n_test_groups: int = 2) -> int:
    """Number of backtest paths: C(N, k) * k / N."""
    return comb(n_groups, n_test_groups) * n_test_groups // n_groups


def cpcv_splits(
    n_samples: int,
    n_groups: int,
    n_test_groups: int = 2,
    t1: Optional[np.ndarray] = None,
    embargo: int = 0,
) -> List[CpcvSplit]:
    """Generate all combinatorial purged train/test splits.

    For each choice of n_test_groups groups out of n_groups, the test set is the
    union of those groups and the train set is everything else, purged of
    label-overlap and embargoed.
    """
    if t1 is None:
        t1 = np.arange(n_samples, dtype=np.int64)
    groups = make_groups(n_samples, n_groups)
    actual_n = len(groups)
    if n_test_groups >= actual_n:
        raise ValueError(f"n_test_groups ({n_test_groups}) must be < number of groups ({actual_n}).")

    splits: List[CpcvSplit] = []
    for combo in combinations(range(actual_n), n_test_groups):
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        train_groups = [g for g in range(actual_n) if g not in combo]
        train_idx = np.sort(np.concatenate([groups[g] for g in train_groups]))
        train_idx = apply_purge_embargo(train_idx, test_idx, t1, embargo)
        if train_idx.size > 0:
            splits.append(CpcvSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo))
    return splits


def cpcv_from_config(
    n_samples: int,
    t1: Optional[np.ndarray] = None,
    n_test_groups: int = 2,
    settings: Optional[Settings] = None,
) -> List[CpcvSplit]:
    """Build CPCV splits from config.validation.cpcv (n_groups, embargo_bars)."""
    settings = settings or load_settings()
    cp = settings.validation.cpcv
    embargo = cp.embargo_bars if settings.validation.purge else 0
    return cpcv_splits(
        n_samples, n_groups=cp.n_groups, n_test_groups=n_test_groups, t1=t1, embargo=embargo
    )
