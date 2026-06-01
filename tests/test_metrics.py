"""Tests for the validation harness: trust numbers and the purged splitters."""

import numpy as np
import pandas as pd

from xortcut.validation.cpcv import cpcv_splits, make_groups, n_paths
from xortcut.validation.metrics import (
    annualized_sharpe,
    compute_trust_numbers,
    is_oos_degradation_pct,
    sharpe_per_period,
)
from xortcut.validation.walk_forward import apply_purge_embargo, walk_forward_splits


def _trial_matrix(T=400, n_noise=8, edge_mean=0.0015, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"noise_{i}": rng.standard_normal(T) * 0.01 for i in range(n_noise)}
    data["edge"] = rng.standard_normal(T) * 0.01 + edge_mean
    return pd.DataFrame(data)


# ----------------------------------------------------------------- sharpe

def test_sharpe_scales_with_frequency():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(1000) * 0.01 + 0.001)
    sp = sharpe_per_period(r)
    ann = annualized_sharpe(r, "1h")
    assert np.isfinite(sp) and np.isfinite(ann)
    assert ann > sp  # annualization inflates the per-period number


def test_degradation_sign():
    # OOS worse than IS -> positive degradation; OOS better -> negative.
    assert is_oos_degradation_pct(2.0, 1.0) == 50.0
    assert is_oos_degradation_pct(2.0, 3.0) == -50.0
    assert np.isnan(is_oos_degradation_pct(0.0, 1.0))


# ----------------------------------------------------------------- trust numbers

def test_trust_numbers_are_scalars_and_finite():
    trials = _trial_matrix()
    tn = compute_trust_numbers(trials, interval="1h")
    for key in ("deflated_sharpe", "pbo", "prob_oos_loss", "oos_sharpe", "is_sharpe"):
        v = getattr(tn, key)
        assert np.isscalar(v) and np.isfinite(v), f"{key} not a finite scalar: {v}"
    assert 0.0 <= tn.pbo <= 1.0
    assert tn.n_trials == trials.shape[1]


def test_pbo_higher_for_pure_noise_than_for_clear_edge():
    # A matrix of pure noise should overfit more readily than one with a real,
    # persistent edge column.
    rng = np.random.default_rng(2)
    noise = pd.DataFrame({f"s_{i}": rng.standard_normal(500) * 0.01 for i in range(10)})
    edge = noise.copy()
    edge["real"] = rng.standard_normal(500) * 0.005 + 0.002
    pbo_noise = compute_trust_numbers(noise, interval="1h").pbo
    pbo_edge = compute_trust_numbers(edge, interval="1h").pbo
    assert pbo_noise >= pbo_edge


# ----------------------------------------------------------------- splitters

def test_walk_forward_train_test_disjoint_and_ordered():
    splits = walk_forward_splits(1000, train_size=400, test_size=100, embargo=10)
    assert len(splits) > 0
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))
        assert tr.max() < te.min()  # train strictly precedes test


def test_purge_removes_overlapping_labels():
    n = 100
    t1 = np.arange(n) + 5  # each label horizon reaches 5 bars forward
    train = np.arange(0, 50)
    test = np.arange(50, 70)
    kept = apply_purge_embargo(train, test, t1, embargo=0)
    # Samples 45..49 have horizons reaching into the test block and must be purged.
    assert max(kept) < 45


def test_embargo_removes_samples_after_test():
    n = 100
    t1 = np.arange(n)  # zero-length horizons, so only embargo acts
    train = np.arange(0, 100)
    test = np.arange(40, 60)
    kept = apply_purge_embargo(train, test, t1, embargo=5)
    # Positions 60..64 are within the embargo window after the test block.
    assert not any(60 <= i <= 64 for i in kept)


def test_cpcv_paths_and_coverage():
    n = 600
    splits = cpcv_splits(n, n_groups=6, n_test_groups=2, embargo=5)
    from math import comb
    assert len(splits) == comb(6, 2)
    assert n_paths(6, 2) == 5
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))


def test_groups_partition_all_samples():
    groups = make_groups(100, 6)
    covered = np.concatenate(groups)
    assert len(covered) == 100 and len(np.unique(covered)) == 100
