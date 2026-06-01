"""The three trust numbers: the lie-detector for a backtest.

It returns:
  - Deflated Sharpe Ratio (DSR): corrects a Sharpe for selection bias when many
    configurations were tried. A high raw Sharpe means little if hundreds of
    variants were searched; the DSR tells you whether it is likely a fluke.
  - Probability of Backtest Overfitting (PBO) via CSCV: a model-free estimate of
    the probability that the configuration that looked best in-sample will
    underperform out-of-sample.
  - In-sample to out-of-sample degradation: how much performance drops from
    train to test. Large degradation is the fingerprint of overfitting.

The two statistics are implemented here from their published definitions:
  - DSR / Probabilistic Sharpe Ratio: Bailey and Lopez de Prado, "The Deflated
    Sharpe Ratio" (2014), http://ssrn.com/abstract=2460551
  - PBO via Combinatorially Symmetric Cross-Validation (CSCV): Bailey, Borwein,
    Lopez de Prado and Zhu, "The Probability of Backtest Overfitting" (2015),
    http://ssrn.com/abstract=2326253

Only numpy/pandas and the standard library are used; the normal CDF and its
inverse come from statistics.NormalDist, so there is no scipy dependency.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from xortcut.config import Settings, load_settings

_NORMAL = NormalDist()

# Bars per year for annualizing Sharpe, by candle interval. Reporting only;
# DSR and PBO use per-period (factor=1) Sharpe internally.
_BARS_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "2h": 4_380, "4h": 2_190, "8h": 1_095, "12h": 730, "1d": 365,
}


def bars_per_year(interval: str) -> float:
    return float(_BARS_PER_YEAR.get(interval, 8_760))


def sharpe_per_period(returns: Sequence[float]) -> float:
    """Raw per-observation Sharpe (mean / std). This is the frequency DSR and
    PBO expect (factor=1)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))


def annualized_sharpe(returns: Sequence[float], interval: str = "1h") -> float:
    sp = sharpe_per_period(returns)
    return sp * np.sqrt(bars_per_year(interval))


def _even_chunks(n_obs: int, requested: int) -> int:
    """A valid even CSCV chunk count S: even, at least 2, small enough that each
    chunk holds a few observations."""
    s = max(2, requested)
    if s % 2 == 1:
        s -= 1
    while s > 2 and n_obs // s < 2:
        s -= 2
    return s


# ----------------------------------------------------------------- DSR / PSR


def _expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of N independent standard-normal Sharpe estimates
    (Bailey and Lopez de Prado). Used as the deflation benchmark SR_0 / sigma.

        E[max] = (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - e^-1 / N)

    where gamma is the Euler-Mascheroni constant and Z^-1 is the inverse
    standard-normal CDF. Defined for N >= 2.
    """
    n = max(2, int(n_trials))
    g = np.euler_gamma
    return (1.0 - g) * _NORMAL.inv_cdf(1.0 - 1.0 / n) + g * _NORMAL.inv_cdf(1.0 - np.e ** -1 / n)


def _probabilistic_sharpe(sharpe: float, n_obs: int, skew: float, kurtosis: float, target_sharpe: float) -> float:
    """Probabilistic Sharpe Ratio: probability that the true Sharpe exceeds
    target_sharpe, given the observed (per-period) Sharpe and the higher moments
    of the return distribution. kurtosis is the full (non-excess) kurtosis.
    """
    denom = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sharpe - target_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_NORMAL.cdf(z))


def deflated_sharpe(chosen_returns: pd.Series, trials_returns: pd.DataFrame) -> float:
    """DSR of the chosen strategy, deflated by the spread of Sharpes across all
    trials (Bailey and Lopez de Prado).

    The deflation uses the chosen strategy's per-period Sharpe, the standard
    deviation of per-period Sharpes ACROSS the N trials, the trial count N, and
    the chosen strategy's own skew and kurtosis. DSR = PSR(SR_0), where the
    benchmark SR_0 is the Sharpe spread scaled by the expected maximum of N
    independent trials.
    """
    chosen = pd.Series(chosen_returns).dropna()
    trials = trials_returns.dropna(how="any")
    if len(chosen) < 3 or trials.shape[1] < 1:
        return float("nan")
    test_sharpe = sharpe_per_period(chosen)
    trial_sharpes = np.array([sharpe_per_period(trials[c]) for c in trials.columns], dtype="float64")
    trial_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
    sharpe_std = float(np.std(trial_sharpes, ddof=1)) if trial_sharpes.size > 1 else 0.0
    n_trials = int(trials.shape[1])
    t_obs = int(len(chosen))
    skew = float(chosen.skew())
    kurtosis = float(chosen.kurtosis() + 3)  # pandas reports excess kurtosis
    target_sharpe = sharpe_std * _expected_max_sharpe(n_trials)
    return _probabilistic_sharpe(test_sharpe, t_obs, skew, kurtosis, target_sharpe)


# ----------------------------------------------------------------- PBO via CSCV

PBOResult = namedtuple("PBOResult", ["pbo", "prob_oos_loss"])


def _column_sharpe(arr: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column of a T x N return matrix."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)


def probability_of_backtest_overfitting(trials_returns: pd.DataFrame, chunks: int = 16):
    """PBO and probability of OOS loss via Combinatorially Symmetric CV.

    Split the T observations into S contiguous chunks. For every way of choosing
    S/2 chunks as the in-sample (IS) set, the complement is the out-of-sample
    (OOS) set. Pick the configuration with the best IS Sharpe, find its rank
    among all configurations OOS, and form the logit of its relative OOS rank.
    PBO is the fraction of splits where the IS-best config lands in the bottom
    half OOS (logit <= 0). prob_oos_loss is the fraction of splits where that
    config's OOS Sharpe is negative.

    trials_returns is a T x N matrix: one column of per-bar returns per config.
    Returns (PBOResult, S_used).
    """
    clean = trials_returns.dropna(how="any")
    n_obs, n_trials = clean.shape
    if n_trials < 2 or n_obs < 8:
        raise ValueError(f"PBO needs at least 2 trials and 8 observations, got {clean.shape}.")

    s = _even_chunks(n_obs, chunks)
    m = clean.to_numpy()
    # Drop the leading remainder so T divides evenly into S chunks.
    residual = m.shape[0] % s
    if residual:
        m = m[residual:]
    sub_t = m.shape[0] // s
    blocks = [m[i * sub_t:(i + 1) * sub_t, :] for i in range(s)]
    all_blocks = set(range(s))
    n = n_trials

    logits: List[float] = []
    oos_selected: List[float] = []
    for combo in combinations(range(s), s // 2):
        is_blocks = sorted(combo)
        oos_blocks = sorted(all_blocks - set(combo))
        r_is = _column_sharpe(np.concatenate([blocks[i] for i in is_blocks], axis=0))
        r_oos = _column_sharpe(np.concatenate([blocks[i] for i in oos_blocks], axis=0))

        n_star = int(np.argmax(r_is))                  # best config in-sample
        oos_value = r_oos[n_star]
        # Average rank (1-based) of the chosen config's OOS Sharpe among all N.
        less = int(np.sum(r_oos < oos_value))
        equal = int(np.sum(r_oos == oos_value))
        rank = less + (equal + 1) / 2.0
        # Relative rank in (0, 1); N+1 denominator avoids w == 1 (infinite logit).
        w = rank / (n + 1)
        logits.append(float(np.log(w / (1.0 - w))))
        oos_selected.append(float(oos_value))

    pbo_value = float(np.mean(np.asarray(logits) <= 0.0))
    prob_loss = float(np.mean(np.asarray(oos_selected) < 0.0))
    return PBOResult(pbo=pbo_value, prob_oos_loss=prob_loss), s


# ----------------------------------------------------------------- trust numbers


@dataclass
class TrustNumbers:
    """The harness output. All three numbers plus the context to read them."""
    deflated_sharpe: float          # DSR statistic (probability), higher is better
    pbo: float                      # probability of backtest overfitting, lower is better
    prob_oos_loss: float            # probability of an out-of-sample loss
    is_oos_degradation_pct: float   # percent drop in Sharpe from train to test
    oos_sharpe: float               # annualized out-of-sample Sharpe (net of costs)
    is_sharpe: float                # annualized in-sample Sharpe
    n_trials: int                   # number of configurations evaluated
    n_obs: int                      # number of return observations
    cscv_chunks: int                # S used in CSCV

    def as_dict(self) -> dict:
        return asdict(self)


def is_oos_degradation_pct(is_sharpe: float, oos_sharpe: float) -> float:
    """Percent drop in Sharpe from in-sample to out-of-sample.

    100 means OOS collapsed to zero; above 100 means OOS went negative. Guards a
    near-zero in-sample Sharpe to avoid a meaningless ratio.
    """
    if not np.isfinite(is_sharpe) or abs(is_sharpe) < 1e-9:
        return float("nan")
    return float((is_sharpe - oos_sharpe) / abs(is_sharpe) * 100.0)


def compute_trust_numbers(
    trials_returns: pd.DataFrame,
    chosen_column: Optional[str] = None,
    is_returns: Optional[pd.Series] = None,
    oos_returns: Optional[pd.Series] = None,
    interval: str = "1h",
    cscv_chunks: int = 16,
) -> TrustNumbers:
    """Compute all three trust numbers on a matrix of trial returns.

    trials_returns: T x N per-bar returns, one column per configuration.
    chosen_column: which configuration is the candidate (default: best by
        per-period Sharpe). DSR deflates its Sharpe by the spread across trials.
    is_returns / oos_returns: optional explicit in-sample and out-of-sample
        return series for the degradation number (e.g. from walk-forward). If
        omitted, the matrix is split 70/30 in time as a fallback.
    """
    trials = trials_returns.dropna(how="any").reset_index(drop=True)
    n_obs, n_trials = trials.shape

    if chosen_column is None:
        sharpes = {c: sharpe_per_period(trials[c]) for c in trials.columns}
        chosen_column = max(sharpes, key=lambda c: (sharpes[c] if np.isfinite(sharpes[c]) else -np.inf))
    chosen = trials[chosen_column]

    dsr = deflated_sharpe(chosen, trials)

    try:
        pbo_res, s_used = probability_of_backtest_overfitting(trials, cscv_chunks)
        pbo_value = float(pbo_res.pbo)
        prob_loss = float(pbo_res.prob_oos_loss)
    except Exception:
        pbo_value, prob_loss, s_used = float("nan"), float("nan"), _even_chunks(n_obs, cscv_chunks)

    if is_returns is None or oos_returns is None:
        split = int(n_obs * 0.7)
        is_returns, oos_returns = chosen.iloc[:split], chosen.iloc[split:]

    is_sh = annualized_sharpe(is_returns, interval)
    oos_sh = annualized_sharpe(oos_returns, interval)
    degradation = is_oos_degradation_pct(is_sh, oos_sh)

    return TrustNumbers(
        deflated_sharpe=dsr,
        pbo=pbo_value,
        prob_oos_loss=prob_loss,
        is_oos_degradation_pct=degradation,
        oos_sharpe=oos_sh,
        is_sharpe=is_sh,
        n_trials=int(n_trials),
        n_obs=int(n_obs),
        cscv_chunks=int(s_used),
    )


@dataclass
class AcceptanceResult:
    passed: bool
    reasons: List[str]
    trust: TrustNumbers


def check_acceptance(trust: TrustNumbers, settings: Optional[Settings] = None) -> AcceptanceResult:
    """Apply the acceptance gate using config.validation.acceptance.

    A strategy may advance only if all checks pass. Thresholds are read from
    config, never hardcoded.
    """
    settings = settings or load_settings()
    acc = settings.validation.acceptance
    reasons: List[str] = []

    if not (np.isfinite(trust.oos_sharpe) and trust.oos_sharpe >= acc.min_oos_sharpe):
        reasons.append(f"OOS Sharpe {trust.oos_sharpe:.2f} < required {acc.min_oos_sharpe}")
    if not (np.isfinite(trust.pbo) and trust.pbo <= acc.max_pbo):
        reasons.append(f"PBO {trust.pbo:.2f} > allowed {acc.max_pbo}")
    if not (np.isfinite(trust.is_oos_degradation_pct) and trust.is_oos_degradation_pct <= acc.max_is_oos_degradation_pct):
        reasons.append(
            f"IS->OOS degradation {trust.is_oos_degradation_pct:.0f}% > allowed {acc.max_is_oos_degradation_pct}%"
        )

    return AcceptanceResult(passed=len(reasons) == 0, reasons=reasons, trust=trust)
