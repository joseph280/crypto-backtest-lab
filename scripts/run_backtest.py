"""Run a sample strategy through the cost-aware backtest and the validation harness.

This proves the Stage 0 gate: a backtest with realistic fees (0.045 taker),
hourly funding, and slippage, feeding a validation harness that returns the three
trust numbers (deflated Sharpe, PBO, and in-sample to out-of-sample degradation)
on a known input.

IMPORTANT: the "strategies" here are throwaway DEMONSTRATION signals defined
locally in this script, not the real strategy modules. Stage 0 builds the
infrastructure only; no strategy or execution code is written yet. These sample
positions exist solely to exercise the engine and the harness end to end.

    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --coin BTC --interval 1h --synthetic
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from cbt_lab.backtest.engine import run_backtest
from cbt_lab.config import load_settings
from cbt_lab.features.builder import build_features, load_candles
from cbt_lab.labeling.triple_barrier import label_matrix
from cbt_lab.validation.cpcv import cpcv_from_config, n_paths
from cbt_lab.validation.metrics import annualized_sharpe, check_acceptance, compute_trust_numbers
from cbt_lab.validation.walk_forward import walk_forward_from_config


# --------------------------------------------------------------- sample signals
# Each returns a target position series in {-1, 0, +1}, causal (row t uses only
# features known at t). These are demonstrations, not the product.

def _gate(raw: pd.Series, strength: pd.Series, threshold: float) -> pd.Series:
    pos = np.sign(raw)
    pos[strength.abs() < threshold] = 0.0
    return pos.fillna(0.0)


def sample_positions(matrix: pd.DataFrame) -> dict:
    """A small grid of demonstration positions to populate the trials matrix."""
    out = {}
    # Funding-fade: hold the unpopular side when the crowd leans hard (collect
    # funding). Short when funding is richly positive, long when richly negative.
    if "funding_z" in matrix:
        fz = matrix["funding_z"].fillna(0.0)
        for thr in (0.5, 1.0, 1.5):
            out[f"fundfade_{thr}"] = _gate(-np.sign(fz), fz, thr)
    # Momentum: follow the 6-bar move when it is decisive.
    if "mom_6" in matrix:
        m6 = matrix["mom_6"].fillna(0.0)
        for thr in (0.005, 0.01, 0.02):
            out[f"mom6_{thr}"] = _gate(np.sign(m6), m6, thr)
    # Mean reversion: fade Bollinger extremes.
    if "bb_pct_b" in matrix:
        b = (matrix["bb_pct_b"] - 0.5).fillna(0.0)
        for thr in (0.3, 0.45):
            out[f"bbrev_{thr}"] = _gate(-np.sign(b), b, thr)
    return out


# --------------------------------------------------------------- data

def ensure_data(coin: str, interval: str, settings, force_synthetic: bool, min_bars: int) -> None:
    """Make sure a usable lake exists. If missing or too short for walk-forward,
    generate a synthetic lake large enough to demonstrate every splitter."""
    from cbt_lab.data.collectors import candles_path
    from cbt_lab.data.synthetic import generate_lake

    path = candles_path(coin, interval, settings)
    enough = False
    if path.exists() and not force_synthetic:
        enough = len(load_candles(coin, interval, settings)) >= min_bars
    if force_synthetic or not enough:
        # 540 days of hourly data comfortably fits a 180/30 walk-forward.
        days = max(settings.data.history_days, 540)
        logger.warning("Using SYNTHETIC data ({} days). Not market data, no edge.", days)
        generate_lake(settings=settings, days=days)


# --------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a sample strategy and run the validation harness.")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data.")
    args = parser.parse_args()

    settings = load_settings()
    coin, interval = args.coin, args.interval

    # Need enough bars for a 180-day train plus a 30-day test, with headroom.
    from cbt_lab.validation.walk_forward import bars_per_day
    bpd = bars_per_day(interval)
    min_bars = int((settings.validation.walk_forward.train_days + 2 * settings.validation.walk_forward.test_days) * bpd)
    ensure_data(coin, interval, settings, args.synthetic, min_bars)

    logger.info("Building features and labels for {} {}", coin, interval)
    matrix = build_features(coin, interval, settings)
    labels = label_matrix(matrix, settings)
    t1 = labels["exit_index"].fillna(-1).astype(int).to_numpy()
    close = matrix["close"]
    funding = matrix["funding_rate"] if "funding_rate" in matrix else None

    # Run the demonstration grid through the cost-aware engine.
    positions = sample_positions(matrix)
    logger.info("Backtesting {} sample configurations through vectorbt with realistic costs", len(positions))
    net_returns = {}
    for name, pos in positions.items():
        res = run_backtest(close, pos, interval=interval, funding_rate=funding, settings=settings)
        net_returns[name] = res.net_returns.to_numpy()
    trials = pd.DataFrame(net_returns)

    # Choose the best configuration by net per-period Sharpe.
    from cbt_lab.validation.metrics import sharpe_per_period
    sharpes = {c: sharpe_per_period(trials[c]) for c in trials.columns}
    chosen = max(sharpes, key=lambda c: (sharpes[c] if np.isfinite(sharpes[c]) else -np.inf))

    # Walk-forward: pool the chosen config's in-sample and out-of-sample returns.
    chosen_ret = trials[chosen]
    wf = walk_forward_from_config(len(matrix), interval, t1=t1, settings=settings)
    is_pool, oos_pool = [], []
    for tr_idx, te_idx in wf:
        is_pool.append(chosen_ret.iloc[tr_idx])
        oos_pool.append(chosen_ret.iloc[te_idx])
    is_returns = pd.concat(is_pool) if is_pool else None
    oos_returns = pd.concat(oos_pool) if oos_pool else None

    cp = cpcv_from_config(len(matrix), t1=t1, settings=settings)

    trust = compute_trust_numbers(
        trials,
        chosen_column=chosen,
        is_returns=is_returns,
        oos_returns=oos_returns,
        interval=interval,
    )
    acceptance = check_acceptance(trust, settings)

    # ---------------------------------------------------------- report
    line = "=" * 64
    print(f"\n{line}\nBacktest validation harness report\n{line}")
    print(f"data:           {coin} {interval}, {len(matrix)} labeled bars")
    print(f"configs tried:  {trust.n_trials} (sample demonstration signals)")
    print(f"chosen config:  {chosen}")
    print(f"walk-forward:   {len(wf)} purged folds")
    print(f"CPCV:           {len(cp)} splits, {n_paths(settings.validation.cpcv.n_groups, 2)} backtest paths")
    print(line)
    print("THE THREE TRUST NUMBERS (net of fees, funding, slippage)")
    print(f"  1. Deflated Sharpe Ratio (DSR):      {trust.deflated_sharpe:.3f}   (higher is better)")
    print(f"  2. Probability of Backtest Overfit:  {trust.pbo:.3f}   (lower is better)")
    print(f"  3. IS -> OOS Sharpe degradation:     {trust.is_oos_degradation_pct:.1f}%  (lower is better)")
    print(line)
    print(f"  in-sample annualized Sharpe:         {trust.is_sharpe:.2f}")
    print(f"  out-of-sample annualized Sharpe:     {trust.oos_sharpe:.2f}")
    print(f"  probability of OOS loss:             {trust.prob_oos_loss:.3f}")
    print(line)
    verdict = "PASS" if acceptance.passed else "FAIL"
    print(f"acceptance gate (config thresholds): {verdict}")
    for reason in acceptance.reasons:
        print(f"  - {reason}")
    if not acceptance.passed:
        print("  (expected: these are random demonstration signals with no real edge.)")
    print(line + "\n")


if __name__ == "__main__":
    main()
