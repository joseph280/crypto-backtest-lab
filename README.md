# xortcut-backtest-lab

A small, self-contained **backtest and overfitting-validation lab** for crypto
perpetual-futures strategies. It runs entirely offline on synthetic data: no API
keys, no venue connection, no network. One command builds features, labels them,
runs a cost-aware backtest, and reports the statistics that tell you whether a
backtest is trustworthy or just lucky.

This is an extracted, self-contained slice of a larger personal research
project, published as a portfolio piece. It contains the **research
infrastructure only** — the strategies it runs are deliberately throwaway demo
signals, not a real edge (see [No edge here](#no-edge-here)).

## Why it exists

Most retail backtests lie. They report a beautiful Sharpe ratio that evaporates
the moment real fees, funding, and slippage are applied, or that was cherry-picked
from hundreds of silent variations. This project is built around the opposite
discipline: a backtest you can trust, measured by three numbers that are hard to
fool.

```
================================================================
Xortcut Stage 0 validation harness report
================================================================
data:           BTC 1h, 12889 labeled bars
configs tried:  8 (sample demonstration signals)
chosen config:  fundfade_0.5
walk-forward:   11 purged folds
CPCV:           15 splits, 5 backtest paths
================================================================
THE THREE TRUST NUMBERS (net of fees, funding, slippage)
  1. Deflated Sharpe Ratio (DSR):      0.001   (higher is better)
  2. Probability of Backtest Overfit:  0.155   (lower is better)
  3. IS -> OOS Sharpe degradation:     -132.6%  (lower is better)
================================================================
  in-sample annualized Sharpe:         -0.84
  out-of-sample annualized Sharpe:     0.27
  probability of OOS loss:             0.989
================================================================
acceptance gate (config thresholds): FAIL
  - OOS Sharpe 0.27 < required 0.8
  (expected: these are random demonstration signals with no real edge.)
```

The harness *correctly rejects* the demo signals at the acceptance gate — here a
near-zero Deflated Sharpe and an OOS Sharpe far below the required threshold.
That is the point: the infrastructure is honest enough to tell you when there is
nothing there. (Exact numbers vary with the synthetic seed and config.)

## What it demonstrates

- **Leakage-safe feature engineering.** Every transform is causal (rolling /
  trailing, never full-sample). Microstructure series (funding, open interest,
  basis) are aligned to each bar with a backward as-of join on the bar's *close*
  time, so a bar never sees information from after it closed. Enforced by
  `tests/test_no_leakage.py`, which asserts the feature row at time *t* is
  identical whether computed on data up to *t* or on the full series.
- **Triple-barrier labeling** (Lopez de Prado): volatility-scaled take-profit /
  stop-loss / time barriers, vectorized, returning each label's exit index for
  purging.
- **A realistic cost model.** Maker/taker fees, slippage, and — the part most
  backtesters skip — **funding** layered onto every bar via the vectorbt engine.
- **Purged, embargoed validation.** Walk-forward and **Combinatorial Purged
  Cross-Validation (CPCV)** with purge + embargo to kill label-overlap leakage
  across the train/test boundary.
- **The three trust numbers**, implemented from their original papers (see
  [METHODOLOGY.md](docs/METHODOLOGY.md)): the **Deflated Sharpe Ratio**, the
  **Probability of Backtest Overfitting** via CSCV, and **IS→OOS degradation**.
- **Config-driven, no magic numbers.** Every fee, threshold, and window is read
  from `config/settings.yaml` through a typed pydantic loader.

## Quick start

Requires [uv](https://docs.astral.sh/uv) and Python 3.12.

```bash
uv sync
uv run python scripts/run_backtest.py --synthetic
uv run pytest -q
```

`run_backtest.py` generates a synthetic data lake if none exists (a seeded
geometric random walk with volatility clustering and mean-reverting funding —
clearly labelled, not market data), then runs the full pipeline and prints the
report above.

## Layout

```
xortcut/
  config.py              # typed settings loader (pydantic)
  data/
    synthetic.py         # seeded synthetic candles + funding (offline data source)
    collectors.py        # parquet lake paths + interval helpers
    microstructure.py    # funding / open-interest / basis features (causal)
  features/
    indicators.py        # technical indicators (causal)
    builder.py           # assembles the leakage-safe feature matrix
  labeling/
    triple_barrier.py    # volatility-scaled triple-barrier labels
  backtest/
    costs.py             # fees, slippage, funding cost model
    engine.py            # vectorbt engine, target-position -> net returns
  validation/
    walk_forward.py      # purged + embargoed walk-forward splits
    cpcv.py              # combinatorial purged cross-validation
    metrics.py           # DSR, PBO (CSCV), IS->OOS degradation
config/settings.yaml     # every number the pipeline uses
scripts/run_backtest.py  # end-to-end entry point
tests/                   # incl. test_no_leakage.py
```

## No edge here

The "strategies" run by `scripts/run_backtest.py` (a funding fade, a momentum
follow, a Bollinger mean-reversion) are **demonstration signals defined in the
script itself**, included only to exercise the engine and the harness end to
end. They carry no real trading edge, and the harness is expected to reject
them. This repository is about the *plumbing and the validation discipline*, not
a profitable system. Nothing here is financial advice.

## License

MIT — see [LICENSE](LICENSE).
