# Methodology

The three "trust numbers" reported by the validation harness, and the
leakage-control techniques behind them. The two statistics in
`cbt_lab/validation/metrics.py` are implemented from their original definitions
(no third-party statistics library), so this document is also the spec for that
code.

## 1. Triple-barrier labeling

For each bar *t₀* the label asks: over the next horizon, does price first hit an
upper barrier (take-profit, **+1**), a lower barrier (stop-loss, **−1**), or
neither before a time limit (**0**)? The barriers are placed at

```
upper = price[t0] * (1 + up_mult * vol[t0])
lower = price[t0] * (1 - dn_mult * vol[t0])
```

so they scale with recent volatility — a label means the same thing in calm and
wild markets. Each label also records its **exit index** *t₁* (the bar where a
barrier was touched or the time limit hit), which the validation splitters need
to purge overlapping samples. Bars without a full forward window are left
unlabeled to avoid optimistic truncation.

Reference: Marcos Lopez de Prado, *Advances in Financial Machine Learning* (2018).

## 2. Leakage control

Naive cross-validation leaks because training samples adjacent in time to the
test block share information with it. Two defenses, both implemented in
`validation/walk_forward.py`:

- **Purge** — drop any training sample whose label horizon [*t*, *t₁*] overlaps
  the test block. Because a label resolves in the future, a sample near the test
  boundary would otherwise be trained on information overlapping the test period.
- **Embargo** — additionally drop training samples in a short window immediately
  after the test block, to remove residual serial correlation.

Feature construction is held to the same standard: every transform is causal,
all standardization is rolling (never full-sample), and microstructure series
are aligned with a backward as-of join on the bar's close time.
`tests/test_no_leakage.py` enforces this directly.

## 3. Deflated Sharpe Ratio (DSR)

A raw Sharpe ratio is easy to inflate by trying many configurations and keeping
the best. The **Probabilistic Sharpe Ratio (PSR)** gives the probability that
the true Sharpe exceeds a benchmark, accounting for track-record length and the
return distribution's skew and kurtosis:

```
PSR(SR*) = Z( (SR_hat - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )
```

where `Z` is the standard-normal CDF, `SR_hat` is the observed per-period
Sharpe, and `T` the number of observations. The **DSR** sets the benchmark `SR*`
to the Sharpe you would expect *by chance alone* from the best of *N* trials:

```
SR* = sigma_SR * E[max_N],   E[max_N] = (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - e^-1 / N)
```

with `sigma_SR` the spread of Sharpes across the *N* trials and `gamma` the
Euler-Mascheroni constant. A DSR near 1 means the result survives the deflation;
near 0 means it is consistent with luck.

Reference: Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* (2014),
[ssrn.com/abstract=2460551](http://ssrn.com/abstract=2460551).

## 4. Probability of Backtest Overfitting (PBO) via CSCV

**Combinatorially Symmetric Cross-Validation** estimates, model-free, the
probability that the configuration that looked best in-sample underperforms
out-of-sample. Split the *T* observations into *S* contiguous chunks. For every
way of choosing *S*/2 chunks as the in-sample (IS) set, the complement is the
out-of-sample (OOS) set. Select the config with the best IS Sharpe, find its
**relative rank** among all configs OOS, and form the logit of that rank. **PBO**
is the fraction of splits where the IS-best config lands in the bottom half OOS
(logit ≤ 0). A low, positively-centered logit distribution indicates a robust
selection process; a PBO near 0.5 means the selection is information-less.

Reference: Bailey, Borwein, Lopez de Prado & Zhu, *The Probability of Backtest
Overfitting* (2015), [ssrn.com/abstract=2326253](http://ssrn.com/abstract=2326253).

## 5. IS→OOS degradation

The simplest of the three: the percent drop in annualized Sharpe from the
in-sample pool to the out-of-sample pool of the chosen configuration. Large
degradation is the fingerprint of overfitting.
