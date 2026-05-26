# Hawkes Process Endogeneity — ES Futures 1998–2010

A replication and extension of **Filimonov & Sornette (2012)** *"Quantifying reflexivity in financial markets: Toward a prediction of flash crashes"*. This project estimates the Hawkes branching ratio **n** on E-mini S&P 500 futures tick data to measure the degree of market endogeneity, and validates the methodology with synthetic experiments.

---

## Background

A univariate Hawkes process models self-exciting event sequences:

```
λ(t) = μ + n·β · Σ_{tᵢ < t} exp(−β·(t − tᵢ))
```

| Parameter | Meaning |
|-----------|---------|
| `μ` | Background (exogenous) intensity — driven by news, macro events |
| `n` | Branching ratio ∈ (0, 1) — fraction of events that are endogenous (self-caused) |
| `β` | Decay rate of the self-excitation kernel |

**n → 1** signals a market on the edge of instability; **n → 0** means all activity is exogenously driven. The paper shows n rose from ~0.3 in 1998 to ~0.7 by 2007–2010, with a spike toward 0.9–0.95 during the May 6, 2010 Flash Crash.

---

## Project Structure

```
project/
├── src/
│   ├── hawkes/
│   │   ├── simulator.py      # Ogata thinning simulation
│   │   ├── estimator.py      # Numba-accelerated MLE (L-BFGS-B)
│   │   └── diagnostics.py    # Residuals, KS test, bootstrap, rolling windows
│   ├── data_processing/      # Tick data loading and preprocessing
│   └── utils/
│       └── plotting.py       # Matplotlib/seaborn figure generation
├── exp/
│   ├── exp_1_longrun_n.py            # Long-run trend 1998–2010
│   ├── exp_2_bootstrap_rejection.py  # KS rejection rates by window length
│   ├── exp_3_synthetic_validation.py # Rounding + de-tie bias study
│   ├── exp_4_reshuffling.py          # Poisson reshuffling control
│   ├── exp_5_case_study.py           # Apr 27 vs May 6, 2010
│   └── exp_6_intraday_seasonality.py # Intraday mu U-shape vs flat n
├── tests/
│   ├── test_hawkes_estimator.py  # Unit tests: MLE, compensator
│   ├── test_diagnostics.py       # Unit tests: residuals, KS, bootstrap
│   └── test_experiments.py       # Integration tests: all 6 experiments
├── data/                         # Tick data (not version-controlled if large)
├── results/                      # Generated figures and parquet outputs
├── requirements.txt
└── README.md
```

---

## Experiments

| # | Name | Key Finding |
|---|------|-------------|
| 1 | Long-run n | n rises from ~0.3 (1998) to ~0.7 (2007–2010) |
| 2 | Bootstrap rejection | 10-min windows: ~0.1% KS rejection; 30-min: ~13% |
| 3 | Synthetic validation | Rounding + de-tie introduces <20% relative bias in n |
| 4 | Poisson reshuffling | Reshuffled series gives n ≈ 0 — high n is not a volume proxy |
| 5 | Case study (Flash Crash) | May 6, 2010: n peaks at ~0.95 vs ~0.80 on Apr 27 |
| 6 | Intraday seasonality | mu has U-shape; n is relatively flat across the session |

---

## Methodology

### Estimation

Parameters are estimated by maximum likelihood using `scipy.optimize.minimize` with `L-BFGS-B`. The objective function uses the recursive identity:

```
R_i = exp(−β·Δt_i) · (R_{i−1} + 1),   R_0 = 0
λ(t_i) = μ + n·β·R_i
Λ(T) = μ·T + n · Σᵢ (1 − exp(−β·(T − tᵢ)))
```

The inner loop is JIT-compiled with **Numba** for speed. Parameters are optimised in unconstrained space: `log_mu`, `logit_n`, `log_beta`.

### Goodness-of-Fit: Bootstrap De-tie Test

Tick data is reported at 1-second resolution, creating many duplicate timestamps. The validation procedure:

1. **Round** continuous times to integer seconds.
2. **De-tie**: add Uniform(0, 0.01 s) jitter to break ties (50 bootstrap realizations).
3. **Re-estimate** Hawkes parameters for each realization.
4. **KS test**: residual inter-arrivals `Δᵢ = Λ(tᵢ) − Λ(tᵢ₋₁)` should be Exp(1); their transforms `Uᵢ = 1 − exp(−Δᵢ)` should be Uniform(0, 1).
5. **Accept** window if at least one of 50 realizations passes at α = 5%.

### Compensator Formula

```
Λ(tᵢ) = μ·(tᵢ − t₀) + n·(i − Rᵢ)
```

where `i` counts previous events and `Rᵢ` is the decaying sum computed recursively.

---

## Installation

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

> **Note**: `tick` requires a C++ compiler. On Linux: `sudo apt-get install g++`.  
> **Numba** compiles on first run — expect a short warm-up delay.

---

## Running Tests

```bash
# From project root
pytest tests/ -v

# Run a single module
pytest tests/test_hawkes_estimator.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

All tests are deterministic (fixed seeds) and do not require tick data.

---

## Running Experiments

```python
# Example: experiment 6 — intraday seasonality
from exp.exp_6_intraday_seasonality import run_exp6
from src.utils.plotting import plot_intraday_seasonality

summary = run_exp6(n_days=21, n_true=0.65, beta=3.0)
plot_intraday_seasonality(summary)
# → results/fig_exp6_intraday.png
```

---

## Key Results

- **Long-run endogeneity**: The branching ratio n approximately tripled from 1998 to 2010, consistent with increased algorithmic and high-frequency trading participation.
- **Flash Crash signal**: On May 6, 2010, n reached ~0.95 in the minutes before the crash — the market was 95% self-referential — versus ~0.80 on a typical 2010 day.
- **Reshuffling control**: Poisson-reshuffled versions of the same data yield n ≈ 0, confirming that high n is not merely a proxy for trading volume.
- **Window robustness**: 10-minute windows (1 200 events on average) have a <0.5% KS rejection rate; 30-minute windows reject ~13%, consistent with the paper's findings.

---

## References

- Filimonov, V., & Sornette, D. (2012). *Quantifying reflexivity in financial markets: Toward a prediction of flash crashes*. Physical Review E, 85(5), 056108.
- Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika, 58(1), 83–90.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Transactions on Information Theory, 27(1), 23–31.
