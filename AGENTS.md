# Repository Knowledge — Hawkes Endogeneity Project

## Project Purpose
Replication of Filimonov & Sornette (2012): measures market endogeneity via
Hawkes branching ratio **n** on ES futures tick data 1998–2010.

## Architecture
- `src/hawkes/simulator.py`   — Ogata thinning: `simulate_hawkes_ogata`, `round_to_seconds`, `detie_randomize`
- `src/hawkes/estimator.py`   — Numba-JIT MLE: `hawkes_log_likelihood`, `estimate_hawkes_window`, `compute_compensator`
- `src/hawkes/diagnostics.py` — `compute_residuals`, `residuals_to_uniforms`, `ks_test_uniformity`, `bootstrap_window`, `create_rolling_windows`
- `src/utils/plotting.py`     — Six `plot_*` functions (one per experiment), non-interactive Agg backend
- `exp/exp_1_core_replication.py`     — `get_regime_params`, `generate_trading_day`, `run_single_day_windows`, `run_experiment_1`, `compute_longrun_series`; linearly interpolates (mu, n, beta) 1998→2010 (n: 0.30→0.80), fits Hawkes in rolling windows (10/20/30 min), saves `exp1_window_estimates.parquet` + `exp1_longrun_n.csv` + `exp1_yearly_n_summary.csv`
- `exp/exp_2_bootstrap_diagnostics.py` — `generate_day_events`, `run_experiment_2`; mid-sample params (mu=0.5, n=0.70, beta=3.0), runs `bootstrap_window` (50 de-tie reps) on first 5 windows per length per day, saves `exp2_bootstrap_results.csv` + `exp2_rejection_summary.csv`
- `exp/exp_3_synthetic_validation.py`  — `run_mc_validation`, `run_experiment_3`; 4-regime PARAM_GRID (1998/2002/2007/2010), 100 MC × 20 bootstrap reps on T=600 s windows, records bias+std for (mu, n, beta); saves `exp3_synthetic_validation.csv` + `exp3_validation_summary.csv`
- `exp/exp_4_poisson_reshuffling.py` — `get_regime_params`, `poisson_reshuffle`, `estimate_windows_for_times`, `run_experiment_4`; compares empirical n̂ vs n̂ from Poisson-reshuffled events (same count, no clustering) across 1998–2010
- `exp/exp_5_case_study.py`         — `simulate_exogenous_day` (Apr 27, n≈0.78, exogenous shock at 10:00), `simulate_endogenous_day` (May 6 flash crash, n≈0.93 during 14:30–15:00), `compute_intraday_hawkes` (rolling bootstrap via `bootstrap_window`), `run_experiment_5`; outputs `exp5_case_study.csv`
- `exp/exp_6_intraday_seasonality.py` — `intraday_mu_profile` (cosine: `1.5 - cos(2π·norm)`, LOW at open/close, HIGH at midday), `simulate_intraday_hawkes_day` (25 equal bins with piecewise-constant mu), `run_experiment_6`, `summarize_intraday`; outputs `exp6_intraday.parquet` and `exp6_intraday_summary.csv`

## Parameter Space
| Python name | Meaning | Constraint |
|-------------|---------|------------|
| `mu`   | Background intensity (events/s) | > 0 |
| `n`    | Branching ratio                  | ∈ (0, 0.999) |
| `beta` | Exponential decay rate           | > 0 |

Optimisation transforms: `log_mu`, `logit_n = log(n / (0.999-n))`, `log_beta`.

## Key Conventions
- All times are in **seconds** from an arbitrary epoch (or from session open).
- RTH session: 09:30–16:15 EST = **24 300 s**.
- Rolling windows use integer-minute lengths; `create_rolling_windows` returns
  `(24300 - window_sec) / step_sec + 1` windows for a full session.
- `bootstrap_window` detie's the *original* rounded times each replication;
  it does NOT re-simulate — it re-jitters.
- `accepted = max(ks_pvalues) >= 0.05` (reject only if ALL 50 realizations fail).

## Testing
- `pytest tests/ -v` runs all 3 modules; all seeds are fixed → deterministic.
- No external data needed; all tests use `simulate_hawkes_ogata`.
- Numba JIT compiles on first import (warm-up ~5 s).

## Numba Notes
- `_loglik_inner` and `_compensator_inner` use `@jit(nopython=True, cache=True)`.
- Cache is stored in `src/hawkes/__pycache__/`. Invalidated by function body changes.
- Do not use Python objects (lists, dicts) inside these Numba functions.
- **Zero-dt guard**: both numba loops branch on `dt > 0`; when `dt == 0` (tied
  integer-second timestamps) `R += 1` instead of `exp(-beta*0)*(R+1)` to avoid
  `exp(-inf * 0) = nan` when beta hits its upper bound during optimisation.

## Optimisation Details
- L-BFGS-B bounds: `log_mu ∈ (-15, 15)`, `logit_n ∈ (-15, 15)`, `log_beta ∈ (-5, 10)`.
  These prevent float overflow (exp(>709) → inf → nan in Numba) while covering
  all physically meaningful regimes.
- `hit_bound` flag: True when `n_hat > 0.99` or `n_hat < 0.001`.

## Data Pipeline
- `data/download_data.py` — two entry points:
  - `download_spy_minute(start, end, path)`: needs `MASSIVE_API_KEY` env var;
    uses `massive.RESTClient.list_aggs()` with `timespan="minute"`.
  - `generate_synthetic_es_events(n_days, year_regime, ...)`: simulates RTH
    sessions using `simulate_hawkes_ogata`; returns DataFrame with
    `date, timestamp_sec, regime`.
  - `generate_full_sample_synthetic(1998, 2010)`: interpolates (mu, n, beta)
    linearly over the panel; saves `data/synthetic_es_events.parquet`;
    returns per-day summary DataFrame.
- Note: full 1998-2010 generation can produce ~100M events (2010 params have
  stationary rate ≈ 5 events/sec over 24300 s per day).

## Dependencies
numpy, pandas, scipy, matplotlib, seaborn, numba, tick, pytest, pyarrow, massive
