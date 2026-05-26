"""
Exp 4: Poisson reshuffling control - preserves daily event counts but destroys clustering.
Expected: reshuffled n << empirical n across all years.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize
from src.hawkes.diagnostics import create_rolling_windows

RESULTS_DIR = Path('/workspace/project/results')
RTH_DURATION = 24300  # 09:30 to 16:15 in seconds

def get_regime_params(year: int) -> tuple:
    t = (year - 1998) / (2010 - 1998)
    n_true = 0.30 + t * 0.50
    mu_true = 0.05 + t * 0.95
    beta_true = 1.5 + t * 2.5
    return mu_true, n_true, beta_true

def poisson_reshuffle(n_events: int, T: float, seed: int = 42) -> np.ndarray:
    """Draw n_events uniform times in [0, T], sorted."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, T, size=n_events))

def estimate_windows_for_times(times: np.ndarray, window_minutes: int = 10) -> list:
    """Estimate Hawkes n in rolling windows, return list of n_hat values."""
    windows = create_rolling_windows(times, 0.0, RTH_DURATION, window_minutes, step_minutes=5)
    n_vals = []
    for w_start, w_end in windows:
        mask = (times >= w_start) & (times <= w_end)
        w_times = times[mask]
        if len(w_times) < 5:
            continue
        w_dt = detie_randomize(w_times, seed=int(w_start * 100))
        est = estimate_hawkes_window(w_dt, w_start, w_end)
        if est is not None and est['converged']:
            n_vals.append(est['n_hat'])
    return n_vals

def run_experiment_4(n_days_per_year: int = 15, window_minutes: int = 10,
                     seed_base: int = 42) -> pd.DataFrame:
    """Compare empirical vs Poisson-reshuffled n across 1998-2010."""
    rows = []
    for year in range(1998, 2011):
        mu, n_true, beta = get_regime_params(year)
        for day_idx in range(n_days_per_year):
            seed = seed_base + year * 1000 + day_idx
            # 1. Generate real Hawkes events
            emp_times = simulate_hawkes_ogata(mu, n_true, beta, T=RTH_DURATION, t0=0.0, seed=seed)
            emp_times = round_to_seconds(emp_times)
            n_emp = len(emp_times)
            # 2. Poisson reshuffle (same count)
            pois_times = poisson_reshuffle(n_emp, RTH_DURATION, seed=seed + 999999)
            # 3. Estimate n in rolling windows
            emp_ns = estimate_windows_for_times(emp_times, window_minutes)
            pois_ns = estimate_windows_for_times(pois_times, window_minutes)
            rows.append({
                'year': year,
                'day': day_idx,
                'n_events': n_emp,
                'n_true': n_true,
                'n_empirical_mean': np.mean(emp_ns) if emp_ns else np.nan,
                'n_poisson_mean': np.mean(pois_ns) if pois_ns else np.nan,
                'n_empirical_std': np.std(emp_ns) if emp_ns else np.nan,
                'n_poisson_std': np.std(pois_ns) if pois_ns else np.nan,
            })
        print(f'  Year {year}: n_true={n_true:.2f}')

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'exp4_reshuffling.csv', index=False)
    return df

if __name__ == '__main__':
    print('=== Experiment 4: Poisson Reshuffling Control ===')
    df = run_experiment_4(n_days_per_year=10)
    summary = df.groupby('year').agg(
        n_true=('n_true', 'first'),
        n_emp=('n_empirical_mean', 'mean'),
        n_pois=('n_poisson_mean', 'mean'),
        gap=('n_empirical_mean', lambda x: x.mean() - df.loc[x.index, 'n_poisson_mean'].mean()),
    ).reset_index()
    print(summary)
    summary.to_csv(RESULTS_DIR / 'exp4_comparison_summary.csv', index=False)
    print('Exp 4 done.')
