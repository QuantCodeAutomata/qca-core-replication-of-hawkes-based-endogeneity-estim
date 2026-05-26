"""
Exp 2: Bootstrap de-tie procedure and goodness-of-fit diagnostics.
Targets: rejection rate ~0.12% for 10-min, ~1.64% for 20-min, ~12.64% for 30-min windows.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.diagnostics import bootstrap_window, create_rolling_windows
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds

RESULTS_DIR = Path('/workspace/project/results')
RTH_DURATION = 24300  # seconds

def generate_day_events(mu: float, n: float, beta: float, seed: int = 42) -> np.ndarray:
    """Generate one day of synthetic Hawkes events, rounded to seconds."""
    times = simulate_hawkes_ogata(mu, n, beta, T=RTH_DURATION, t0=0.0, seed=seed)
    return round_to_seconds(times)

def run_experiment_2(n_days: int = 30, n_bootstrap: int = 50,
                     window_lengths: list = [10, 20, 30]) -> pd.DataFrame:
    """Run bootstrap diagnostics experiment."""
    # Use mid-sample parameters (year ~2005)
    mu, n_true, beta = 0.5, 0.70, 3.0
    all_results = []
    for day_idx in range(n_days):
        seed = 1000 + day_idx
        times = generate_day_events(mu, n_true, beta, seed=seed)
        for wl in window_lengths:
            windows = create_rolling_windows(times, 0.0, RTH_DURATION, wl, step_minutes=5)
            for w_start, w_end in windows[:5]:  # limit for speed
                mask = (times >= w_start) & (times <= w_end)
                w_times = times[mask]
                boot = bootstrap_window(w_times, w_start, w_end,
                                        n_bootstrap=n_bootstrap,
                                        seed_base=seed * 1000 + int(w_start))
                row = {
                    'day': day_idx,
                    'window_min': wl,
                    'w_start': w_start,
                    'w_end': w_end,
                    'n_events': boot['n_events'],
                    'accepted': boot['accepted'],
                    'max_pvalue': max(boot['ks_pvalues']) if boot['ks_pvalues'] else np.nan,
                    'n_mean': np.mean(boot['ns']) if boot['ns'] else np.nan,
                    'n_std': np.std(boot['ns']) if boot['ns'] else np.nan,
                    'n_true': n_true,
                }
                all_results.append(row)
        if day_idx % 5 == 0:
            print(f'  Bootstrap day {day_idx}/{n_days}...')

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'exp2_bootstrap_results.csv', index=False)
    # Rejection rates
    for wl in window_lengths:
        sub = df[df['window_min'] == wl]
        reject_rate = (~sub['accepted']).mean() * 100
        print(f'  Window {wl}-min: rejection rate = {reject_rate:.2f}%')
    return df

if __name__ == '__main__':
    print('=== Experiment 2: Bootstrap De-tie and KS Diagnostics ===')
    df = run_experiment_2(n_days=10, n_bootstrap=50)
    # Summary
    summary = df.groupby('window_min').agg(
        rejection_rate=('accepted', lambda x: (~x).mean()),
        n_mean=('n_mean', 'mean'),
        n_std_mean=('n_std', 'mean'),
    ).reset_index()
    summary['rejection_pct'] = summary['rejection_rate'] * 100
    print(summary)
    summary.to_csv(RESULTS_DIR / 'exp2_rejection_summary.csv', index=False)
    print('Exp 2 done.')
