"""
Exp 1: Core replication - Hawkes endogeneity estimation on synthetic ES data.
Shows rising branching ratio n from ~0.3 in 1998 to ~0.8 in 2010.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize
from src.hawkes.diagnostics import create_rolling_windows

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

# Session parameters
RTH_START = 9 * 3600 + 30 * 60  # 09:30 in seconds since midnight
RTH_END = 16 * 3600 + 15 * 60   # 16:15 in seconds since midnight
RTH_DURATION = RTH_END - RTH_START  # 24300 seconds

def get_regime_params(year: int) -> tuple:
    """Interpolate true Hawkes parameters for a given year (1998-2010)."""
    # Linearly interpolate: n goes from 0.30 (1998) to 0.80 (2010)
    # mu: 0.05 (1998) to 1.0 (2010), beta: 1.5 (1998) to 4.0 (2010)
    t = (year - 1998) / (2010 - 1998)
    n_true = 0.30 + t * 0.50
    mu_true = 0.05 + t * 0.95
    beta_true = 1.5 + t * 2.5
    return mu_true, n_true, beta_true

def generate_trading_day(date: pd.Timestamp, seed: int = 42) -> np.ndarray:
    """Generate synthetic ES mid-price change events for one trading day."""
    year = date.year
    mu, n, beta = get_regime_params(year)
    # Add some day-to-day variation
    rng = np.random.default_rng(seed)
    mu_day = mu * rng.uniform(0.8, 1.2)
    n_day = min(0.95, n * rng.uniform(0.9, 1.1))
    beta_day = beta * rng.uniform(0.9, 1.1)
    times = simulate_hawkes_ogata(mu_day, n_day, beta_day, T=RTH_DURATION, t0=0.0, seed=seed)
    times = round_to_seconds(times)
    return times

def run_single_day_windows(date: pd.Timestamp, times: np.ndarray,
                           window_minutes: int = 10) -> list:
    """Fit Hawkes model in rolling windows for a single trading day."""
    windows = create_rolling_windows(times, 0.0, RTH_DURATION, window_minutes, step_minutes=5)
    results = []
    for w_start, w_end in windows:
        mask = (times >= w_start) & (times <= w_end)
        w_times = times[mask]
        # De-tie randomize (single realization for exp_1)
        rng = np.random.default_rng(int(date.timestamp()) + int(w_start))
        w_times_dt = detie_randomize(w_times, seed=int(rng.integers(0, 2**31)))
        est = estimate_hawkes_window(w_times_dt, w_start, w_end)
        if est is not None:
            est['date'] = date
            est['w_start'] = w_start
            est['w_end'] = w_end
            est['window_min'] = window_minutes
            results.append(est)
    return results

def run_experiment_1(n_days_per_year: int = 20, window_lengths: list = [10, 20, 30],
                     seed_base: int = 42) -> pd.DataFrame:
    """Run core replication experiment across 1998-2010."""
    trading_dates = []
    # Generate ~20 trading days per year for demo (full run would be ~252)
    for year in range(1998, 2011):
        # First Monday of each month for n_days_per_year days
        dates = pd.bdate_range(start=f'{year}-01-01', end=f'{year}-12-31')
        rng = np.random.default_rng(year)
        idx = rng.choice(len(dates), size=min(n_days_per_year, len(dates)), replace=False)
        trading_dates.extend(sorted(dates[idx].tolist()))

    all_results = []
    for i, date in enumerate(trading_dates):
        seed = seed_base + i
        times = generate_trading_day(date, seed=seed)
        for wl in window_lengths:
            day_res = run_single_day_windows(date, times, window_minutes=wl)
            all_results.extend(day_res)
        if i % 20 == 0:
            print(f'  Processed {i}/{len(trading_dates)} days...')

    df = pd.DataFrame(all_results)
    # Two-month centered rolling average of n by window length
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.to_parquet(RESULTS_DIR / 'exp1_window_estimates.parquet', index=False)
    print(f'Saved {len(df)} window estimates to results/')
    return df

def compute_longrun_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 2-month centered rolling average of n by window length."""
    results = []
    for wl in df['window_min'].unique():
        sub = df[df['window_min'] == wl].copy()
        sub = sub.set_index('date').sort_index()
        # Daily mean n
        daily = sub.groupby(sub.index.date)['n_hat'].mean()
        daily.index = pd.to_datetime(daily.index)
        # 60-day rolling centered average
        rolling = daily.rolling(60, center=True, min_periods=5).mean()
        for dt, val in rolling.items():
            if not np.isnan(val):
                results.append({'date': dt, 'window_min': wl, 'n_rolling': val,
                                 'n_raw': daily.get(dt, np.nan)})
    return pd.DataFrame(results)

if __name__ == '__main__':
    print('=== Experiment 1: Core Hawkes Endogeneity Replication ===')
    df = run_experiment_1(n_days_per_year=20)
    lr = compute_longrun_series(df)
    lr.to_csv(RESULTS_DIR / 'exp1_longrun_n.csv', index=False)
    # Summary statistics
    print('\nLong-run n by period (10-min windows):')
    lr10 = lr[lr['window_min'] == 10].copy()
    lr10['year'] = lr10['date'].dt.year
    summary = lr10.groupby('year')['n_rolling'].agg(['mean', 'std', 'min', 'max'])
    print(summary)
    summary.to_csv(RESULTS_DIR / 'exp1_yearly_n_summary.csv')
    print('Exp 1 done.')
