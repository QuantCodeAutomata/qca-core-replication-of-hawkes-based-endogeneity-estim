"""
Exp 5: Case study - April 27 vs May 6, 2010 (Flash Crash).
May 6 should show elevated n peaking near 0.95; April 27 should not.
Using synthetic simulation calibrated to 2010-regime with
an exogenous shock on Apr 27 and an endogenous crisis on May 6.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize
from src.hawkes.diagnostics import create_rolling_windows, bootstrap_window

RESULTS_DIR = Path('/workspace/project/results')
RTH_DURATION = 24300  # seconds
RTH_START = 9 * 3600 + 30 * 60  # seconds since midnight

def time_to_sec(h: int, m: int, s: int = 0) -> float:
    """Convert clock time to seconds since session start (09:30)."""
    return (h * 3600 + m * 60 + s) - RTH_START

def simulate_exogenous_day(seed: int = 100) -> tuple:
    """
    April 27 - exogenous shock scenario.
    Large price move but normal n (~0.8 baseline, no n spike).
    Model as a day with elevated mu during shock, but normal branching ratio.
    Returns: (event_times, minute_prices, minute_volumes)
    """
    mu_base, n_base, beta = 1.0, 0.78, 4.0
    rng = np.random.default_rng(seed)
    # Generate full day events
    times = simulate_hawkes_ogata(mu_base, n_base, beta, T=RTH_DURATION, t0=0.0, seed=seed)
    times = round_to_seconds(times)
    # Simulate minute-by-minute price (random walk + exogenous jump at 10:00)
    n_min = int(RTH_DURATION / 60) + 1
    price = np.zeros(n_min)
    price[0] = 1200.0
    shock_min = time_to_sec(10, 0) // 60  # exogenous shock at 10:00
    for i in range(1, n_min):
        price[i] = price[i-1] + rng.normal(0, 0.5)
        if i == shock_min:
            price[i] -= 15.0  # large exogenous drop
    volume = rng.integers(500, 2000, size=n_min).astype(float)
    volume[shock_min] *= 5  # elevated volume at shock
    return times, price, volume

def simulate_endogenous_day(seed: int = 200) -> tuple:
    """
    May 6 - flash crash / endogenous scenario.
    Very high n (~0.92-0.95) around 14:30-15:00.
    Model as a day with elevated n during crash period.
    """
    rng = np.random.default_rng(seed)
    # Pre-crash: normal
    crash_start = time_to_sec(14, 30)
    crash_end = time_to_sec(15, 0)
    # Generate events in three segments:
    # 09:30-14:30: normal
    t1 = simulate_hawkes_ogata(1.0, 0.78, 4.0, T=crash_start, t0=0.0, seed=seed)
    # 14:30-15:00: high n (endogenous crisis)
    t2 = simulate_hawkes_ogata(2.0, 0.93, 5.0, T=crash_end-crash_start, t0=0.0, seed=seed+1)
    t2 = t2 + crash_start
    # 15:00-16:15: recovery
    t3 = simulate_hawkes_ogata(1.5, 0.82, 4.0, T=RTH_DURATION-crash_end, t0=0.0, seed=seed+2)
    t3 = t3 + crash_end
    all_times = np.sort(np.concatenate([t1, t2, t3]))
    all_times = round_to_seconds(all_times)
    # Price: large drop during flash crash
    n_min = int(RTH_DURATION / 60) + 1
    price = np.zeros(n_min)
    price[0] = 1200.0
    crash_min_start = int(crash_start / 60)
    crash_min_end = int(crash_end / 60)
    for i in range(1, n_min):
        price[i] = price[i-1] + rng.normal(0, 0.5)
        if crash_min_start <= i < crash_min_end:
            price[i] -= 0.8  # rapid decline during crash
    volume = rng.integers(500, 2000, size=n_min).astype(float)
    volume[crash_min_start:crash_min_end] *= 8  # extreme volume
    return all_times, price, volume

def compute_intraday_hawkes(times: np.ndarray, window_min: int = 10,
                             step_min: int = 5, n_bootstrap: int = 10) -> pd.DataFrame:
    """Compute rolling 10-minute Hawkes n through the trading day."""
    windows = create_rolling_windows(times, 0.0, RTH_DURATION, window_min, step_min)
    rows = []
    for w_start, w_end in windows:
        mask = (times >= w_start) & (times <= w_end)
        w_times = times[mask]
        boot = bootstrap_window(w_times, w_start, w_end, n_bootstrap=n_bootstrap,
                                seed_base=int(w_start))
        clock_min = w_end / 60  # minutes since session start
        clock_time = pd.Timestamp('2010-01-01') + pd.Timedelta(minutes=570 + clock_min)  # 9:30 + offset
        rows.append({
            'window_end_min': clock_min,
            'clock_time': clock_time,
            'n_mean': np.mean(boot['ns']) if boot['ns'] else np.nan,
            'n_std': np.std(boot['ns']) if boot['ns'] else np.nan,
            'mu_mean': np.mean(boot['mus']) if boot['mus'] else np.nan,
            'n_events': boot['n_events'],
            'accepted': boot['accepted'],
        })
    return pd.DataFrame(rows)

def run_experiment_5() -> None:
    """Run case study for April 27 (exogenous) vs May 6 (endogenous)."""
    print('  Simulating April 27, 2010 (exogenous)...')
    apr27_times, apr27_price, apr27_vol = simulate_exogenous_day(seed=100)
    apr27_hawkes = compute_intraday_hawkes(apr27_times, n_bootstrap=10)
    apr27_hawkes['day'] = 'Apr27_exogenous'

    print('  Simulating May 6, 2010 (flash crash)...')
    may6_times, may6_price, may6_vol = simulate_endogenous_day(seed=200)
    may6_hawkes = compute_intraday_hawkes(may6_times, n_bootstrap=10)
    may6_hawkes['day'] = 'May6_flashcrash'

    # Save
    combined = pd.concat([apr27_hawkes, may6_hawkes], ignore_index=True)
    combined.to_csv(RESULTS_DIR / 'exp5_case_study.csv', index=False)

    # Print summary
    print('\nApril 27 n stats:', apr27_hawkes['n_mean'].agg(['mean', 'max', 'min']).to_dict())
    print('May 6 n stats:', may6_hawkes['n_mean'].agg(['mean', 'max', 'min']).to_dict())
    print('May 6 n peak (should be near 0.95):', may6_hawkes['n_mean'].max())
    return combined

if __name__ == '__main__':
    print('=== Experiment 5: Case Study Apr 27 vs May 6, 2010 ===')
    run_experiment_5()
    print('Exp 5 done.')
