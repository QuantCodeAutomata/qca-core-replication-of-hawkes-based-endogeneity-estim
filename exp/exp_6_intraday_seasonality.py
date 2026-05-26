"""
Exp 6: Intraday seasonality - mu exhibits U-shape, n is comparatively flat.
Using synthetic data for March 2009 (year ~2009, t=0.846).
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize
from src.hawkes.diagnostics import create_rolling_windows

RESULTS_DIR = Path('/workspace/project/results')
RTH_DURATION = 24300

def intraday_mu_profile(t_sec: float) -> float:
    """
    Intraday U-shaped mu profile: higher at open/close, lower midday.
    t_sec: seconds since 09:30 (0 to 24300)
    Returns: relative multiplier for mu.
    """
    # U-shape: use cosine with period = full session
    norm = t_sec / RTH_DURATION  # 0 to 1
    # U-shape: high at 0 and 1, low at 0.5
    return 1.5 + np.cos(2 * np.pi * norm)  # ranges from 0.5 (midday) to 2.5 (open/close)

def simulate_intraday_hawkes_day(mu_base: float, n_true: float, beta: float,
                                  seed: int = 42, n_bins: int = 25) -> np.ndarray:
    """
    Simulate a trading day with intraday-varying mu (U-shape) and constant n.
    Split day into n_bins, each with its own mu drawn from the U-shape profile.
    Returns event times over the full day [0, RTH_DURATION].
    """
    bin_duration = RTH_DURATION / n_bins
    all_times = []
    for b in range(n_bins):
        t_center = (b + 0.5) * bin_duration
        mu_b = mu_base * intraday_mu_profile(t_center)
        t0 = b * bin_duration
        T_b = (b + 1) * bin_duration
        seeded = seed + b
        times_b = simulate_hawkes_ogata(mu_b, n_true, beta, T=bin_duration, t0=0.0, seed=seeded)
        times_b = times_b + t0
        all_times.append(times_b)
    return round_to_seconds(np.concatenate(all_times))

def run_experiment_6(n_days: int = 22, window_lengths: list = [10, 20, 30]) -> pd.DataFrame:
    """
    Analyze intraday seasonality in mu and n for March 2009.
    22 trading days in March 2009.
    """
    # 2009: t = (2009 - 1998) / 12 = 0.917
    t = (2009 - 1998) / (2010 - 1998)
    mu_base = 0.05 + t * 0.95  # ~0.92
    n_true = 0.30 + t * 0.50   # ~0.758
    beta = 1.5 + t * 2.5        # ~3.79

    all_results = []
    for day_idx in range(n_days):
        seed = 9000 + day_idx
        times = simulate_intraday_hawkes_day(mu_base, n_true, beta, seed=seed)
        for wl in window_lengths:
            windows = create_rolling_windows(times, 0.0, RTH_DURATION, wl, step_minutes=5)
            for w_start, w_end in windows:
                mask = (times >= w_start) & (times <= w_end)
                w_times = times[mask]
                if len(w_times) < 5:
                    continue
                w_dt = detie_randomize(w_times, seed=seed * 100 + int(w_start))
                est = estimate_hawkes_window(w_dt, w_start, w_end)
                if est is not None:
                    # 10-min bin index for clock-time alignment
                    bin_idx = int(w_end / 600)  # 10-minute bins
                    all_results.append({
                        'day': day_idx,
                        'w_start': w_start,
                        'w_end': w_end,
                        'bin_10min': bin_idx,
                        'window_min': wl,
                        'n_hat': est['n_hat'],
                        'mu_hat': est['mu_hat'],
                        'beta_hat': est['beta_hat'],
                        'n_events': est['n_events'],
                        'converged': est['converged'],
                    })
        if day_idx % 5 == 0:
            print(f'  Day {day_idx+1}/{n_days} processed...')

    df = pd.DataFrame(all_results)
    df.to_parquet(RESULTS_DIR / 'exp6_intraday.parquet', index=False)
    return df

def summarize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Average n and mu by clock-time bin and window length."""
    summary = df.groupby(['bin_10min', 'window_min']).agg(
        n_mean=('n_hat', 'mean'),
        n_p10=('n_hat', lambda x: np.percentile(x, 10)),
        n_p90=('n_hat', lambda x: np.percentile(x, 90)),
        mu_mean=('mu_hat', 'mean'),
        mu_p10=('mu_hat', lambda x: np.percentile(x, 10)),
        mu_p90=('mu_hat', lambda x: np.percentile(x, 90)),
        n_events_mean=('n_events', 'mean'),
        count=('n_hat', 'count'),
    ).reset_index()
    # Add clock time label
    summary['clock_min'] = (summary['bin_10min'] * 10)  # minutes from open
    summary.to_csv(RESULTS_DIR / 'exp6_intraday_summary.csv', index=False)
    return summary

def simulate_intraday_session(n_true: float = 0.65, beta: float = 3.0,
                              window_min: int = 20, step_min: int = 5,
                              seed: int = 0) -> pd.DataFrame:
    """Simulate one synthetic session and return rolling Hawkes estimates.

    Returns DataFrame with columns: clock_min, n_hat, mu_hat, beta_hat, converged.
    """
    rng = np.random.default_rng(seed)
    n_bins = int(RTH_DURATION / 60)
    mu_base = 0.5
    all_times: list = []
    for chunk in range(n_bins):
        t0_chunk = float(chunk * 60)
        mu_chunk = mu_base * intraday_mu_profile(t0_chunk + 30.0)
        chunk_times = simulate_hawkes_ogata(
            mu=mu_chunk, n=n_true, beta=beta, T=60.0,
            seed=int(rng.integers(0, 2 ** 31))
        )
        all_times.extend(chunk_times + t0_chunk)

    times = round_to_seconds(np.sort(np.array(all_times)))
    times_dt = detie_randomize(times, seed=seed)

    windows = create_rolling_windows(times_dt, 0.0, RTH_DURATION, window_min, step_min)
    rows = []
    for ws, we in windows:
        mask = (times_dt >= ws) & (times_dt <= we)
        est = estimate_hawkes_window(times_dt[mask], ws, we)
        if est is None:
            continue
        rows.append({
            'clock_min': (ws + we) / 2.0 / 60.0,
            'n_hat': est['n_hat'],
            'mu_hat': est['mu_hat'],
            'beta_hat': est['beta_hat'],
            'converged': est['converged'],
        })
    return pd.DataFrame(rows)


def run_exp6(n_days: int = 21, n_true: float = 0.65, beta: float = 3.0,
             window_lengths: tuple = (10, 20, 30), step_min: int = 5,
             save_path: str = None) -> pd.DataFrame:
    """Run Experiment 6 across n_days synthetic sessions.

    Returns summary DataFrame with cross-day percentile bands for mu and n
    as a function of intraday clock time. Compatible with plot_intraday_seasonality.
    """
    all_rows = []
    for day in range(n_days):
        for wl in window_lengths:
            df_day = simulate_intraday_session(
                n_true=n_true, beta=beta, window_min=wl, step_min=step_min,
                seed=day * 100 + wl
            )
            if df_day.empty:
                continue
            df_day['day'] = day
            df_day['window_min'] = wl
            all_rows.append(df_day)

    if not all_rows:
        return pd.DataFrame()

    raw = pd.concat(all_rows, ignore_index=True)
    summary = (
        raw.groupby(['window_min', 'clock_min'])
        .agg(
            mu_mean=('mu_hat', 'mean'),
            mu_p10=('mu_hat', lambda x: np.percentile(x, 10)),
            mu_p90=('mu_hat', lambda x: np.percentile(x, 90)),
            n_mean=('n_hat', 'mean'),
            n_p10=('n_hat', lambda x: np.percentile(x, 10)),
            n_p90=('n_hat', lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )
    if save_path:
        summary.to_parquet(save_path, index=False)
        print(f'Saved: {save_path}')
    return summary


if __name__ == '__main__':
    print('=== Experiment 6: Intraday Seasonality of mu and n ===')
    df = run_experiment_6(n_days=10)
    summary = summarize_intraday(df)
    print('\nIntraday n averages by window length:')
    print(summary.groupby('window_min')['n_mean'].agg(['mean', 'std']))
    print('\nIntraday mu range (showing U-shape):')
    print(summary[summary['window_min'] == 10][['clock_min', 'mu_mean']].to_string())
    print('Exp 6 done.')
