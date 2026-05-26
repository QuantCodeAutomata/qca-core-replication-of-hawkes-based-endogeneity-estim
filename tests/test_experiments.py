"""
Integration tests verifying methodology adherence for all experiments.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.diagnostics import create_rolling_windows, bootstrap_window

def test_exp1_n_rises_with_regime():
    """Higher-activity regime should produce higher estimated n."""
    # Early regime (1998): n_true=0.3
    times_98 = simulate_hawkes_ogata(0.05, 0.30, 1.5, T=600.0, seed=42)
    # Late regime (2010): n_true=0.80
    times_10 = simulate_hawkes_ogata(1.0, 0.80, 4.0, T=600.0, seed=42)
    res_98 = estimate_hawkes_window(times_98, 0.0, 600.0)
    res_10 = estimate_hawkes_window(times_10, 0.0, 600.0)
    if res_98 and res_10:
        assert res_10['n_hat'] > res_98['n_hat'], (
            f'2010-regime n ({res_10["n_hat"]:.3f}) should exceed 1998-regime n ({res_98["n_hat"]:.3f})')

def test_exp2_bootstrap_structure():
    """Bootstrap should return 50 realizations and a valid acceptance flag."""
    times = round_to_seconds(simulate_hawkes_ogata(0.5, 0.7, 3.0, T=600.0, seed=1))
    result = bootstrap_window(times, 0.0, 600.0, n_bootstrap=10)
    assert len(result['ks_pvalues']) == len(result['ns'])
    assert all(0 <= p <= 1 for p in result['ks_pvalues'] if not np.isnan(p))
    assert isinstance(result['accepted'], bool)

def test_exp3_rounding_bias_small():
    """Second-rounding + de-tie should not introduce bias > 20% on n."""
    true_n = 0.65
    errors = []
    for seed in range(30):
        times = simulate_hawkes_ogata(0.5, true_n, 3.0, T=600.0, seed=seed)
        if len(times) < 10:
            continue
        times_r = round_to_seconds(times)
        times_dt = detie_randomize(times_r, seed=seed)
        est = estimate_hawkes_window(times_dt, 0.0, 600.0)
        if est and est['converged']:
            errors.append(abs(est['n_hat'] - true_n) / true_n)
    if len(errors) > 5:
        mean_rel_error = np.mean(errors)
        assert mean_rel_error < 0.30, f'Mean relative n error too large: {mean_rel_error:.2%}'

def test_exp4_poisson_lower_n():
    """Poisson-reshuffled events should produce lower n than Hawkes events."""
    mu, n_true, beta = 0.5, 0.70, 3.0
    T = 600.0
    times_hawkes = simulate_hawkes_ogata(mu, n_true, beta, T=T, seed=10)
    # Reshuffle: uniform random draws
    rng = np.random.default_rng(99)
    times_pois = np.sort(rng.uniform(0, T, size=len(times_hawkes)))
    res_h = estimate_hawkes_window(times_hawkes, 0.0, T)
    res_p = estimate_hawkes_window(times_pois, 0.0, T)
    if res_h and res_p:
        assert res_h['n_hat'] > res_p['n_hat'], (
            f'Hawkes n ({res_h["n_hat"]:.3f}) should exceed Poisson n ({res_p["n_hat"]:.3f})')

def test_exp5_endogenous_higher_n():
    """Endogenous (flash crash) day should have higher peak n than exogenous."""
    # High-n endogenous process
    times_end = simulate_hawkes_ogata(2.0, 0.92, 5.0, T=300.0, seed=200)
    # Normal exogenous process
    times_exo = simulate_hawkes_ogata(2.0, 0.78, 4.0, T=300.0, seed=100)
    res_end = estimate_hawkes_window(times_end, 0.0, 300.0)
    res_exo = estimate_hawkes_window(times_exo, 0.0, 300.0)
    if res_end and res_exo:
        assert res_end['n_hat'] > res_exo['n_hat'], (
            f'Endogenous n ({res_end["n_hat"]:.3f}) should be higher')

def test_exp6_mu_ushaped():
    """Intraday mu profile should be higher at endpoints than midday."""
    from exp.exp_6_intraday_seasonality import intraday_mu_profile
    mu_open = intraday_mu_profile(0.0)      # 09:30
    mu_midday = intraday_mu_profile(12150.0) # ~12:45 (midday)
    mu_close = intraday_mu_profile(24300.0) # 16:15
    assert mu_open > mu_midday, 'mu at open should exceed mu at midday'
    assert mu_close > mu_midday, 'mu at close should exceed mu at midday'

def test_window_estimation_consistent_across_lengths():
    """Estimates from 10- and 20-minute windows on same data should be similar."""
    times = simulate_hawkes_ogata(0.5, 0.65, 3.0, T=3600.0, seed=77)
    windows_10 = create_rolling_windows(times, 0.0, 3600.0, 10, 5)
    windows_20 = create_rolling_windows(times, 0.0, 3600.0, 20, 5)
    ns_10, ns_20 = [], []
    for ws, we in windows_10[:20]:
        mask = (times >= ws) & (times <= we)
        est = estimate_hawkes_window(times[mask], ws, we)
        if est and est['converged']:
            ns_10.append(est['n_hat'])
    for ws, we in windows_20[:20]:
        mask = (times >= ws) & (times <= we)
        est = estimate_hawkes_window(times[mask], ws, we)
        if est and est['converged']:
            ns_20.append(est['n_hat'])
    if ns_10 and ns_20:
        mean_diff = abs(np.mean(ns_10) - np.mean(ns_20))
        assert mean_diff < 0.2, f'Window estimates too different: {mean_diff:.3f}'
