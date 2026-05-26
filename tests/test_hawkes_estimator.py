"""
Unit tests for the Hawkes MLE estimator.
Tests methodology adherence: custom exponential kernel, parameter recovery, etc.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.hawkes.estimator import hawkes_log_likelihood, estimate_hawkes_window, compute_compensator
from src.hawkes.simulator import simulate_hawkes_ogata

def test_log_likelihood_not_nan():
    """Log-likelihood should return a finite value for valid inputs."""
    rng = np.random.default_rng(42)
    times = np.sort(rng.uniform(0, 100, 50))
    params = np.array([np.log(0.1), np.log(0.5 / 0.5), np.log(2.0)])  # log_mu, logit_n, log_beta
    ll = hawkes_log_likelihood(params, times, T=100.0, t0=0.0)
    assert np.isfinite(ll), f'Log-likelihood is not finite: {ll}'

def test_estimate_returns_valid_params():
    """Estimator should return valid parameter estimates for simulated data."""
    times = simulate_hawkes_ogata(mu=0.5, n=0.6, beta=2.0, T=600.0, seed=42)
    assert len(times) > 10, 'Too few simulated events'
    result = estimate_hawkes_window(times, t_start=0.0, t_end=600.0)
    assert result is not None
    assert 0 < result['mu_hat'], 'mu must be positive'
    assert 0 <= result['n_hat'] < 1, f'n must be in [0,1): got {result["n_hat"]}'
    assert 0 < result['beta_hat'], 'beta must be positive'

def test_n_bounded():
    """n_hat must always be in [0, 0.999] - stationarity condition."""
    for seed in range(10):
        times = simulate_hawkes_ogata(mu=0.3, n=0.7, beta=3.0, T=600.0, seed=seed)
        result = estimate_hawkes_window(times, 0.0, 600.0)
        if result is not None:
            assert result['n_hat'] < 1.0, f'n_hat >= 1: {result["n_hat"]}'
            assert result['n_hat'] >= 0.0, f'n_hat < 0: {result["n_hat"]}'

def test_parameter_recovery_high_activity():
    """For high event counts, estimator should recover parameters reasonably."""
    true_n = 0.7
    times = simulate_hawkes_ogata(mu=1.0, n=true_n, beta=4.0, T=3600.0, seed=99)
    result = estimate_hawkes_window(times, 0.0, 3600.0)
    assert result is not None
    # With enough data, n should be within 30% of truth
    assert abs(result['n_hat'] - true_n) < 0.3, f'n_hat={result["n_hat"]:.3f} far from truth={true_n}'

def test_empty_window_returns_none():
    """Window with no events should return None."""
    result = estimate_hawkes_window(np.array([]), 0.0, 600.0)
    assert result is None

def test_single_event_returns_none():
    """Window with single event should return None (not enough data)."""
    result = estimate_hawkes_window(np.array([100.0]), 0.0, 600.0)
    assert result is None

def test_compensator_nonnegative():
    """Compensator values must be non-negative and non-decreasing."""
    times = np.array([1.0, 2.5, 5.0, 8.0, 10.0])
    comp = compute_compensator(times, mu=0.1, n=0.6, beta=2.0, t0=0.0)
    assert all(comp >= 0), 'Compensator values must be non-negative'
    assert all(np.diff(comp) >= 0), 'Compensator must be non-decreasing'

def test_large_n_closer_to_1():
    """Hawkes process with true n=0.9 should estimate n much higher than n=0.3."""
    low_n_times = simulate_hawkes_ogata(mu=0.5, n=0.3, beta=3.0, T=600.0, seed=7)
    high_n_times = simulate_hawkes_ogata(mu=0.5, n=0.9, beta=3.0, T=600.0, seed=7)
    res_low = estimate_hawkes_window(low_n_times, 0.0, 600.0)
    res_high = estimate_hawkes_window(high_n_times, 0.0, 600.0)
    if res_low and res_high:
        assert res_high['n_hat'] > res_low['n_hat'], (
            f'High-n process ({res_high["n_hat"]:.3f}) should have higher n_hat '
            f'than low-n process ({res_low["n_hat"]:.3f})')
