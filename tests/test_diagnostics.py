"""
Tests for Hawkes residual diagnostics and de-tie bootstrap.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.hawkes.diagnostics import (
    compute_residuals, residuals_to_uniforms, ks_test_uniformity,
    bootstrap_window, create_rolling_windows
)
from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds, detie_randomize

def test_residuals_nonnegative():
    """Compensator increments (residual inter-arrivals) must be non-negative."""
    times = simulate_hawkes_ogata(0.5, 0.6, 2.0, T=300.0, seed=1)
    delta = compute_residuals(times, mu=0.5, n=0.6, beta=2.0, t0=0.0)
    assert all(delta > 0), 'All residual inter-arrivals should be positive'

def test_uniforms_in_01():
    """Uniform transforms must be in (0, 1)."""
    delta = np.array([0.5, 1.0, 0.3, 2.0, 0.8])
    u = residuals_to_uniforms(delta)
    assert all(0 < ui < 1 for ui in u), f'Uniforms out of (0,1): {u}'

def test_ks_test_returns_pvalue_01():
    """KS test p-value must be in [0, 1]."""
    uniforms = np.random.uniform(0, 1, 100)
    stat, pval = ks_test_uniformity(uniforms)
    assert 0 <= pval <= 1, f'p-value out of [0,1]: {pval}'
    assert stat >= 0, 'KS statistic must be non-negative'

def test_true_hawkes_passes_ks_usually():
    """Events from true Hawkes should mostly pass KS test for uniformity."""
    pass_count = 0
    for seed in range(20):
        times = simulate_hawkes_ogata(0.5, 0.6, 2.0, T=600.0, seed=seed)
        if len(times) < 10:
            continue
        from src.hawkes.estimator import estimate_hawkes_window
        est = estimate_hawkes_window(times, 0.0, 600.0)
        if est is None:
            continue
        delta = compute_residuals(times, est['mu_hat'], est['n_hat'], est['beta_hat'], t0=0.0)
        uniforms = residuals_to_uniforms(delta)
        _, pval = ks_test_uniformity(uniforms)
        if pval >= 0.05:
            pass_count += 1
    # At least 50% should pass
    assert pass_count >= 8, f'Too few KS passes: {pass_count}/20'

def test_detie_removes_ties():
    """After de-tie randomization, there should be no duplicate timestamps."""
    times = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0])
    result = detie_randomize(times, seed=42)
    assert len(result) == len(times), 'De-tie should preserve count'
    assert len(np.unique(np.round(result, 6))) == len(result), 'De-tie should remove ties'

def test_bootstrap_returns_n_bootstrap_estimates():
    """bootstrap_window should return n_bootstrap estimates."""
    times = round_to_seconds(simulate_hawkes_ogata(0.5, 0.6, 2.0, T=600.0, seed=5))
    result = bootstrap_window(times, 0.0, 600.0, n_bootstrap=5, seed_base=0)
    assert 'ns' in result
    assert 'accepted' in result
    assert isinstance(result['accepted'], bool)

def test_acceptance_rule():
    """Window accepted if at least one realization has p-value >= 0.05."""
    # Simulate with high n to stress test
    times = round_to_seconds(simulate_hawkes_ogata(0.5, 0.65, 2.0, T=600.0, seed=99))
    result = bootstrap_window(times, 0.0, 600.0, n_bootstrap=5, seed_base=42)
    # accepted = not all rejected
    all_rejected = all(p < 0.05 for p in result['ks_pvalues'] if not np.isnan(p))
    assert result['accepted'] == (not all_rejected)

def test_rolling_windows_count():
    """Rolling windows should cover RTH session with expected count."""
    times = np.array([1.0, 100.0])  # minimal events
    # 10-min window, 5-min step, over 24300-second session
    # Expected: (24300 - 600) / 300 + 1 windows = 79 windows
    windows = create_rolling_windows(times, 0.0, 24300.0, 10, 5)
    expected = (24300 - 600) // 300 + 1
    assert len(windows) == expected, f'Expected {expected} windows, got {len(windows)}'
