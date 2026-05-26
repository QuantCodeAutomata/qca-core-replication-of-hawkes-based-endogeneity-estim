"""
Maximum-likelihood estimation for the univariate exponential Hawkes process.

Intensity:  lambda(t) = mu + n*beta * sum_{t_i < t} exp(-beta*(t - t_i))
where n = alpha/beta is the branching ratio, 0 < n < 1.

All times are in seconds.  Parameters are optimised in an unconstrained
space via the transforms:
    log_mu   = log(mu)
    logit_n  = log(n / (0.999 - n))   [maps (0, 0.999) -> (-inf, +inf)]
    log_beta = log(beta)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numba import jit
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Numba inner loops
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _loglik_inner(times: np.ndarray, mu: float, n: float, beta: float,
                  T: float, t0: float) -> float:
    """Return the *negative* log-likelihood using the recursive R trick.

    R_i = exp(-beta*(t_i - t_{i-1})) * (R_{i-1} + 1),   R_0 = 0
    lambda(t_i) = mu + n*beta * R_i
    integral = mu*(T-t0) + n * sum_i (1 - exp(-beta*(T - t_i)))

    The zero-dt guard (dt == 0 → decay factor = 1) prevents nan from
    exp(-inf * 0) when beta overflows and duplicate timestamps are present.
    """
    N = len(times)
    log_sum = 0.0
    R = 0.0

    for i in range(N):
        if i > 0:
            dt = times[i] - times[i - 1]
            if dt > 0.0:
                R = np.exp(-beta * dt) * (R + 1.0)
            else:
                R = R + 1.0  # exp(-beta * 0) == 1 for any finite beta
        lam = mu + n * beta * R
        if lam <= 0.0 or lam != lam:  # <= 0 or NaN
            return 1e18
        log_sum += np.log(lam)

    # Compensating integral  ∫_{t0}^{T} lambda(u) du
    integral_excit = 0.0
    for i in range(N):
        gap = T - times[i]
        if gap > 0.0:
            integral_excit += 1.0 - np.exp(-beta * gap)
        # gap == 0 contributes 0 to the sum (event exactly at T)

    integral = mu * (T - t0) + n * integral_excit
    return -(log_sum - integral)


@jit(nopython=True, cache=True)
def _compensator_inner(times: np.ndarray, mu: float, n: float, beta: float,
                       t0: float) -> np.ndarray:
    """Compute the compensator Lambda(t_i) for each event time.

    Exploits:
        Lambda(t_i) = mu*(t_i - t0) + n*(i - R_i)
    where R_i = sum_{j<i} exp(-beta*(t_i - t_j)) satisfies the recursion
        R_i = exp(-beta*(t_i - t_{i-1})) * (R_{i-1} + 1),   R_0 = 0.

    The zero-dt guard mirrors ``_loglik_inner``.
    """
    N = len(times)
    out = np.empty(N)
    R = 0.0
    for i in range(N):
        if i > 0:
            dt = times[i] - times[i - 1]
            if dt > 0.0:
                R = np.exp(-beta * dt) * (R + 1.0)
            else:
                R = R + 1.0
        # number of *previous* events is i (0-indexed)
        out[i] = mu * (times[i] - t0) + n * (i - R)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hawkes_log_likelihood(params: np.ndarray, times: np.ndarray,
                           T: float, t0: float = 0.0) -> float:
    """Negative log-likelihood for the univariate exponential Hawkes process.

    Parameters
    ----------
    params : array-like, shape (3,)
        [log_mu, logit_n, log_beta] — unconstrained optimisation parameters.
    times : np.ndarray
        Sorted event times in seconds.  Only events in [t0, T] are used.
    T : float
        End of observation window (seconds).
    t0 : float
        Start of observation window (seconds).

    Returns
    -------
    float
        Negative log-likelihood value (scalar); returns 1e18 on infeasibility.
    """
    log_mu, logit_n, log_beta = params[0], params[1], params[2]

    mu = np.exp(log_mu)
    n = 0.999 * expit(logit_n)          # maps R -> (0, 0.999)
    beta = np.exp(log_beta)

    if len(times) == 0:
        return mu * (T - t0)            # only the base rate contributes

    return _loglik_inner(times, mu, n, beta, T, t0)


def estimate_hawkes_window(times: np.ndarray, t_start: float, t_end: float,
                            n_starts: int = 3) -> Optional[dict]:
    """Estimate Hawkes parameters for events within [t_start, t_end].

    Uses MLE with L-BFGS-B in unconstrained (transformed) parameter space.
    Multiple random starting points are tried for robustness.

    Parameters
    ----------
    times : np.ndarray
        All event times in seconds (need not be pre-filtered).
    t_start : float
        Window start in seconds.
    t_end : float
        Window end in seconds.
    n_starts : int
        Number of distinct starting points to try.

    Returns
    -------
    dict or None
        ``None`` when fewer than 5 events fall in the window.
        Otherwise a dict with keys:
        ``mu_hat``, ``n_hat``, ``beta_hat``, ``log_lik``,
        ``n_events``, ``converged``, ``hit_bound``.
    """
    mask = (times >= t_start) & (times <= t_end)
    win_times = np.asarray(times[mask], dtype=np.float64)
    n_events = len(win_times)

    if n_events < 5:
        return None

    T = t_end
    t0 = t_start

    # ------------------------------------------------------------------
    # Starting points: (mu, n, beta) in natural space -> transformed
    # ------------------------------------------------------------------
    def _to_unconstrained(mu_0: float, n_0: float, beta_0: float) -> np.ndarray:
        logit_n_0 = np.log(n_0 / (0.999 - n_0))
        return np.array([np.log(mu_0), logit_n_0, np.log(beta_0)])

    # Data-driven rate estimate for mu initialisation
    rate_approx = n_events / (t_end - t_start)

    candidate_starts = [
        _to_unconstrained(max(rate_approx * 0.2, 1e-4), 0.50, 1.0),
        _to_unconstrained(max(rate_approx * 0.5, 1e-4), 0.30, 2.0),
        _to_unconstrained(max(rate_approx * 0.1, 1e-4), 0.70, 5.0),
        _to_unconstrained(max(rate_approx * 0.8, 1e-4), 0.15, 0.5),
        _to_unconstrained(max(rate_approx * 0.4, 1e-4), 0.60, 3.0),
    ][:n_starts]

    best_nll = np.inf
    best_result = None

    # Wide bounds prevent exp-overflow while still covering any realistic regime.
    # log_mu  in (-15,  15) → mu  in (3e-7,  3.3e6)
    # logit_n in (-15,  15) → n   in (~0,    ~0.999)
    # log_beta in (-5,  10) → beta in (0.007, 22026)
    _BOUNDS = [(-15.0, 15.0), (-15.0, 15.0), (-5.0, 10.0)]

    for x0 in candidate_starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    hawkes_log_likelihood,
                    x0,
                    args=(win_times, T, t0),
                    method="L-BFGS-B",
                    bounds=_BOUNDS,
                    options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
                )
        except Exception:
            continue

        if res.fun < best_nll:
            best_nll = res.fun
            best_result = res

    if best_result is None:
        return None

    log_mu_hat, logit_n_hat, log_beta_hat = best_result.x
    mu_hat = float(np.exp(log_mu_hat))
    n_hat = float(0.999 * expit(logit_n_hat))
    beta_hat = float(np.exp(log_beta_hat))

    hit_bound = (n_hat > 0.99) or (n_hat < 0.001)

    return {
        "mu_hat": mu_hat,
        "n_hat": n_hat,
        "beta_hat": beta_hat,
        "log_lik": float(-best_nll),
        "n_events": n_events,
        "converged": bool(best_result.success),
        "hit_bound": hit_bound,
    }


def compute_compensator(times: np.ndarray, mu: float, n: float,
                         beta: float, t0: float) -> np.ndarray:
    """Integrated intensity (compensator) evaluated at each event time.

    Lambda(t_i) = mu*(t_i - t0) + n * sum_{j<i} (1 - exp(-beta*(t_i - t_j)))

    Computed recursively in O(N) via:
        Lambda(t_i) = mu*(t_i - t0) + n*(i - R_i)
    where R_i = sum_{j<i} exp(-beta*(t_i - t_j)).

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in seconds.
    mu, n, beta : float
        Fitted Hawkes parameters.
    t0 : float
        Start of the observation window (seconds).

    Returns
    -------
    np.ndarray, shape (N,)
        Compensator value at each event time.
    """
    times = np.asarray(times, dtype=np.float64)
    return _compensator_inner(times, mu, n, beta, t0)
