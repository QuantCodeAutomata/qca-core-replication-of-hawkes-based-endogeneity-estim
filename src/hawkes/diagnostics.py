"""
Residual diagnostics for the fitted univariate exponential Hawkes process.

Under a correctly-specified model the compensator increments
    Delta_i = Lambda(t_i) - Lambda(t_{i-1})
are i.i.d. Exp(1), and the transformed residuals
    U_i = 1 - exp(-Delta_i)
are i.i.d. Uniform[0, 1].  Deviation from uniformity signals mis-fit.

Bootstrap de-tie randomization is used throughout to handle the integer-second
rounding that is typical of high-frequency tick data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest

from src.hawkes.estimator import compute_compensator, estimate_hawkes_window
from src.hawkes.simulator import detie_randomize


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

def compute_residuals(times: np.ndarray, mu: float, n: float,
                       beta: float, t0: float) -> np.ndarray:
    """Compute compensator increments (residual inter-arrivals).

    Under the null model the increments
        Delta_i = Lambda(t_i) - Lambda(t_{i-1}),   Delta_0 = Lambda(t_0)
    are i.i.d. Exp(1).

    Parameters
    ----------
    times : np.ndarray
        Sorted (de-tied) event times in seconds.
    mu, n, beta : float
        Fitted Hawkes parameters.
    t0 : float
        Start of the observation window (seconds).

    Returns
    -------
    np.ndarray, shape (N,)
        Compensator increments Delta_i >= 0.
    """
    xi = compute_compensator(times, mu, n, beta, t0)
    # Delta_0 = Lambda(t_0) - Lambda(t0) = xi[0] (since Lambda(t0) = 0)
    delta = np.empty_like(xi)
    delta[0] = xi[0]
    delta[1:] = np.diff(xi)
    return delta


def residuals_to_uniforms(delta: np.ndarray) -> np.ndarray:
    """Transform Exp(1) residuals to Uniform[0, 1] via U_i = 1 - exp(-Delta_i).

    Parameters
    ----------
    delta : np.ndarray
        Compensator increments (should be i.i.d. Exp(1) under the null).

    Returns
    -------
    np.ndarray
        Values in (0, 1); i.i.d. Uniform[0, 1] under the null.
    """
    return 1.0 - np.exp(-delta)


def ks_test_uniformity(uniforms: np.ndarray) -> tuple[float, float]:
    """One-sample KS test of Uniform[0, 1] against the supplied sample.

    Parameters
    ----------
    uniforms : np.ndarray
        Sample values, nominally Uniform[0, 1] under the null.

    Returns
    -------
    (statistic, pvalue) : tuple of float
        KS test statistic and two-sided p-value.
    """
    stat, pval = kstest(uniforms, "uniform")
    return float(stat), float(pval)


# ---------------------------------------------------------------------------
# Bootstrap de-tie window analysis
# ---------------------------------------------------------------------------

def bootstrap_window(times: np.ndarray, t_start: float, t_end: float,
                      n_bootstrap: int = 50,
                      seed_base: int = 0) -> dict:
    """Run bootstrap de-tie randomizations and Hawkes fitting on one window.

    For each of the ``n_bootstrap`` de-tie realizations the pipeline is:
        1. ``detie_randomize`` — add Uniform(0,1) jitter to integer seconds.
        2. ``estimate_hawkes_window`` — MLE parameter estimation.
        3. ``compute_residuals`` + ``residuals_to_uniforms`` — transform to U[0,1].
        4. ``ks_test_uniformity`` — KS goodness-of-fit test.

    Acceptance rule (from the paper): a window is **rejected** only when
    *all* 50 realizations individually reject at the 5 % level, i.e.
        reject iff max(p_1, ..., p_50) < 0.05.

    Parameters
    ----------
    times : np.ndarray
        Integer-second event times (possibly with ties).
    t_start, t_end : float
        Window boundaries in seconds.
    n_bootstrap : int
        Number of de-tie realizations (default 50).
    seed_base : int
        Seeds used are ``seed_base``, ``seed_base + 1``, …

    Returns
    -------
    dict with keys:
        ``mus``, ``ns``, ``betas`` — lists of fitted parameter values.
        ``ks_pvalues``             — list of KS p-values.
        ``accepted``               — bool; True if window passes the test.
        ``n_events``               — number of events in [t_start, t_end].
    """
    mask = (times >= t_start) & (times <= t_end)
    n_events = int(mask.sum())

    mus: list[float] = []
    ns: list[float] = []
    betas: list[float] = []
    ks_pvalues: list[float] = []

    for b in range(n_bootstrap):
        seed = seed_base + b
        de_tied = detie_randomize(times[mask], seed=seed)

        result = estimate_hawkes_window(de_tied, t_start, t_end)
        if result is None:
            continue

        delta = compute_residuals(
            de_tied, result["mu_hat"], result["n_hat"], result["beta_hat"],
            t_start,
        )
        uniforms = residuals_to_uniforms(delta)
        _, pval = ks_test_uniformity(uniforms)

        mus.append(result["mu_hat"])
        ns.append(result["n_hat"])
        betas.append(result["beta_hat"])
        ks_pvalues.append(pval)

    # Reject only if every realization rejects at 5 %
    if len(ks_pvalues) == 0:
        accepted = False
    else:
        accepted = max(ks_pvalues) >= 0.05

    return {
        "mus": mus,
        "ns": ns,
        "betas": betas,
        "ks_pvalues": ks_pvalues,
        "accepted": accepted,
        "n_events": n_events,
    }


# ---------------------------------------------------------------------------
# Rolling window utilities
# ---------------------------------------------------------------------------

def create_rolling_windows(day_times: np.ndarray,
                            session_start: float, session_end: float,
                            window_minutes: int,
                            step_minutes: int = 5) -> list[tuple[float, float]]:
    """Generate rolling window boundaries over a regular trading session.

    Only windows fully contained within [session_start, session_end] are
    returned.

    Parameters
    ----------
    day_times : np.ndarray
        All event times for the day (seconds from an arbitrary epoch).
        Not directly used in generating boundaries but kept for API clarity.
    session_start : float
        Session open time in seconds.
    session_end : float
        Session close time in seconds.
    window_minutes : int
        Width of each window in minutes.
    step_minutes : int
        Step size between consecutive window starts, in minutes.

    Returns
    -------
    list of (w_start, w_end) tuples
        Window boundaries in seconds.
    """
    window_sec = window_minutes * 60.0
    step_sec = step_minutes * 60.0

    windows: list[tuple[float, float]] = []
    w_start = session_start
    while True:
        w_end = w_start + window_sec
        if w_end > session_end:
            break
        windows.append((w_start, w_end))
        w_start += step_sec

    return windows
