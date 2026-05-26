"""
Simulation of the univariate exponential Hawkes process.

Two backends are provided:
* ``simulate_hawkes_ogata`` — pure-NumPy Ogata thinning algorithm.
* ``simulate_hawkes_tick``  — wraps the *tick* library's SimuHawkesExpKernels.

Utility helpers ``round_to_seconds`` and ``detie_randomize`` mirror the
de-tying procedure described in the companion paper.
"""

from __future__ import annotations

import numpy as np
from tick.hawkes import SimuHawkesExpKernels


# ---------------------------------------------------------------------------
# Core simulators
# ---------------------------------------------------------------------------

def simulate_hawkes_ogata(mu: float, n: float, beta: float,
                           T: float, t0: float = 0.0,
                           seed: int | None = None) -> np.ndarray:
    """Simulate a univariate exponential Hawkes process via Ogata thinning.

    The intensity at time t is:
        lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta*(t - t_i))
    where alpha = n * beta.

    A running excitation sum S is maintained so that updating the upper
    bound M = mu + alpha*S costs O(1) per candidate, giving O(N) overall.

    Parameters
    ----------
    mu : float
        Background (base) intensity (events / second).
    n : float
        Branching ratio, 0 < n < 1.
    beta : float
        Exponential decay rate (1 / second).
    T : float
        End of simulation interval (seconds).
    t0 : float
        Start of simulation interval (seconds).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted array of event times in [t0, T].
    """
    rng = np.random.default_rng(seed)
    alpha = n * beta

    events: list[float] = []
    t = t0
    # S = sum_{accepted t_j <= t} exp(-beta*(t - t_j))  at current position t
    S = 0.0
    lambda_star = mu  # tight upper bound at t0 (no history)

    while True:
        # Inter-arrival of next Poisson candidate with rate lambda_star
        w = -np.log(rng.random()) / lambda_star
        t_cand = t + w

        if t_cand > T:
            break

        # Decay S forward to t_cand and compute true intensity
        S_cand = S * np.exp(-beta * (t_cand - t))
        lam_cand = mu + alpha * S_cand

        if rng.random() <= lam_cand / lambda_star:
            # Accept: register new event
            events.append(t_cand)
            S = S_cand + 1.0              # accepted event contributes exp(0)=1
            lambda_star = mu + alpha * S  # intensity jumps up by alpha
        else:
            # Reject: tighten bound to actual value at t_cand
            S = S_cand
            lambda_star = lam_cand

        t = t_cand

    return np.array(events, dtype=np.float64)


def simulate_hawkes_tick(mu: float, n: float, beta: float,
                          T: float, seed: int | None = None) -> np.ndarray:
    """Simulate a univariate exponential Hawkes process using the tick library.

    # Using tick SimuHawkesExpKernels - Context7 confirmed

    Parameters
    ----------
    mu : float
        Background intensity.
    n : float
        Branching ratio (adjacency coefficient).
    beta : float
        Exponential decay rate.
    T : float
        End of simulation interval (simulation starts at 0).
    seed : int or None
        RNG seed; pass ``None`` for a non-reproducible run.

    Returns
    -------
    np.ndarray
        Sorted array of event times in [0, T].
    """
    tick_seed = -1 if seed is None else int(seed)

    simu = SimuHawkesExpKernels(
        adjacency=[[n]],   # 1×1 branching matrix
        decays=[[beta]],
        baseline=[mu],
        end_time=T,
        seed=tick_seed,
        verbose=False,
    )
    simu.simulate()
    return np.asarray(simu.timestamps[0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def round_to_seconds(times: np.ndarray) -> np.ndarray:
    """Round continuous event times to the nearest integer second.

    Parameters
    ----------
    times : np.ndarray
        Array of event times (seconds, arbitrary precision).

    Returns
    -------
    np.ndarray
        Integer-valued times stored as float64.
    """
    return np.round(times).astype(np.float64)


def detie_randomize(times: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Break ties in integer-second event times with Uniform(0, 1) jitter.

    For each event a jitter value is drawn i.i.d. from Uniform[0, 1) and
    added to its integer-second timestamp, placing it randomly within the
    same one-second bin.  The output array is re-sorted.

    # Custom - Context7 found no library equivalent (paper de-tie procedure)

    Parameters
    ----------
    times : np.ndarray
        Event times, typically integer-valued after ``round_to_seconds``.
    seed : int or None
        RNG seed for reproducibility of the jitter.

    Returns
    -------
    np.ndarray
        Sorted de-tied event times.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.0, 1.0, size=len(times))
    return np.sort(times.astype(np.float64) + jitter)
