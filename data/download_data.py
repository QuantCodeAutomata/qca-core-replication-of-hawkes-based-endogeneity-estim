"""
Data acquisition and synthetic data generation for the Hawkes estimation study.

Real data
---------
``download_spy_minute`` fetches SPY 1-minute OHLCV bars from the MASSIVE API
(Polygon.io wrapper).  Requires the ``MASSIVE_API_KEY`` environment variable.

Synthetic data
--------------
``generate_synthetic_es_events`` simulates Hawkes event streams for a single
date/regime using the Ogata thinning algorithm.

``generate_full_sample_synthetic`` builds the full 1998-2010 panel, linearly
interpolating the branching ratio n (0.30 → 0.80), base rate mu (0.05 → 1.0),
and decay rate beta (1.5 → 4.0) over the sample.  The full event-level data
is saved to ``data/synthetic_es_events.parquet``; a per-day summary is
returned and written to ``data/synthetic_summary.csv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from massive import RESTClient

# ---------------------------------------------------------------------------
# Locate the data directory relative to this file
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent


def _ensure_data_dir(path: str) -> Path:
    """Return an absolute Path, defaulting to the data/ directory."""
    p = Path(path)
    if not p.is_absolute():
        p = _DATA_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Real-data download
# ---------------------------------------------------------------------------

def download_spy_minute(start_date: str, end_date: str,
                         save_path: str = "data/spy_minute.csv") -> pd.DataFrame:
    """Download SPY 1-minute OHLCV bars from the MASSIVE (Polygon) API.

    Requires the ``MASSIVE_API_KEY`` environment variable to be set.

    Parameters
    ----------
    start_date : str
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date : str
        Inclusive end date in ``YYYY-MM-DD`` format.
    save_path : str
        Destination CSV file path (relative to data/ or absolute).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, open, high, low, close, volume, vwap, n_trades.

    Raises
    ------
    EnvironmentError
        When ``MASSIVE_API_KEY`` is not set.
    """
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MASSIVE_API_KEY environment variable is not set. "
            "Export your Polygon/MASSIVE API key before calling this function."
        )

    client = RESTClient(api_key=api_key)

    records = []
    for agg in client.list_aggs(
        ticker="SPY",
        multiplier=1,
        timespan="minute",
        from_=start_date,
        to=end_date,
        limit=50_000,
        adjusted=True,
    ):
        records.append({
            "timestamp": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC"),
            "open":     agg.open,
            "high":     agg.high,
            "low":      agg.low,
            "close":    agg.close,
            "volume":   agg.volume,
            "vwap":     getattr(agg, "vwap", None),
            "n_trades": getattr(agg, "transactions", None),
        })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    out_path = _ensure_data_dir(save_path)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} SPY minute bars -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# Synthetic ES event generation
# ---------------------------------------------------------------------------

# Regime presets: (mu, n, beta)
_REGIME_PARAMS: dict[str, tuple[float, float, float]] = {
    "1998": (0.05, 0.30, 1.5),
    "2002": (0.15, 0.60, 2.0),
    "2007": (0.50, 0.75, 3.0),
    "2010": (1.00, 0.80, 4.0),
}

# RTH session length in seconds: 6.75 hours
_RTH_SECONDS = int(6.75 * 3600)  # 24 300 s


def generate_synthetic_es_events(n_days: int = 100,
                                   year_regime: str = "1998",
                                   window_minutes: int = 10,
                                   seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ES mid-price change events for a given regime.

    Parameters
    ----------
    n_days : int
        Number of independent trading-day sessions to simulate.
    year_regime : str
        One of ``'1998'``, ``'2002'``, ``'2007'``, ``'2010'``.
    window_minutes : int
        Kept for API consistency; the full 6.75-hour RTH session is always
        simulated and stored.
    seed : int
        Base random seed; each day uses ``seed + day_index``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (YYYY-MM-DD string), ``timestamp_sec`` (float,
        seconds since session start), ``regime`` (year_regime string).
    """
    # Import here to avoid circular imports at module level
    from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds

    if year_regime not in _REGIME_PARAMS:
        raise ValueError(
            f"year_regime must be one of {list(_REGIME_PARAMS)}, got '{year_regime}'"
        )
    mu, n, beta = _REGIME_PARAMS[year_regime]

    rows: list[dict] = []
    base_date = pd.Timestamp("1998-01-02")

    for day_idx in range(n_days):
        date_str = (base_date + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d")
        times = simulate_hawkes_ogata(
            mu=mu, n=n, beta=beta,
            T=float(_RTH_SECONDS), t0=0.0,
            seed=seed + day_idx,
        )
        times_sec = round_to_seconds(times)
        for ts in times_sec:
            rows.append({"date": date_str, "timestamp_sec": ts, "regime": year_regime})

    return pd.DataFrame(rows, columns=["date", "timestamp_sec", "regime"])


# ---------------------------------------------------------------------------
# Full 1998-2010 synthetic panel
# ---------------------------------------------------------------------------

def generate_full_sample_synthetic(start_year: int = 1998,
                                    end_year: int = 2010,
                                    seed: int = 42) -> pd.DataFrame:
    """Generate daily synthetic event data for the full 1998-2010 sample.

    Parameters are linearly interpolated between the 1998 and 2010 presets:
        n    : 0.30 → 0.80
        mu   : 0.05 → 1.00
        beta : 1.50 → 4.00

    ~252 trading days are generated per calendar year.  Event timestamps are
    rounded to the nearest integer second.  The full event-level DataFrame is
    saved to ``data/synthetic_es_events.parquet``; this function returns a
    per-day summary.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range.
    seed : int
        Global RNG seed; each day uses a deterministic offset.

    Returns
    -------
    pd.DataFrame
        Per-day summary with columns:
        ``date``, ``n_events``, ``n_true``, ``mu_true``, ``beta_true``.
    """
    from src.hawkes.simulator import simulate_hawkes_ogata, round_to_seconds

    n_years = end_year - start_year          # = 12 for 1998-2010
    trading_days_per_year = 252
    total_days = n_years * trading_days_per_year + trading_days_per_year  # inclusive

    # Parameter interpolation anchors
    mu_start,   mu_end   = _REGIME_PARAMS["1998"][0], _REGIME_PARAMS["2010"][0]
    n_start,    n_end    = _REGIME_PARAMS["1998"][1], _REGIME_PARAMS["2010"][1]
    beta_start, beta_end = _REGIME_PARAMS["1998"][2], _REGIME_PARAMS["2010"][2]

    event_rows: list[dict] = []
    summary_rows: list[dict] = []

    base_date = pd.Timestamp(f"{start_year}-01-02")
    global_day = 0

    for year in range(start_year, end_year + 1):
        for day_in_year in range(trading_days_per_year):
            # Linear interpolation fraction [0, 1] over the full panel
            frac = global_day / max(total_days - 1, 1)

            mu_true   = mu_start   + frac * (mu_end   - mu_start)
            n_true    = n_start    + frac * (n_end    - n_start)
            beta_true = beta_start + frac * (beta_end - beta_start)

            date_str = (
                base_date + pd.Timedelta(days=global_day)
            ).strftime("%Y-%m-%d")

            day_seed = seed + global_day
            times = simulate_hawkes_ogata(
                mu=mu_true, n=n_true, beta=beta_true,
                T=float(_RTH_SECONDS), t0=0.0,
                seed=day_seed,
            )
            times_sec = round_to_seconds(times)

            for ts in times_sec:
                event_rows.append({"date": date_str, "timestamp_sec": float(ts)})

            summary_rows.append({
                "date":     date_str,
                "n_events": len(times_sec),
                "n_true":   n_true,
                "mu_true":  mu_true,
                "beta_true": beta_true,
            })

            global_day += 1

    # Save full event-level data
    events_df = pd.DataFrame(event_rows, columns=["date", "timestamp_sec"])
    parquet_path = _DATA_DIR / "synthetic_es_events.parquet"
    events_df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(events_df):,} events -> {parquet_path}")

    summary_df = pd.DataFrame(
        summary_rows, columns=["date", "n_events", "n_true", "mu_true", "beta_true"]
    )
    return summary_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add project root to path so src.hawkes imports work when run directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("Generating synthetic ES data...")
    df = generate_full_sample_synthetic()
    summary_csv = _DATA_DIR / "synthetic_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Generated {len(df)} trading days")
    print(df.head())

    # Also download SPY as demo (requires MASSIVE_API_KEY)
    try:
        spy = download_spy_minute("2009-01-01", "2010-08-29", "data/spy_minute.csv")
        print(f"SPY: {len(spy)} rows")
    except EnvironmentError as exc:
        print(f"Skipping SPY download: {exc}")
