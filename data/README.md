# Data Sources

This directory contains raw data, generated synthetic datasets, and the
scripts that produce them.

---

## Real market data — SPY 1-minute bars

| Field | Value |
|-------|-------|
| **Ticker** | SPY (SPDR S&P 500 ETF) |
| **Frequency** | 1-minute OHLCV |
| **Source** | [MASSIVE API](https://massive.io) (Polygon.io wrapper) |
| **Script** | `download_data.py → download_spy_minute()` |
| **Output** | `spy_minute.csv` |

### Requirements

Set the `MASSIVE_API_KEY` environment variable to your Polygon/MASSIVE API
key before running the download:

```bash
export MASSIVE_API_KEY="your_key_here"
python data/download_data.py
```

### Columns

| Column | Description |
|--------|-------------|
| `timestamp` | UTC bar open time (ISO 8601) |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `close` | Closing price |
| `volume` | Share volume |
| `vwap` | Volume-weighted average price |
| `n_trades` | Number of trades in bar |

---

## Synthetic ES mid-price event data

| Field | Value |
|-------|-------|
| **Instrument** | E-mini S&P 500 futures (ES), synthetic |
| **Content** | Timestamps of simulated mid-price change events |
| **Method** | Ogata thinning algorithm for univariate exponential Hawkes |
| **Script** | `download_data.py → generate_full_sample_synthetic()` |
| **Output** | `synthetic_es_events.parquet`, `synthetic_summary.csv` |

### Simulation parameters

Parameters are linearly interpolated over the 1998–2010 panel to reproduce
the broad trend of increasing market activity documented in the companion
paper:

| Year | μ (base rate) | n (branching ratio) | β (decay rate) |
|------|---------------|---------------------|----------------|
| 1998 | 0.05 | 0.30 | 1.5 |
| 2002 | 0.15 | 0.60 | 2.0 |
| 2007 | 0.50 | 0.75 | 3.0 |
| 2010 | 1.00 | 0.80 | 4.0 |

All rates are in **events per second**; each session covers 6.75 RTH hours
(24 300 seconds).  Event times are rounded to the nearest integer second.

### Parquet schema (`synthetic_es_events.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Trading date (YYYY-MM-DD) |
| `timestamp_sec` | float64 | Seconds since session open |

### Summary CSV schema (`synthetic_summary.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Trading date |
| `n_events` | int | Number of events on that day |
| `n_true` | float | True branching ratio used |
| `mu_true` | float | True base rate used |
| `beta_true` | float | True decay rate used |

---

## Generating all data

```bash
# From the project root
cd /workspace/project
export MASSIVE_API_KEY="your_key_here"   # optional — only needed for real data
python data/download_data.py
```

The script first generates the full synthetic panel (~3 276 trading days) and
then attempts the SPY download.  The SPY step is silently skipped if
`MASSIVE_API_KEY` is not set.
