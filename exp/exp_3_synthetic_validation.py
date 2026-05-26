"""
Exp 3: Synthetic data validation of timestamp rounding and de-tie bias.
Targets: std ~1.4% for mu, ~0.6% for n, ~3.5% for beta.
"""
import sys; sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
from pathlib import Path
from src.hawkes.estimator import estimate_hawkes_window
from src.hawkes.simulator import simulate_hawkes_ogata, simulate_hawkes_tick, round_to_seconds, detie_randomize

RESULTS_DIR = Path('/workspace/project/results')

# True parameters for validation
PARAM_GRID = [
    # (mu, n, beta, label)
    (0.05, 0.30, 1.5, '1998-regime'),
    (0.20, 0.55, 2.5, '2002-regime'),
    (0.60, 0.75, 3.5, '2007-regime'),
    (1.00, 0.80, 4.0, '2010-regime'),
]

def run_mc_validation(mu_true: float, n_true: float, beta_true: float,
                      label: str, T: float = 600.0, n_mc: int = 200,
                      n_bootstrap: int = 50, seed_base: int = 0) -> pd.DataFrame:
    """Run Monte Carlo validation for one parameter set."""
    rows = []
    for mc in range(n_mc):
        seed = seed_base + mc * 100
        # 1. Simulate true Hawkes process
        try:
            times = simulate_hawkes_ogata(mu_true, n_true, beta_true, T=T, t0=0.0, seed=seed)
        except Exception:
            continue
        if len(times) < 5:
            continue
        # 2. Round to seconds
        times_rounded = round_to_seconds(times)
        # 3. 50 de-tie bootstrap realizations
        boot_ns, boot_mus, boot_betas = [], [], []
        for b in range(n_bootstrap):
            times_dt = detie_randomize(times_rounded, seed=seed + b)
            est = estimate_hawkes_window(times_dt, 0.0, T)
            if est is not None and est['converged']:
                boot_ns.append(est['n_hat'])
                boot_mus.append(est['mu_hat'])
                boot_betas.append(est['beta_hat'])
        if len(boot_ns) == 0:
            continue
        # 4. Record errors
        n_est_mean = np.mean(boot_ns)
        mu_est_mean = np.mean(boot_mus)
        beta_est_mean = np.mean(boot_betas)
        rows.append({
            'mc': mc, 'label': label,
            'mu_true': mu_true, 'n_true': n_true, 'beta_true': beta_true,
            'n_hat_mean': n_est_mean, 'mu_hat_mean': mu_est_mean, 'beta_hat_mean': beta_est_mean,
            'n_hat_std': np.std(boot_ns), 'mu_hat_std': np.std(boot_mus), 'beta_hat_std': np.std(boot_betas),
            'n_error': n_est_mean - n_true,
            'mu_error': mu_est_mean - mu_true,
            'beta_error': beta_est_mean - beta_true,
            'n_rel_error': (n_est_mean - n_true) / n_true if n_true != 0 else np.nan,
            'mu_rel_error': (mu_est_mean - mu_true) / mu_true if mu_true != 0 else np.nan,
            'beta_rel_error': (beta_est_mean - beta_true) / beta_true if beta_true != 0 else np.nan,
        })
    return pd.DataFrame(rows)

def run_experiment_3() -> pd.DataFrame:
    """Run full synthetic validation."""
    all_dfs = []
    for mu_t, n_t, beta_t, label in PARAM_GRID:
        print(f'  Validating {label}: mu={mu_t}, n={n_t}, beta={beta_t}')
        df = run_mc_validation(mu_t, n_t, beta_t, label, T=600.0, n_mc=100, n_bootstrap=20)
        if len(df) > 0:
            all_dfs.append(df)
            rel_std_n = df['n_hat_std'].mean() / n_t * 100
            rel_std_mu = df['mu_hat_std'].mean() / mu_t * 100
            rel_std_beta = df['beta_hat_std'].mean() / beta_t * 100
            print(f'    n rel-std={rel_std_n:.2f}%, mu rel-std={rel_std_mu:.2f}%, beta rel-std={rel_std_beta:.2f}%')
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(RESULTS_DIR / 'exp3_synthetic_validation.csv', index=False)
    return df_all

if __name__ == '__main__':
    print('=== Experiment 3: Synthetic Validation of Rounding Bias ===')
    df = run_experiment_3()
    summary = df.groupby('label').agg(
        mu_bias=('mu_rel_error', 'mean'),
        n_bias=('n_rel_error', 'mean'),
        beta_bias=('beta_rel_error', 'mean'),
        mu_rel_std=('mu_hat_std', lambda x: (x / df.loc[x.index, 'mu_true']).mean() * 100),
        n_rel_std=('n_hat_std', lambda x: (x / df.loc[x.index, 'n_true']).mean() * 100),
        beta_rel_std=('beta_hat_std', lambda x: (x / df.loc[x.index, 'beta_true']).mean() * 100),
    ).reset_index()
    print(summary)
    summary.to_csv(RESULTS_DIR / 'exp3_validation_summary.csv', index=False)
    print('Exp 3 done.')
