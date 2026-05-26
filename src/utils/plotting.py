"""
Visualization utilities for Hawkes endogeneity experiments.
All plots use matplotlib with seaborn styling.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional, List

RESULTS_DIR = Path('/workspace/project/results')
sns.set_theme(style='whitegrid', palette='tab10')

def plot_longrun_n(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot long-run trend in branching ratio n for 3 window lengths (exp_1).
    Expected: rising from ~0.3 (1998) to ~0.8 (2010).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'10': '#1f77b4', '20': '#ff7f0e', '30': '#2ca02c'}
    for wl in sorted(df['window_min'].unique()):
        sub = df[df['window_min'] == wl].sort_values('date')
        ax.plot(sub['date'], sub['n_rolling'], label=f'{wl}-min window',
                alpha=0.8, linewidth=1.5, color=colors.get(str(wl), None))
    ax.axhline(0.3, color='gray', linestyle='--', linewidth=0.8, label='n=0.3 (1998 target)')
    ax.axhline(0.7, color='black', linestyle='--', linewidth=0.8, label='n=0.7 (2007 target)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Branching ratio n')
    ax.set_title('Long-run Hawkes Branching Ratio n: ES Futures 1998-2010\n(2-month centered rolling average)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp1_longrun_n.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_bootstrap_rejection(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot bootstrap rejection rates and n bootstrap dispersion (exp_2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: rejection rates by window length
    wl_order = sorted(df['window_min'].unique())
    reject_rates = []
    for wl in wl_order:
        sub = df[df['window_min'] == wl]
        reject_rates.append((~sub['accepted']).mean() * 100)
    axes[0].bar([str(w) for w in wl_order], reject_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_xlabel('Window length (minutes)')
    axes[0].set_ylabel('Rejection rate (%)')
    axes[0].set_title('KS Test Rejection Rates by Window Length\n(paper targets: 10-min~0.12%, 20-min~1.64%, 30-min~12.64%)')
    # Right: bootstrap std of n over time
    df_agg = df.groupby(['day', 'window_min'])['n_std'].mean().reset_index()
    for wl in wl_order:
        sub = df_agg[df_agg['window_min'] == wl]
        axes[1].scatter(sub['day'], sub['n_std'], label=f'{wl}-min', alpha=0.5, s=20)
    axes[1].set_xlabel('Day index')
    axes[1].set_ylabel('Bootstrap std(n)')
    axes[1].set_title('Bootstrap Dispersion of n by Window Length')
    axes[1].legend()
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp2_bootstrap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_synthetic_validation(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot estimation errors from synthetic validation (exp_3).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    params = [('n_error', 'n'), ('mu_error', 'mu'), ('beta_error', 'beta')]
    labels = df['label'].unique()
    colors = sns.color_palette('Set2', n_colors=len(labels))
    for i, (col, name) in enumerate(params):
        for j, label in enumerate(labels):
            sub = df[df['label'] == label]
            axes[i].hist(sub[col], bins=20, alpha=0.6, label=label, color=colors[j])
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1)
        axes[i].set_xlabel(f'Error in {name}')
        axes[i].set_title(f'{name} estimation error distribution')
        if i == 0:
            axes[i].legend(fontsize=8)
    plt.suptitle('Synthetic Validation: Estimation Error after Rounding + De-tie', fontsize=12)
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp3_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_reshuffling_comparison(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot empirical vs reshuffled n over time (exp_4).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby('year').agg(
        n_emp=('n_empirical_mean', 'mean'),
        n_pois=('n_poisson_mean', 'mean'),
        n_true=('n_true', 'first'),
    ).reset_index()
    ax.plot(yearly['year'], yearly['n_emp'], 'b-o', label='Empirical Hawkes n', linewidth=2)
    ax.plot(yearly['year'], yearly['n_pois'], 'r--s', label='Poisson reshuffled n', linewidth=2)
    ax.plot(yearly['year'], yearly['n_true'], 'g:', label='True n (simulated)', linewidth=1.5)
    ax.fill_between(yearly['year'], yearly['n_pois'] - 0.05, yearly['n_pois'] + 0.05,
                    alpha=0.2, color='red', label='Reshuffled ±5% band')
    ax.set_xlabel('Year')
    ax.set_ylabel('Branching ratio n')
    ax.set_title('Empirical vs Poisson-Reshuffled Hawkes n: ES 1998-2010\n(High n is not just an activity proxy)')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp4_reshuffling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_case_study(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot intraday Hawkes n for Apr 27 (exogenous) vs May 6 (endogenous) (exp_5).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {'Apr27_exogenous': '#1f77b4', 'May6_flashcrash': '#d62728'}
    titles = {'Apr27_exogenous': 'April 27, 2010 (Exogenous shock)', 
              'May6_flashcrash': 'May 6, 2010 (Flash Crash / Endogenous)'}
    for ax, day in zip(axes, ['Apr27_exogenous', 'May6_flashcrash']):
        sub = df[df['day'] == day].sort_values('window_end_min')
        ax.plot(sub['window_end_min'], sub['n_mean'], color=colors[day], linewidth=2)
        if 'n_std' in sub.columns:
            ax.fill_between(sub['window_end_min'],
                            sub['n_mean'] - sub['n_std'],
                            sub['n_mean'] + sub['n_std'],
                            alpha=0.3, color=colors[day])
        ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8, label='n=0.95 (paper peak)')
        ax.axhline(0.80, color='gray', linestyle=':', linewidth=0.8, label='n=0.80 (normal 2010)')
        ax.set_xlabel('Minutes from open')
        ax.set_ylabel('Branching ratio n')
        ax.set_title(titles[day])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    plt.suptitle('Case Study: Exogenous vs Endogenous Extreme Events\n(Apr 27 vs May 6, 2010)', fontsize=12)
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp5_case_study.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_intraday_seasonality(summary: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot intraday U-shape in mu vs flat n (exp_6).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {10: '#1f77b4', 20: '#ff7f0e', 30: '#2ca02c'}
    for wl in sorted(summary['window_min'].unique()):
        sub = summary[summary['window_min'] == wl].sort_values('clock_min')
        axes[0].plot(sub['clock_min'], sub['mu_mean'], label=f'{wl}-min', color=colors.get(wl))
        axes[0].fill_between(sub['clock_min'], sub['mu_p10'], sub['mu_p90'],
                             alpha=0.2, color=colors.get(wl))
        axes[1].plot(sub['clock_min'], sub['n_mean'], label=f'{wl}-min', color=colors.get(wl))
        axes[1].fill_between(sub['clock_min'], sub['n_p10'], sub['n_p90'],
                             alpha=0.2, color=colors.get(wl))
    axes[0].set_xlabel('Minutes from open (09:30)')
    axes[0].set_ylabel('mu (background intensity)')
    axes[0].set_title('Intraday seasonality in mu\n(U-shape expected: high at open/close)')
    axes[0].legend()
    axes[1].set_xlabel('Minutes from open (09:30)')
    axes[1].set_ylabel('n (branching ratio)')
    axes[1].set_title('Intraday profile of n\n(Expected: relatively flat)')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    plt.suptitle('Intraday Seasonality: March 2009 (Synthetic ES)', fontsize=12)
    plt.tight_layout()
    path = save_path or str(RESULTS_DIR / 'fig_exp6_intraday.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
