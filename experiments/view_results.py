#!/usr/bin/env python
"""
Baseline Results Visualization
==============================

Visualizes results from baseline survival models:
  - CoxPH: Cox Proportional Hazards
  - DeepSurv: Neural network survival model
  - RSF: Random Survival Forest
  - XGBoost: XGBoost survival model
  - NFG: Neural Fine-Gray

Plots:
  - C-Index comparison: 3 dataset groups × 5 baseline models
  - IBS comparison: Same layout for calibration metric

Usage:
    python -m experiments.view_results
    python -m experiments.view_results --dataset METABRIC
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import datasets as dataset_loader
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'results'

# Dataset configurations
DATASET_CONFIG = {
    'METABRIC': {
        'name': 'METABRIC',
        'description': 'Breast Cancer',
    },
    'PBC': {
        'name': 'PBC',
        'description': 'Primary Biliary Cirrhosis',
    },
    'SUPPORT': {
        'name': 'SUPPORT',
        'description': 'ICU Study',
    }
}

# Model display names and order
MODELS = [
    ('coxph', 'CoxPH'),
    ('deepsurv', 'DeepSurv'),
    ('rsf', 'RSF'),
    ('xgboost', 'XGBoost'),
    ('nfg', 'NFG'),
]

# Colors for each model
MODEL_COLORS = {
    'coxph': '#3498db',      # Blue
    'deepsurv': '#2ecc71',   # Green
    'rsf': '#9b59b6',        # Purple
    'xgboost': '#e74c3c',    # Red
    'nfg': '#f39c12',        # Orange
}


def load_predictions(results_dir, dataset, mode='raw'):
    """Load all model prediction CSVs for a dataset."""
    predictions = {}
    
    for model_key, model_name in MODELS:
        csv_path = results_dir / f'{dataset}_{mode}_{model_key}.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
                predictions[model_key] = df
            except Exception as e:
                print(f"  Error loading {model_key}: {e}")
    
    return predictions


def compute_metrics(predictions, t, e):
    """Compute C-index and IBS from saved predictions."""
    results = {}
    
    for model_key, pred_df in predictions.items():
        try:
            # Get survival prediction columns (exclude 'Use' column)
            surv_cols = [c for c in pred_df.columns if 'Use' not in str(c)]
            
            if len(surv_cols) == 0:
                continue
            
            # Extract times from column names
            times = np.array([float(c[1]) for c in surv_cols])
            
            # Get survival predictions
            surv_preds = pred_df[surv_cols].values
            
            # Use indices that match
            valid_idx = pred_df.index
            t_valid = t.iloc[valid_idx] if hasattr(t, 'iloc') else t[valid_idx]
            e_valid = e.iloc[valid_idx] if hasattr(e, 'iloc') else e[valid_idx]
            
            # Convert to numpy
            t_np = np.array(t_valid).flatten()
            e_np = np.array(e_valid).flatten()
            
            # Compute C-index at median time
            mid_idx = len(times) // 2
            risk_scores = 1 - surv_preds[:, mid_idx]
            
            try:
                c_idx = concordance_index_censored(e_np > 0, t_np, risk_scores)[0]
            except Exception:
                c_idx = np.nan
            
            # Compute IBS
            try:
                y_struct = Surv.from_arrays(event=e_np > 0, time=t_np)
                max_time = t_np.max() * 0.9
                min_time = t_np.min()
                time_mask = (times > min_time) & (times < max_time)
                
                if time_mask.sum() > 5:
                    times_ibs = times[time_mask]
                    surv_ibs = surv_preds[:, time_mask]
                    ibs = integrated_brier_score(y_struct, y_struct, surv_ibs, times_ibs)
                else:
                    ibs = np.nan
            except Exception:
                ibs = np.nan
            
            results[model_key] = {
                'c_index': c_idx,
                'ibs': ibs,
                'n_samples': len(valid_idx)
            }
            
        except Exception as ex:
            print(f"    {model_key} failed: {ex}")
            results[model_key] = {'c_index': np.nan, 'ibs': np.nan, 'n_samples': 0}
    
    return results


def plot_main_comparison(all_data: dict, output_dir: Path, metric='c_index'):
    """
    Main comparison plot: 3 dataset groups × 5 baseline models
    
    Args:
        all_data: Dict of {dataset: {model: {c_index, ibs}}}
        output_dir: Where to save the plot
        metric: 'c_index' or 'ibs'
    """
    datasets = [d for d in ['METABRIC', 'PBC', 'SUPPORT'] if d in all_data]
    if not datasets:
        print("No data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#fafafa')
    
    n_datasets = len(datasets)
    n_models = len(MODELS)
    
    # Bar positioning
    bar_width = 0.14
    group_width = bar_width * n_models + 0.25  # Extra space between dataset groups
    
    # Collect data
    model_data = {m[0]: [] for m in MODELS}
    
    for dataset in datasets:
        results = all_data[dataset]
        for model_key, model_name in MODELS:
            if model_key in results:
                model_data[model_key].append(results[model_key].get(metric, 0))
            else:
                model_data[model_key].append(0)
    
    # Plot bars
    x = np.arange(n_datasets) * group_width
    
    for i, (model_key, model_name) in enumerate(MODELS):
        offset = (i - n_models / 2 + 0.5) * bar_width
        values = model_data[model_key]
        
        bars = ax.bar(
            x + offset, values, bar_width,
            label=model_name,
            color=MODEL_COLORS[model_key],
            edgecolor='white',
            linewidth=1.5,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0 and not np.isnan(val):
                ax.annotate(
                    f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val + 0.008),
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    rotation=0
                )
    
    # Formatting
    if metric == 'c_index':
        title = 'Baseline Model Comparison: C-Index\n(5-Fold Cross-Validation, Seed 42)'
        ylabel = 'C-Index (↑ higher is better)'
        y_min, y_max = 0.5, 1.0
    else:
        title = 'Baseline Model Comparison: Integrated Brier Score\n(5-Fold Cross-Validation, Seed 42)'
        ylabel = 'IBS (↓ lower is better)'
        y_min, y_max = 0.0, 0.5
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    
    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{DATASET_CONFIG[d]['name']}\n({DATASET_CONFIG[d]['description']})" 
         for d in datasets],
        fontsize=11, fontweight='medium'
    )
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS[m[0]], label=m[1])
        for m in MODELS
    ]
    ax.legend(
        handles=legend_patches,
        title='Baseline Models',
        loc='upper right' if metric == 'ibs' else 'lower right',
        fontsize=10,
        title_fontsize=11
    )
    
    # Y-axis limits
    all_vals = [v for m in model_data.values() for v in m if v > 0 and not np.isnan(v)]
    if all_vals:
        val_min = min(all_vals)
        val_max = max(all_vals)
        padding = (val_max - val_min) * 0.15
        ax.set_ylim(max(val_min - padding, y_min), min(val_max + padding + 0.05, y_max))
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # Add reference line for C-index = 0.5 (random)
    if metric == 'c_index':
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(x[-1] + group_width/2, 0.505, 'random', fontsize=8, color='gray', ha='right')
    
    # Add summary box showing best model per dataset
    summary_lines = []
    for dataset in datasets:
        results = all_data[dataset]
        best_model = None
        best_val = -np.inf if metric == 'c_index' else np.inf
        
        for model_key, _ in MODELS:
            if model_key in results:
                val = results[model_key].get(metric, np.nan)
                if not np.isnan(val):
                    if metric == 'c_index' and val > best_val:
                        best_val = val
                        best_model = model_key
                    elif metric == 'ibs' and val < best_val:
                        best_val = val
                        best_model = model_key
        
        if best_model:
            model_name = dict(MODELS)[best_model]
            summary_lines.append(f"{DATASET_CONFIG[dataset]['name']}: {model_name} ({best_val:.3f})")
    
    if summary_lines:
        summary_text = f"Best per dataset:\n" + "\n".join(summary_lines)
        ax.text(
            0.02, 0.98 if metric == 'ibs' else 0.02,
            summary_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top' if metric == 'ibs' else 'bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    filename = f'baseline_{metric}_comparison.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def print_summary_table(all_data: dict):
    """Print comprehensive results table."""
    print(f"\n{'='*80}")
    print("BASELINE MODEL COMPARISON")
    print(f"{'='*80}")
    
    # C-Index table
    print("\nC-Index (↑ higher is better):")
    print(f"{'Model':<12}", end='')
    for dataset in ['METABRIC', 'PBC', 'SUPPORT']:
        if dataset in all_data:
            print(f"{dataset:<12}", end='')
    print()
    print("-" * 48)
    
    for model_key, model_name in MODELS:
        print(f"{model_name:<12}", end='')
        for dataset in ['METABRIC', 'PBC', 'SUPPORT']:
            if dataset in all_data and model_key in all_data[dataset]:
                val = all_data[dataset][model_key].get('c_index', np.nan)
                if not np.isnan(val):
                    print(f"{val:<12.4f}", end='')
                else:
                    print(f"{'N/A':<12}", end='')
            else:
                print(f"{'-':<12}", end='')
        print()
    
    # IBS table
    print(f"\nIBS (↓ lower is better):")
    print(f"{'Model':<12}", end='')
    for dataset in ['METABRIC', 'PBC', 'SUPPORT']:
        if dataset in all_data:
            print(f"{dataset:<12}", end='')
    print()
    print("-" * 48)
    
    for model_key, model_name in MODELS:
        print(f"{model_name:<12}", end='')
        for dataset in ['METABRIC', 'PBC', 'SUPPORT']:
            if dataset in all_data and model_key in all_data[dataset]:
                val = all_data[dataset][model_key].get('ibs', np.nan)
                if not np.isnan(val):
                    print(f"{val:<12.4f}", end='')
                else:
                    print(f"{'N/A':<12}", end='')
            else:
                print(f"{'-':<12}", end='')
        print()
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Visualize baseline experiment results')
    parser.add_argument('--dataset', '-d', default='all',
                        help='Dataset name (METABRIC, SUPPORT, PBC) or "all"')
    parser.add_argument('--mode', '-m', default='raw',
                        help='Feature mode (default: raw)')
    parser.add_argument('--results-dir', '-r', default=None,
                        help='Results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    
    datasets_to_analyze = [args.dataset.upper()]
    if args.dataset.lower() == 'all':
        datasets_to_analyze = ['METABRIC', 'SUPPORT', 'PBC']
    
    print("=" * 70)
    print("Baseline Results Visualization")
    print("=" * 70)
    
    all_data = {}
    
    for dataset in datasets_to_analyze:
        print(f"\nLoading {dataset}...", end=' ')
        
        # Load ground truth
        try:
            x, t, e, feature_names = dataset_loader.load_dataset(dataset, competing=False)
        except Exception as ex:
            print(f"Error: {ex}")
            continue
        
        # Load predictions
        predictions = load_predictions(results_dir, dataset, args.mode)
        
        if not predictions:
            print(f"No results found")
            continue
        
        # Compute metrics
        results = compute_metrics(predictions, t, e)
        all_data[dataset] = results
        print(f"OK ({len(results)} models)")
    
    if not all_data:
        print("\nNo results found! Run experiments first:")
        print("  sbatch experiments/run_experiment.sbatch all coxph raw 20 42")
        return
    
    # Print summary table
    print_summary_table(all_data)
    
    # Generate plots
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    plot_main_comparison(all_data, plots_dir, metric='c_index')
    plot_main_comparison(all_data, plots_dir, metric='ibs')
    
    print(f"\nPlots saved to: {plots_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
