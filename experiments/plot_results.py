#!/usr/bin/env python
"""
Plot and compare results from all baseline models.

Usage:
    python -m experiments.plot_results --dataset METABRIC
    python -m experiments.plot_results --dataset all --mode raw
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import datasets


def load_experiment_results(results_dir, dataset, mode='raw'):
    """Load all model results for a dataset."""
    import io
    import torch
    
    results = {}
    models = ['coxph', 'deepsurv', 'rsf', 'xgboost', 'nfg']
    
    # Custom unpickler that maps CUDA tensors to CPU
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            else:
                return super().find_class(module, name)
    
    for model in models:
        pickle_path = os.path.join(results_dir, f'{dataset}_{mode}_{model}.pickle')
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    exp = CPU_Unpickler(f).load()
                    results[model] = exp
                    print(f"Loaded {model}: {pickle_path}")
            except Exception as e:
                print(f"Error loading {model}: {e}")
        else:
            print(f"Missing: {pickle_path}")
    
    return results


def evaluate_experiment(exp, x, t, e, times_eval=None):
    """
    Evaluate an experiment's predictions on the validation folds.
    
    Returns C-index and IBS for each fold.
    """
    from sksurv.util import Surv
    
    fold_metrics = []
    
    # Get predictions from each fold
    for fold_idx in exp.best_model.keys():
        # Get test indices for this fold
        test_mask = exp.fold_assignment == fold_idx
        test_indices = test_mask[test_mask].index.tolist()
        
        if len(test_indices) == 0:
            continue
        
        t_test = t[test_indices]
        e_test = e[test_indices]
        
        # Get survival predictions for this fold
        model = exp.best_model[fold_idx]
        x_test = exp.scaler.transform(x[test_indices])
        
        # Use the experiment's predict method if available
        try:
            # Get survival curves at evaluation times
            times = exp.times
            
            # Different models have different prediction interfaces
            if hasattr(model, 'predict_survival'):
                surv = model.predict_survival(x_test, times.tolist(), 1)
            elif hasattr(model, 'predict_survival_function'):
                sf = model.predict_survival_function(x_test)
                surv = np.array([[fn(t) for t in times] for fn in sf])
            elif hasattr(exp, '_predict_'):
                pred_df = exp._predict_(model, x_test, 1, range(len(x_test)))
                surv = pred_df.values
            else:
                # Try generic approach
                risk = model.predict(x_test)
                # For risk scores, we can't compute IBS directly
                continue
            
            # Compute metrics
            # C-index using time-dependent concordance
            y_train = Surv.from_arrays(event=e[exp.fold_assignment != fold_idx], 
                                       time=t[exp.fold_assignment != fold_idx])
            y_test = Surv.from_arrays(event=e_test.astype(bool), time=t_test)
            
            # Risk at median time (for C-index)
            mid_time_idx = len(times) // 2
            risk_scores = 1 - surv[:, mid_time_idx]  # Convert survival to risk
            
            try:
                tau = min(t_test.max(), t[exp.fold_assignment != fold_idx].max()) * 0.95
                c_index = concordance_index_ipcw(y_train, y_test, risk_scores, tau=tau)[0]
            except Exception as e:
                c_index = np.nan
            
            # IBS
            try:
                # Create time grid for IBS
                max_time = min(t_test.max(), t[exp.fold_assignment != fold_idx].max())
                time_grid = times[(times > 0) & (times < max_time)]
                if len(time_grid) > 5:
                    # Interpolate survival to grid
                    from scipy.interpolate import interp1d
                    interp = interp1d(times, surv, axis=1, fill_value='extrapolate')
                    surv_grid = interp(time_grid)
                    ibs = integrated_brier_score(y_train, y_test, surv_grid, time_grid)
                else:
                    ibs = np.nan
            except Exception as e:
                ibs = np.nan
            
            fold_metrics.append({
                'fold': fold_idx,
                'c_index': c_index,
                'ibs': ibs
            })
            
        except Exception as e:
            print(f"    Fold {fold_idx} evaluation failed: {e}")
            continue
    
    return fold_metrics


def compute_summary_table(results, x, t, e):
    """Compute summary statistics for all models."""
    summary = []
    
    for model_name, exp in results.items():
        print(f"Evaluating {model_name}...")
        fold_metrics = evaluate_experiment(exp, x, t, e)
        
        if fold_metrics:
            c_indices = [m['c_index'] for m in fold_metrics if not np.isnan(m['c_index'])]
            ibs_values = [m['ibs'] for m in fold_metrics if not np.isnan(m['ibs'])]
            
            summary.append({
                'Model': model_name.upper(),
                'C-index': f"{np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}" if c_indices else "N/A",
                'C-index_mean': np.mean(c_indices) if c_indices else np.nan,
                'C-index_std': np.std(c_indices) if c_indices else np.nan,
                'IBS': f"{np.mean(ibs_values):.4f} ± {np.std(ibs_values):.4f}" if ibs_values else "N/A",
                'IBS_mean': np.mean(ibs_values) if ibs_values else np.nan,
                'IBS_std': np.std(ibs_values) if ibs_values else np.nan,
                'n_folds': len(fold_metrics)
            })
        else:
            summary.append({
                'Model': model_name.upper(),
                'C-index': "Error",
                'IBS': "Error",
                'n_folds': 0
            })
    
    return pd.DataFrame(summary)


def plot_comparison(summary_df, dataset, mode, output_dir=None):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter valid results
    valid_df = summary_df[summary_df['C-index'] != 'Error'].copy()
    
    if len(valid_df) == 0:
        print("No valid results to plot")
        return
    
    # C-index bar plot
    ax1 = axes[0]
    x_pos = range(len(valid_df))
    colors = sns.color_palette('husl', len(valid_df)) if HAS_SEABORN else plt.cm.tab10.colors[:len(valid_df)]
    bars = ax1.bar(x_pos, valid_df['C-index_mean'], 
                   yerr=valid_df['C-index_std'], capsize=5, 
                   color=colors)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(valid_df['Model'], rotation=45, ha='right')
    ax1.set_ylabel('C-index')
    ax1.set_title(f'{dataset} - C-index (higher is better)')
    ax1.set_ylim(0.5, 0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, valid_df['C-index_mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # IBS bar plot  
    ax2 = axes[1]
    ibs_valid = valid_df[valid_df['IBS'] != 'N/A']
    if len(ibs_valid) > 0:
        x_pos = range(len(ibs_valid))
        colors = sns.color_palette('husl', len(ibs_valid)) if HAS_SEABORN else plt.cm.tab10.colors[:len(ibs_valid)]
        bars = ax2.bar(x_pos, ibs_valid['IBS_mean'],
                       yerr=ibs_valid['IBS_std'], capsize=5,
                       color=colors)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(ibs_valid['Model'], rotation=45, ha='right')
        ax2.set_ylabel('IBS')
        ax2.set_title(f'{dataset} - IBS (lower is better)')
        ax2.set_ylim(0, 0.3)
        
        for bar, val in zip(bars, ibs_valid['IBS_mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Baseline Comparison - {dataset} ({mode} features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = os.path.join(output_dir, f'{dataset}_{mode}_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot and compare baseline results')
    parser.add_argument('--dataset', '-d', default='METABRIC',
                        help='Dataset name (METABRIC, SUPPORT, PBC) or "all"')
    parser.add_argument('--mode', '-m', default='raw',
                        choices=['raw', 'tabpfn'],
                        help='Feature mode')
    parser.add_argument('--results-dir', '-r', default=None,
                        help='Results directory (default: experiments/results/)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting, only print table')
    args = parser.parse_args()
    
    if args.results_dir is None:
        args.results_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results')
    
    datasets_to_analyze = [args.dataset]
    if args.dataset.lower() == 'all':
        datasets_to_analyze = ['METABRIC', 'SUPPORT', 'PBC']
    
    for dataset in datasets_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset} ({args.mode} features)")
        print(f"{'='*60}")
        
        # Load raw data for evaluation
        x, t, e, feature_names = datasets.load_dataset(dataset, competing=False)
        
        # Load experiment results
        results = load_experiment_results(args.results_dir, dataset, args.mode)
        
        if not results:
            print(f"No results found for {dataset}")
            continue
        
        # Compute summary
        summary_df = compute_summary_table(results, x, t, e)
        
        # Print table
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(summary_df[['Model', 'C-index', 'IBS', 'n_folds']].to_string(index=False))
        
        # Plot if requested
        if not args.no_plot:
            try:
                plot_comparison(summary_df, dataset, args.mode, args.results_dir)
            except Exception as e:
                print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
