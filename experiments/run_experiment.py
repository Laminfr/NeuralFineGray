#!/usr/bin/env python
"""
Unified Experiment Runner

Consolidates all experiment_fw_*.py scripts into a single parameterized entry point.

Usage:
    python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw
    python -m experiments.run_experiment --dataset SUPPORT --model xgboost --mode tabpfn --fold 0
    python -m experiments.run_experiment --dataset PBC --model nfg --mode raw

Models: coxph, deepsurv, rsf, xgboost, nfg
Modes: raw, tabpfn
"""

import argparse
import os
import sys

import numpy as np

# Project root for results path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets import datasets
from experiments.experiment import (
    CoxPH_TabPFN_embeddings,
    CoxPHExperiment,
    DeepSurv_TabPFN_embeddings_Experiment,
    DeepSurvExperiment,
    NFG_TabPFN_embeddings_Experiment,
    NFGExperiment,
    RSF_TabPFN_embeddings_Experiment,
    RSFExperiment,
    XGBoost_TabPFN_embeddings_Experiment,
    XGBoostExperiment,
)

# Model configurations: param_grid for each model type
MODEL_CONFIGS = {
    'coxph': {
        'class_raw': CoxPHExperiment,
        'class_tabpfn': CoxPH_TabPFN_embeddings,
        'param_grid': {
            'penalizer': [0.001, 0.01, 0.1, 1.0],
            'epochs': [1000],
            'learning_rate': [1e-3, 1e-4],
            'batch': [100, 250],
        },
    },
    'deepsurv': {
        'class_raw': DeepSurvExperiment,
        'class_tabpfn': DeepSurv_TabPFN_embeddings_Experiment,
        'param_grid': {
            'layers': [[50], [50, 50], [100, 100]],
            'learning_rate': [1e-3, 1e-4],
            'dropout': [0.0, 0.25, 0.5],
            'epochs': [500],
            'batch': [256],
            'patience_max': [5],
        },
    },
    'rsf': {
        'class_raw': RSFExperiment,
        'class_tabpfn': RSF_TabPFN_embeddings_Experiment,
        'param_grid': {
            'n_estimators': [200],
            'max_depth': [10],
            'min_samples_split': [20],
            'min_samples_leaf': [10],
            'random_state': [42],
            'epochs': [1000],
            'learning_rate': [1e-3, 1e-4],
            'batch': [100, 250],
        },
    },
    'xgboost': {
        'class_raw': XGBoostExperiment,
        'class_tabpfn': XGBoost_TabPFN_embeddings_Experiment,
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'random_state': [42],
            'min_child_weight': [5, 10],
            'epochs': [1000],
            'batch': [100, 250],
        },
    },
    'nfg': {
        'class_raw': NFGExperiment,
        'class_tabpfn': NFG_TabPFN_embeddings_Experiment,
        'param_grid': {
            'epochs': [1000],
            'learning_rate': [1e-3, 1e-4],
            'batch': [100, 250],
            'dropout': [0., 0.25, 0.5, 0.75],
            'layers_surv': [[i] * (j + 1) for i in [25, 50] for j in range(4)],
            'layers': [[i] * (j + 1) for i in [25, 50] for j in range(4)],
            'act': ['Tanh'],
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for survival models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw
    python -m experiments.run_experiment --dataset SUPPORT --model xgboost --mode tabpfn --fold 0
    python -m experiments.run_experiment --dataset all --model nfg --mode raw
        """
    )
    parser.add_argument('--dataset', '-d', required=True,
                        help='Dataset name (METABRIC, SUPPORT, PBC) or "all"')
    parser.add_argument('--model', '-m', required=True,
                        choices=['coxph', 'deepsurv', 'rsf', 'xgboost', 'nfg'],
                        help='Model type')
    parser.add_argument('--mode', default='raw',
                        choices=['raw', 'tabpfn'],
                        help='Feature mode: raw or tabpfn embeddings')
    parser.add_argument('--fold', '-f', type=int, default=None,
                        help='Specific fold to run (default: all folds)')
    parser.add_argument('--grid-search', '-g', type=int, default=100,
                        help='Number of random search iterations')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Custom output directory (default: experiments/results/)')
    return parser.parse_args()


def run_experiment(dataset: str, model: str, mode: str, fold: int = None,
                   grid_search: int = 100, seed: int = 0, output_dir: str = None):
    """Run a single experiment with the specified configuration."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {model} on {dataset} ({mode} features)")
    print(f"{'='*60}")
    
    # Load dataset - get raw data for TabPFN mode
    if mode == 'tabpfn':
        x, t, e, feature_names, x_raw = datasets.load_dataset(dataset, competing=False, return_raw=True)
    else:
        x, t, e, feature_names = datasets.load_dataset(dataset, competing=False)
        x_raw = None
    
    # Handle competing risks for NFG single-risk mode
    if model == 'nfg' and np.max(e) > 1:
        print("Converting competing risks to single-risk (primary event only)")
        e = (e == 1).astype(int)
    
    # Get model configuration
    config = MODEL_CONFIGS[model]
    class_key = f'class_{mode}'
    experiment_class = config[class_key]
    param_grid = config['param_grid']
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results')
    
    path = os.path.join(output_dir, f'{dataset}_{mode}_{model}')
    
    print(f"Output path: {path}")
    print(f"Grid search iterations: {grid_search}")
    print(f"Random seed: {seed}")
    if fold is not None:
        print(f"Running fold: {fold}")
    
    # Create and run experiment
    experiment = experiment_class.create(
        param_grid,
        n_iter=grid_search,
        path=path,
        random_seed=seed,
        fold=fold,
    )
    
    # Train the experiment (x_raw and feature_names only needed for TabPFN mode)
    if mode == 'tabpfn':
        experiment.train(x, t, e, x_raw=x_raw, feature_names=list(feature_names))
    else:
        experiment.train(x, t, e)
    
    print(f"\nExperiment completed. Results saved to: {path}")
    return experiment


def main():
    args = parse_args()
    
    datasets_to_run = [args.dataset]
    if args.dataset.lower() == 'all':
        datasets_to_run = ['METABRIC', 'SUPPORT', 'PBC']
    
    for dataset in datasets_to_run:
        run_experiment(
            dataset=dataset,
            model=args.model,
            mode=args.mode,
            fold=args.fold,
            grid_search=args.grid_search,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
