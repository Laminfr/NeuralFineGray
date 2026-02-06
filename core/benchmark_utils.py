"""
Shared utilities for benchmark runners.

Provides common functionality for CV setup, results aggregation, and I/O.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def create_cv_splits(
    X: np.ndarray,
    E: np.ndarray,
    n_folds: int = 5,
    val_fraction: float = 0.15,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create stratified CV splits with validation sets.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    E : np.ndarray
        Event indicators (used for stratification)
    n_folds : int
        Number of CV folds
    val_fraction : float
        Fraction of training data to use for validation
    random_state : int
        Random seed
        
    Returns
    -------
    splits : list of dicts
        Each dict contains 'train_idx', 'val_idx', 'test_idx'
    """
    from sklearn.model_selection import StratifiedKFold
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = []
    
    for fold, (train_full_idx, test_idx) in enumerate(cv.split(X, E)):
        # Create validation split from training data
        rng = np.random.RandomState(random_state + fold)
        n_val = int(val_fraction * len(train_full_idx))
        val_local_idx = rng.choice(len(train_full_idx), size=n_val, replace=False)
        
        val_idx = train_full_idx[val_local_idx]
        train_mask = np.ones(len(train_full_idx), dtype=bool)
        train_mask[val_local_idx] = False
        train_idx = train_full_idx[train_mask]
        
        splits.append({
            'fold': fold,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })
    
    return splits


def aggregate_fold_results(
    fold_results: List[Dict[str, Any]],
    exclude_keys: List[str] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute mean and std of metrics across folds.
    
    Parameters
    ----------
    fold_results : list
        List of dicts with fold metrics
    exclude_keys : list
        Keys to exclude from aggregation
        
    Returns
    -------
    mean_metrics : dict
    std_metrics : dict
    """
    if exclude_keys is None:
        exclude_keys = ['fold', 'status', 'error', 'phase', 'n_train', 'n_val', 'n_test']
    
    mean_metrics = {}
    std_metrics = {}
    
    # Collect all numeric keys
    numeric_keys = []
    for result in fold_results:
        for key, value in result.items():
            if key not in exclude_keys and isinstance(value, (int, float)) and key not in numeric_keys:
                numeric_keys.append(key)
    
    for key in numeric_keys:
        values = [r.get(key) for r in fold_results 
                  if r.get('status', 'success') == 'success' and r.get(key) is not None]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
    
    return mean_metrics, std_metrics


def save_benchmark_results(
    results: Dict[str, Any],
    output_dir: Path,
    filename: str
) -> Path:
    """Save benchmark results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return filepath


def create_results_template(
    dataset: str,
    n_samples: int,
    n_features: int,
    n_folds: int,
    **kwargs
) -> Dict[str, Any]:
    """Create a standard results dictionary template."""
    return {
        'dataset': dataset,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_folds': n_folds,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
