"""
TabPFN Embedding Extractor for Survival Analysis

Consolidated from:
- tfm/TabPFN/extract_embeddings.py
- survivalStacking/tools.py

Features:
- Deep embeddings from TabPFN model
- Optional concatenation with raw features (deep+raw mode)
- Optional PCA compression for tree-based models
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    warnings.warn("tabpfn/tabpfn_extensions not found. Install with: pip install tabpfn tabpfn-extensions")


def _tabpfn_emb_to_2d(emb: np.ndarray, expected_n: int) -> np.ndarray:
    """
    Convert TabPFN embeddings to (n_samples, emb_dim).

    Handles common shapes:
      - (n_samples, d)
      - (k, n_samples, d)  -> average over k
      - (n_samples, k, d)  -> average over k
      - (1, n_samples, d)  -> squeeze/average over axis 0
    """
    emb = np.asarray(emb)

    # Already 2D
    if emb.ndim == 2:
        if emb.shape[0] != expected_n:
            raise ValueError(f"2D emb has wrong n: {emb.shape}, expected first dim {expected_n}")
        return emb

    if emb.ndim == 3:
        a, b, d = emb.shape

        if b == expected_n:
            return emb.mean(axis=0)  # (n_samples, d)

        if a == expected_n:
            return emb.mean(axis=1)  # (n_samples, d)

        if a == 1:
            return _tabpfn_emb_to_2d(emb[0], expected_n)  # becomes (b, d)
        if b == 1:
            return _tabpfn_emb_to_2d(emb[:, 0, :], expected_n)  # becomes (a, d)

        raise ValueError(f"Unrecognized 3D embedding shape {emb.shape} for expected_n={expected_n}")

    raise ValueError(f"Unrecognized embedding ndim={emb.ndim}, shape={emb.shape}")


def _apply_pca_compression(X_train_emb: np.ndarray, X_val_emb: np.ndarray, 
                           X_test_emb: np.ndarray, X_dev_emb: np.ndarray = None,
                           n_components: int = 32, verbose: bool = False) -> Tuple:
    """
    Apply PCA to compress high-dimensional embeddings.
    Fits on training data only to prevent leakage.
    """
    pca = PCA(n_components=min(n_components, X_train_emb.shape[1], X_train_emb.shape[0]))

    X_train_pca = pca.fit_transform(X_train_emb)
    X_val_pca = pca.transform(X_val_emb)
    X_test_pca = pca.transform(X_test_emb)
    X_dev_pca = pca.transform(X_dev_emb) if X_dev_emb is not None else None

    if verbose:
        explained_var = pca.explained_variance_ratio_.sum() * 100
        print(f"  → PCA: {X_train_emb.shape[1]}D → {X_train_pca.shape[1]}D ({explained_var:.1f}% variance)")

    return X_train_pca, X_val_pca, X_test_pca, X_dev_pca, pca


def apply_tabpfn_embedding(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    E_train: np.ndarray,
    T_train: np.ndarray = None,
    X_dev: np.ndarray = None,
    feature_names: list = None,
    use_deep_embeddings: bool = True,
    concat_with_raw: bool = True,
    pca_for_trees: bool = False,
    pca_n_components: int = 32,
    n_estimators: int = 1,
    n_fold: int = 0,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], object]:
    """
    Apply TabPFN embedding extraction to train/val/test splits.

    Args:
        X_train, X_val, X_test: Feature arrays (numpy)
        E_train: Event indicator for training (used to create pseudo-target)
        T_train: Time to event for training (used to create pseudo-target)
        X_dev: Optional dev set
        feature_names: List of feature names (optional)
        use_deep_embeddings: Extract embeddings (True) or return raw (False)
        concat_with_raw: Concatenate embeddings with raw features (deep+raw mode)
        pca_for_trees: Apply PCA compression (for tree models)
        pca_n_components: Target PCA dimensions (default 32)
        n_estimators: Number of TabPFN estimators
        n_fold: Fold parameter for TabPFN

    Returns:
        (X_train_emb, X_val_emb, X_test_emb, X_dev_emb, embedder)
    """
    if not TABPFN_AVAILABLE:
        warnings.warn("TabPFN not available, returning raw features")
        return X_train, X_val, X_test, X_dev, None

    if not use_deep_embeddings:
        # Raw mode - no embeddings
        return X_train, X_val, X_test, X_dev, None

    if verbose:
        print(f"TabPFN: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Check for valid inputs
    if not (np.isfinite(X_train).all() and np.isfinite(X_val).all() and np.isfinite(X_test).all()):
        raise ValueError("Non-finite values found in X. Please impute/clean before TabPFN embeddings.")

    # Create pseudo-target for TabPFN (combines time bins and event)
    # This helps TabPFN learn useful representations for survival
    if T_train is not None:
        time_bins = pd.qcut(T_train, q=5, labels=False, duplicates='drop')
        y_train = (time_bins * 2 + E_train.astype(int)).astype(int)
    else:
        y_train = E_train.astype(int)

    # Initialize TabPFN
    clf = TabPFNClassifier(n_estimators=n_estimators)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    
    # Fit on training data
    embedder.fit(X_train, y_train)

    # Extract embeddings
    train_emb = embedder.get_embeddings(X_train, y_train, X_train, data_source="train")
    val_emb = embedder.get_embeddings(X_train, y_train, X_val, data_source="test")
    test_emb = embedder.get_embeddings(X_train, y_train, X_test, data_source="test")
    dev_emb = embedder.get_embeddings(X_train, y_train, X_dev, data_source="test") if X_dev is not None else None

    # Handle shape conversion
    train_emb = _tabpfn_emb_to_2d(train_emb, expected_n=X_train.shape[0])
    val_emb = _tabpfn_emb_to_2d(val_emb, expected_n=X_val.shape[0])
    test_emb = _tabpfn_emb_to_2d(test_emb, expected_n=X_test.shape[0])
    if dev_emb is not None:
        dev_emb = _tabpfn_emb_to_2d(dev_emb, expected_n=X_dev.shape[0])

    if verbose:
        print(f"  → TabPFN embeddings: {train_emb.shape[1]}D")

    # Apply PCA if requested (for tree models)
    if pca_for_trees and train_emb.shape[1] > pca_n_components:
        train_emb, val_emb, test_emb, dev_emb, _ = _apply_pca_compression(
            train_emb, val_emb, test_emb, dev_emb,
            n_components=pca_n_components, verbose=verbose
        )

    # Concatenate with raw features if requested
    if concat_with_raw:
        X_train_out = np.concatenate([X_train, train_emb], axis=1)
        X_val_out = np.concatenate([X_val, val_emb], axis=1)
        X_test_out = np.concatenate([X_test, test_emb], axis=1)
        X_dev_out = np.concatenate([X_dev, dev_emb], axis=1) if dev_emb is not None else None
        
        if verbose:
            print(f"  → Final: {X_train_out.shape[1]}D (raw + embeddings)")
    else:
        X_train_out, X_val_out, X_test_out, X_dev_out = train_emb, val_emb, test_emb, dev_emb

    return X_train_out, X_val_out, X_test_out, X_dev_out, embedder


# =============================================================================
# LEGACY API (for backward compatibility with old mixin classes)
# =============================================================================

def get_embeddings_tabpfn(X_train, X_test, t_train, e_train, data_frame_output=True):
    """
    Legacy function for backward compatibility with old TabPFN mixin classes.
    
    This wraps the new apply_tabpfn_embedding() function but provides the old
    API signature expected by CoxPH_TabPFN_embeddings, RSF_TabPFN_embeddings, etc.
    
    Args:
        X_train: Training features (array or DataFrame)
        X_test: Test features (array or DataFrame)
        t_train: Training times
        e_train: Training event indicators
        data_frame_output: If True, return DataFrames; if False, return arrays
        
    Returns:
        (train_embeddings, test_embeddings) as DataFrames or arrays
    """
    # Convert to numpy if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
    t_train_np = t_train.values.flatten() if hasattr(t_train, 'values') else np.asarray(t_train).flatten()
    e_train_np = e_train.values.flatten() if hasattr(e_train, 'values') else np.asarray(e_train).flatten()
    
    # Get feature names
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f'x{i}' for i in range(X_train_np.shape[1])]
    
    # Use the new embedding function (embeddings only, no raw concat)
    train_emb, _, test_emb, _, _ = apply_tabpfn_embedding(
        X_train=X_train_np,
        X_val=X_test_np,  # Not used for legacy API
        X_test=X_test_np,
        E_train=e_train_np,
        T_train=t_train_np,
        feature_names=feature_names,
        use_deep_embeddings=True,
        concat_with_raw=False,
        pca_for_trees=False,
        verbose=False
    )
    
    if data_frame_output:
        # Get indices
        train_index = X_train.index if hasattr(X_train, 'index') else range(len(X_train_np))
        test_index = X_test.index if hasattr(X_test, 'index') else range(len(X_test_np))
        
        train_embeddings = pd.DataFrame(
            train_emb, 
            columns=[f"x{i}" for i in range(train_emb.shape[1])],
            index=train_index
        )
        test_embeddings = pd.DataFrame(
            test_emb, 
            columns=[f"x{i}" for i in range(test_emb.shape[1])],
            index=test_index
        )
        return train_embeddings, test_embeddings
    else:
        return train_emb, test_emb
