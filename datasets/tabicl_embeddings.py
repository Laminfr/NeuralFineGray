"""
TabICL Embedding Extractor for Survival Analysis
Updated version with improved deep encoder embedding extraction.
Fixes shape mismatch by correctly handling (1, N, D) output tensors.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import warnings
import torch

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    warnings.warn("tabicl package not found. Install with: pip install tabicl")


def _extract_row_embeddings(clf, X_df: pd.DataFrame, split_name: str, device: str = 'cpu') -> Optional[np.ndarray]:
    """
    Capture deep row embeddings by hooking TabICL's row_interactor module.
    Correctly handles TabICL's batching behavior (1, N_rows, Dim).
    """
    try:
        if not hasattr(clf, 'model_'):
            if hasattr(clf, '_load_model'):
                clf._load_model()
            else:
                print("DEBUG: TabICL classifier has no model_ attribute")
                return None

        model = clf.model_
        if not hasattr(model, 'row_interactor'):
            print("DEBUG: TabICL model has no row_interactor module")
            return None

        model.eval()
        captured_batches: List[torch.Tensor] = []

        def capture_hook(module, inputs, outputs):
            # Output is typically the row embeddings H
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach().cpu()

            # Fix: TabICL row_interactor often returns (1, N_rows, Dim)
            # We need to squeeze the batch dimension if it's 1 and Dim 2 is the embedding dim.
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                # Shape (1, N, D) -> Squeeze to (N, D)
                tensor = tensor.squeeze(0)
            
            # If shape is already (N, D), keep it. 
            # If shape is (Batch, Seq, Dim) with Batch > 1, we append as is and concat later.
            
            if tensor.dim() == 2:
                captured_batches.append(tensor)
            elif tensor.dim() == 3:
                # Unexpected 3D shape with Batch > 1. 
                # This might happen if TabICL batches the data.
                # Assuming (Batch, N_rows_in_batch, Dim) -> Flatten to (Batch*N, Dim)
                # But usually row_interactor dim 1 is sequence length.
                # If we are here, we just append and hope torch.cat handles it or we process later.
                captured_batches.append(tensor)

        # Register hook
        hook_handle = model.row_interactor.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                _ = clf.predict_proba(X_df)
        except Exception as exc:
            print(f"DEBUG: predict_proba failed for split '{split_name}': {exc}")
            return None
        finally:
            hook_handle.remove()

        if not captured_batches:
            print(f"DEBUG: No row_interactor outputs captured for split '{split_name}'")
            return None

        # Concatenate all captured batches
        # Note: If predict_proba loops, we might have multiple tensors.
        embeddings = torch.cat(captured_batches, dim=0)
        
        # If we still have 3D tensor after concat (e.g. from the weird fallback), flatten it?
        # But our hook logic tries to ensure 2D. 
        if embeddings.dim() == 3 and embeddings.shape[0] == 1:
             embeddings = embeddings.squeeze(0)

        expected_rows = len(X_df)

        # Handle Context + Query case
        # If TabICL includes context rows in the forward pass, we might get N_context + N_query rows.
        # We only want the last expected_rows (the queries).
        if embeddings.shape[0] > expected_rows:
            embeddings = embeddings[-expected_rows:]
        
        if embeddings.shape[0] != expected_rows:
            print(f"DEBUG: Shape mismatch. Expected {expected_rows} rows, got {embeddings.shape[0]}. Shape: {embeddings.shape}")
            # If we missed rows, we can't use this embedding.
            return None

        print(f"DEBUG: Extracted embeddings for split '{split_name}' with shape {embeddings.shape}")
        return embeddings.numpy()

    except Exception as exc:
        print(f"DEBUG: Exception during row embedding extraction for split '{split_name}': {exc}")
        import traceback
        traceback.print_exc()
        return None


def apply_tabicl_embedding(
    X_train: np.ndarray,
    X_val: np.ndarray, 
    X_test: np.ndarray,
    E_train: np.ndarray,
    feature_names: list,
    verbose: bool = True,
    use_deep_embeddings: bool = True,
    concat_with_raw: bool = True,
    **tabicl_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Apply TabICL embedding extraction to train/val/test splits.
    Returns: (X_train_emb, X_val_emb, X_test_emb, classifier)
    """
    if not TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed. Install with: pip install tabicl")
    
    if verbose:
        print("=" * 60)
        print("Applying TabICL Embedding Extraction")
        print("=" * 60)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Binary event indicator for TabICL
    y_train_cls = E_train.astype(int)
    
    if verbose:
        print(f"\n[1/4] Initializing TabICL...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
    
    # Initialize TabICL
    clf = TabICLClassifier(
        checkpoint_version=tabicl_kwargs.get('checkpoint_version', 'tabicl-classifier-v1.1-0506.ckpt'),
        n_estimators=tabicl_kwargs.get('n_estimators', 1),
        random_state=tabicl_kwargs.get('random_state', 42),
        device=tabicl_kwargs.get('device', 'cpu'),
        verbose=tabicl_kwargs.get('verbose', False),
        n_jobs=tabicl_kwargs.get('n_jobs', 1)
    )
    
    if verbose:
        print(f"\n[2/4] Fitting TabICL on training data...")
    
    clf.fit(X_train_df, y_train_cls)
    
    if verbose:
        print(f"  ✓ TabICL fitted successfully")
        if use_deep_embeddings:
            print(f"  Mode: Deep encoder embeddings")
        else:
            print(f"  Mode: Shallow predict_proba features")
    
    device = tabicl_kwargs.get('device', 'cpu')
    
    # Try to extract deep embeddings using row_interactor hooks
    X_train_emb = X_val_emb = X_test_emb = None
    if use_deep_embeddings:
        splits = [
            ('train', X_train_df),
            ('val', X_val_df),
            ('test', X_test_df),
        ]
        deep_embeddings = {}
        for split_name, df in splits:
            emb = _extract_row_embeddings(clf, df, split_name, device)
            
            # Check for validity: must not be None, must match rows, must be high-dim (>20)
            if emb is None or emb.shape[1] < 32:
                if verbose:
                    print(f"DEBUG: Extraction failed or low dim ({emb.shape[1] if emb is not None else 'None'}) for {split_name}")
                use_deep_embeddings = False
                deep_embeddings = {}
                break
            deep_embeddings[split_name] = emb
        
        if use_deep_embeddings:
            X_train_emb = deep_embeddings['train']
            X_val_emb = deep_embeddings['val']
            X_test_emb = deep_embeddings['test']
            if verbose:
                print(f"\n✓ Deep embeddings extracted: {X_train_emb.shape[1]}D")
        else:
            if verbose:
                print(f"\n⚠ Deep embedding extraction failed")
                print(f"  Using enhanced predict_proba features instead")
    
    # Fallback to predict_proba-based features if deep extraction failed or was not requested
    if not use_deep_embeddings:
        train_proba = clf.predict_proba(X_train_df)
        val_proba = clf.predict_proba(X_val_df)
        test_proba = clf.predict_proba(X_test_df)
        
        eps = 1e-10
        def proba_to_enhanced_features(proba):
            p_event = np.clip(proba[:, 1], eps, 1 - eps)
            p_censored = np.clip(proba[:, 0], eps, 1 - eps)
            return np.column_stack([
                p_event, np.log(p_event / p_censored), p_event ** 2, 
                np.sqrt(p_event), -np.log(p_censored),
                p_event * (1 - p_event), np.abs(p_event - 0.5)
            ])
        
        X_train_emb = proba_to_enhanced_features(train_proba)
        X_val_emb = proba_to_enhanced_features(val_proba)
        X_test_emb = proba_to_enhanced_features(test_proba)
    
    # Combine with raw features if requested
    if concat_with_raw:
        if verbose:
            print(f"\n[4/4] Concatenating with original features...")
        X_train_final = np.column_stack([X_train_emb, X_train])
        X_val_final = np.column_stack([X_val_emb, X_val])
        X_test_final = np.column_stack([X_test_emb, X_test])
    else:
        if verbose:
            print(f"\n[4/4] Using embeddings only (no raw features)...")
        X_train_final = X_train_emb
        X_val_final = X_val_emb
        X_test_final = X_test_emb
        
    if verbose:
        print("=" * 60)
        print(f"TabICL Embedding Complete")
        print(f"Final Train Shape: {X_train_final.shape}")
        print("=" * 60 + "\n")
    
    return X_train_final, X_val_final, X_test_final, clf