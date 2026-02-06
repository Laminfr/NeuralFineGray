"""
Discrete Time Transformation for Survival Analysis

Transforms continuous survival data into discrete person-period format
suitable for binary classification.

Refactored to use BaseDiscreteTimeTransformer for shared logic.
"""

from typing import Tuple, Union

import numpy as np

from core.discrete_time_base import BaseDiscreteTimeTransformer


class DiscreteTimeTransformer(BaseDiscreteTimeTransformer):
    """
    Transforms survival data to discrete person-period format for binary classification.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals to create
    strategy : str, default='quantile'
        How to create intervals: 'quantile' (balanced events) or 'uniform' (equal width)
    include_time_features : bool, default=True
        Whether to add time-related features (interval index, elapsed time, etc.)
    """
    
    def transform(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        return_indices: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform to person-period format with binary targets.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray
            Event indicator (1=event, 0=censored)
        return_indices : bool
            If True, also return original patient indices
            
        Returns
        -------
        X_expanded : np.ndarray
            Expanded features with time index
        y_expanded : np.ndarray
            Binary targets for each person-period (0 or 1)
        patient_indices : np.ndarray (optional)
            Original patient index for each row
        """
        def assign_binary_target(current_interval, event_interval, event_type):
            # y=1 only if event occurred in this interval
            if current_interval == event_interval and event_type > 0:
                return 1
            return 0
        
        X_expanded, y_expanded, patient_indices = self._base_transform_loop(
            X, T, E, assign_binary_target
        )
        
        if return_indices:
            return X_expanded, y_expanded, patient_indices
        return X_expanded, y_expanded


def create_person_period_dataset(
    X: np.ndarray,
    T: np.ndarray, 
    E: np.ndarray,
    n_intervals: int = 20,
    strategy: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray, DiscreteTimeTransformer]:
    """
    Convenience function to create person-period dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    T : np.ndarray
        Event times
    E : np.ndarray
        Event indicators
    n_intervals : int
        Number of time intervals
    strategy : str
        'quantile' or 'uniform'
        
    Returns
    -------
    X_expanded : np.ndarray
        Person-period features
    y_expanded : np.ndarray
        Binary targets
    transformer : DiscreteTimeTransformer
        Fitted transformer for inference
    """
    transformer = DiscreteTimeTransformer(
        n_intervals=n_intervals,
        strategy=strategy,
        include_time_features=True
    )
    transformer.fit(T, E)
    X_expanded, y_expanded = transformer.transform(X, T, E)
    
    return X_expanded, y_expanded, transformer
