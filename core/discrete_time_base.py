"""
Base class for Discrete Time Transformations.

Provides shared functionality for both binary survival and competing risks
transformations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseDiscreteTimeTransformer(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for discrete time transformers.
    
    Implements shared logic for interval creation and time feature generation.
    Subclasses implement specific transform logic for binary vs multi-class targets.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals to create
    strategy : str, default='quantile'
        How to create intervals: 'quantile' (balanced events) or 'uniform' (equal width)
    include_time_features : bool, default=True
        Whether to add time-related features (interval index, elapsed time, etc.)
    
    Attributes
    ----------
    cut_points_ : np.ndarray
        The boundaries of time intervals (fitted from training data)
    interval_midpoints_ : np.ndarray
        Midpoint of each interval (useful for evaluation)
    max_time_ : float
        Maximum observed time in training data
    """
    
    def __init__(
        self, 
        n_intervals: int = 20,
        strategy: str = 'quantile',
        include_time_features: bool = True
    ):
        self.n_intervals = n_intervals
        self.strategy = strategy
        self.include_time_features = include_time_features
        
        # Fitted attributes
        self.cut_points_ = None
        self.interval_midpoints_ = None
        self.max_time_ = None
        
    def fit(self, T: np.ndarray, E: np.ndarray) -> 'BaseDiscreteTimeTransformer':
        """
        Fit the transformer by computing time interval boundaries.
        
        Parameters
        ----------
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray  
            Event indicator (binary: 0/1, competing risks: 0/1/2/...)
            
        Returns
        -------
        self
        """
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        self.max_time_ = T.max()
        
        if self.strategy == 'quantile':
            # Use event times for quantile computation (more balanced)
            event_times = T[E > 0]
            if len(event_times) < self.n_intervals:
                event_times = T
            
            percentiles = np.linspace(0, 100, self.n_intervals + 1)
            self.cut_points_ = np.percentile(event_times, percentiles)
            self.cut_points_ = np.unique(self.cut_points_)
            
            if self.cut_points_[0] > 0:
                self.cut_points_ = np.concatenate([[0], self.cut_points_])
            if self.cut_points_[-1] < self.max_time_:
                self.cut_points_ = np.concatenate([self.cut_points_, [self.max_time_ + 1e-6]])
                
        elif self.strategy == 'uniform':
            self.cut_points_ = np.linspace(0, self.max_time_ + 1e-6, self.n_intervals + 1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.interval_midpoints_ = (self.cut_points_[:-1] + self.cut_points_[1:]) / 2
        
        # Hook for subclasses to add additional fitting logic
        self._post_fit(T, E)
        
        return self
    
    def _post_fit(self, T: np.ndarray, E: np.ndarray) -> None:
        """Hook for subclasses to add fitting logic. Override as needed."""
        pass
    
    def _create_time_features(self, interval_idx: int, n_intervals: int) -> np.ndarray:
        """Create time-related features for a given interval."""
        features = [
            interval_idx,  # Integer time index
            interval_idx / n_intervals,  # Normalized time (0-1)
            self.interval_midpoints_[interval_idx],  # Actual time value
        ]
        return np.array(features)
    
    def transform_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Create prediction matrix for new patients.
        
        Each patient needs predictions for ALL intervals to construct
        their survival curve or CIF.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        X_pred : np.ndarray
            Expanded matrix (n_samples * n_intervals, n_features + time_features)
        """
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_intervals = len(self.cut_points_) - 1
        
        rows = []
        for i in range(n_samples):
            for j in range(n_intervals):
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i].copy()
                rows.append(row)
        
        return np.array(rows)
    
    def get_interval_times(self) -> np.ndarray:
        """Return the midpoint times of each interval."""
        return self.interval_midpoints_.copy()
    
    def get_cut_points(self) -> np.ndarray:
        """Return the interval boundaries."""
        return self.cut_points_.copy()
    
    def time_to_interval(self, t: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert continuous time to interval index."""
        return np.searchsorted(self.cut_points_[1:], t, side='left')
    
    @abstractmethod
    def transform(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        return_indices: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform to person-period format.
        
        Must be implemented by subclasses for specific target encoding.
        """
        pass
    
    def _base_transform_loop(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        assign_target_fn
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shared transform loop logic.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        T : np.ndarray
            Times
        E : np.ndarray
            Event indicators
        assign_target_fn : callable
            Function (interval_idx, event_interval_idx, event_type) -> target_value
            
        Returns
        -------
        X_expanded, y_expanded, patient_indices
        """
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        n_samples = len(T)
        n_intervals = len(self.cut_points_) - 1
        
        expanded_rows = []
        expanded_y = []
        expanded_patient_idx = []
        
        for i in range(n_samples):
            t_i = T[i]
            e_i = E[i]
            
            # Find which interval the event/censoring falls into
            interval_idx = np.searchsorted(self.cut_points_[1:], t_i, side='left')
            interval_idx = min(interval_idx, n_intervals - 1)
            
            # Patient contributes rows for all intervals up to and including their event interval
            for j in range(interval_idx + 1):
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i].copy()
                
                expanded_rows.append(row)
                expanded_patient_idx.append(i)
                
                # Use the provided function to determine target
                target = assign_target_fn(j, interval_idx, e_i)
                expanded_y.append(target)
        
        return (
            np.array(expanded_rows),
            np.array(expanded_y),
            np.array(expanded_patient_idx)
        )
