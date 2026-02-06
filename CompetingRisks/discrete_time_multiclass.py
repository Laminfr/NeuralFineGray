"""
Discrete Time Transformation for Competing Risks (Multi-Class)

Transforms continuous competing risks data into discrete person-period format
suitable for multi-class classification.

Refactored to use BaseDiscreteTimeTransformer for shared logic.
"""

from typing import Dict, Tuple, Union

import numpy as np

from core.discrete_time_base import BaseDiscreteTimeTransformer


class DiscreteTimeCompetingRisksTransformer(BaseDiscreteTimeTransformer):
    """
    Transforms competing risks survival data to discrete person-period format
    for multi-class classification.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals to create
    strategy : str, default='quantile'
        How to create intervals: 'quantile' (balanced events) or 'uniform' (equal width)
    include_time_features : bool, default=True
        Whether to add time-related features
    
    Attributes
    ----------
    n_classes_ : int
        Number of classes (n_risks + 1 for survival)
    n_risks_ : int
        Number of competing event types
    """
    
    def __init__(
        self, 
        n_intervals: int = 20,
        strategy: str = 'quantile',
        include_time_features: bool = True
    ):
        super().__init__(n_intervals, strategy, include_time_features)
        self.n_classes_ = None
        self.n_risks_ = None
    
    def _post_fit(self, T: np.ndarray, E: np.ndarray) -> None:
        """Compute number of competing risks after base fit."""
        self.n_risks_ = int(E.max())
        self.n_classes_ = self.n_risks_ + 1  # +1 for "survived this interval"
    
    def transform(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        return_indices: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform to person-period format with multi-class targets.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray
            Event indicator (0=censored, 1,2,...=event types)
        return_indices : bool
            If True, also return original patient indices
            
        Returns
        -------
        X_expanded : np.ndarray
            Expanded features with time features
        y_expanded : np.ndarray
            Multi-class targets:
            - 0 = survived this interval
            - 1 = Event type 1 occurred
            - 2 = Event type 2 occurred, etc.
        patient_indices : np.ndarray (optional)
            Original patient index for each row
        """
        def assign_multiclass_target(current_interval, event_interval, event_type):
            # y = event_type if event occurred in this interval, else 0
            if current_interval == event_interval and event_type > 0:
                return int(event_type)
            return 0
        
        X_expanded, y_expanded, patient_indices = self._base_transform_loop(
            X, T, E, assign_multiclass_target
        )
        
        if return_indices:
            return X_expanded, y_expanded, patient_indices
        return X_expanded, y_expanded
    
    def get_interval_for_time(self, t: float) -> int:
        """Get the interval index for a specific time point."""
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        idx = np.searchsorted(self.cut_points_[1:], t, side='left')
        return min(idx, len(self.interval_midpoints_) - 1)
    
    def compute_cif_from_probs(
        self, 
        probs: np.ndarray,
        n_samples: int
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Compute Cumulative Incidence Functions from interval probabilities.
        
        The CIF for cause k at time t is:
        CIF_k(t) = sum_{j: t_j <= t} P(event=k | interval j, survived to j) * S(t_{j-1})
        
        Parameters
        ----------
        probs : np.ndarray
            Predicted probabilities (n_samples * n_intervals, n_classes)
        n_samples : int
            Number of original patients
            
        Returns
        -------
        cif : dict
            CIF for each risk: {risk: array (n_samples, n_intervals)}
        survival : np.ndarray
            Overall survival function (n_samples, n_intervals)
        """
        n_intervals = len(self.interval_midpoints_)
        n_classes = probs.shape[1]
        n_risks = n_classes - 1
        
        # Reshape: (n_samples, n_intervals, n_classes)
        probs_reshaped = probs.reshape(n_samples, n_intervals, n_classes)
        
        survival = np.ones((n_samples, n_intervals))
        cif = {k: np.zeros((n_samples, n_intervals)) for k in range(1, n_risks + 1)}
        
        for j in range(n_intervals):
            p_survive = probs_reshaped[:, j, 0]
            
            if j == 0:
                survival[:, j] = p_survive
            else:
                survival[:, j] = survival[:, j-1] * p_survive
            
            for k in range(1, n_risks + 1):
                p_event_k = probs_reshaped[:, j, k]
                if j == 0:
                    cif[k][:, j] = p_event_k
                else:
                    cif[k][:, j] = cif[k][:, j-1] + survival[:, j-1] * p_event_k
        
        return cif, survival


if __name__ == '__main__':
    # Test transformer
    np.random.seed(42)
    
    n = 100
    n_features = 5
    X = np.random.randn(n, n_features)
    T = np.random.exponential(10, n)
    E = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    
    transformer = DiscreteTimeCompetingRisksTransformer(n_intervals=10)
    transformer.fit(T, E)
    
    print(f"Cut points: {transformer.cut_points_}")
    print(f"N classes: {transformer.n_classes_}")
    
    X_exp, y_exp, idx = transformer.transform(X, T, E, return_indices=True)
    print(f"\nOriginal: {X.shape}")
    print(f"Expanded: {X_exp.shape}")
    print(f"Y distribution: {dict(zip(*np.unique(y_exp, return_counts=True)))}")
