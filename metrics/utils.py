import numpy as np
from lifelines import KaplanMeierFitter


def concordance_index_from_risk_scores(e, t, risk_scores, tied_tol=1e-8):
    """
    Compute C-index directly from risk scores (for Cox-like models).
    Higher risk score should correspond to higher risk (shorter survival).
    
    Args:
        e: Event indicator (1 if event occurred, 0 if censored). Can be pandas or numpy.
        t: Time to event/censoring. Can be pandas or numpy.
        risk_scores: Risk scores from model. Higher = higher risk.
        tied_tol: Tolerance for considering risk scores as tied.
        
    Returns:
        C-index value, or np.nan if not computable.
    """
    # Handle pandas objects
    event = e.values.astype(bool) if hasattr(e, 'values') else np.asarray(e).astype(bool)
    t = t.values if hasattr(t, 'values') else np.asarray(t)
    risk_scores = risk_scores.values if hasattr(risk_scores, 'values') else np.asarray(risk_scores)
    
    n_events = event.sum()
    if n_events == 0:
        return np.nan

    concordant = 0
    permissible = 0

    for i in range(len(t)):
        if not event[i]:
            continue

        # Compare with all samples at risk at time t[i]
        at_risk = t > t[i]

        # Higher risk score means higher risk (shorter time to event)
        concordant += (risk_scores[at_risk] < risk_scores[i]).sum()
        concordant += 0.5 * (np.abs(risk_scores[at_risk] - risk_scores[i]) <= tied_tol).sum()
        permissible += at_risk.sum()

    if permissible == 0:
        return np.nan

    return concordant / permissible


def estimate_ipcw(km):
    if isinstance(km, tuple):
        kmf = KaplanMeierFitter()
        e_train, t_train = km
        kmf.fit(t_train, e_train == 0)
        if (e_train == 0).sum() == 0:
            kmf = None
    else: kmf = km
    return kmf


def convert_cpu_numpy(tensor):
    """
    Convert tensor (PyTorch or numpy) to numpy array on CPU.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        numpy array
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(tensor)
    return tensor


def estimate_survival_from_cox(risk_scores_test, t_train, e_train, time_grid):
    """
    Estimate survival probabilities using Breslow estimator.
    S(t|x) = S_0(t) ^ exp(risk_score)
    
    Args:
        risk_scores_test: Risk scores for test samples
        t_train: Training event times
        e_train: Training event indicators
        time_grid: Time points for survival estimation
        
    Returns:
        Survival probabilities array (n_samples, n_times)
    """
    # Estimate baseline survival using Kaplan-Meier on training data
    kmf = KaplanMeierFitter()
    kmf.fit(t_train, event_observed=(e_train > 0))

    # Get baseline survival at time grid points
    baseline_surv = kmf.survival_function_at_times(time_grid).values

    # Clip risk scores to prevent overflow
    risk_scores_clipped = np.clip(risk_scores_test, -10, 10)

    # Calculate survival probabilities for each sample
    # S(t|x) = S_0(t) ^ exp(risk_score)
    survival_probs = np.vstack(
        [baseline_surv ** np.exp(risk) for risk in risk_scores_clipped]
    )

    return survival_probs