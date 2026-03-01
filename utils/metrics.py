"""
Evaluation Metrics — vectorized C-Index and helpers.
"""

import numpy as np
from typing import Optional


def compute_c_index(
    risk_scores: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    tied_tol: float = 1e-8,
) -> float:
    """
    Compute Concordance Index (C-Index) — vectorized.

    C-Index = P(risk_i > risk_j | time_i < time_j, event_i = 1)

    Args:
        risk_scores: (N,) predicted risk (higher = worse prognosis)
        events: (N,) binary event indicators (1=death, 0=censored)
        times: (N,) survival times
        tied_tol: tolerance for tied risk scores

    Returns:
        C-Index in [0, 1].  0.5 = random, 1.0 = perfect.
    """
    n = len(risk_scores)
    if n < 2:
        return 0.5

    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0

    # only consider pairs where i had an event
    event_idx = np.where(events == 1)[0]

    for i in event_idx:
        # j must have survived longer than i (or censored after i)
        valid_j = np.where(times > times[i])[0]
        if len(valid_j) == 0:
            continue

        diff = risk_scores[i] - risk_scores[valid_j]
        concordant += np.sum(diff > tied_tol)
        discordant += np.sum(diff < -tied_tol)
        tied_risk += np.sum(np.abs(diff) <= tied_tol)

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    c_index = (concordant + 0.5 * tied_risk) / total
    return float(c_index)


def compute_c_index_lifelines(
    risk_scores: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
) -> float:
    """
    C-Index via lifelines (for cross-validation with the vectorized version).
    Falls back to manual computation if lifelines not installed.
    """
    try:
        from lifelines.utils import concordance_index
        return concordance_index(times, -risk_scores, events)
    except ImportError:
        return compute_c_index(risk_scores, events, times)
