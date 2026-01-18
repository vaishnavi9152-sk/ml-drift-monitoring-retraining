import numpy as np
import pandas as pd
from drift_system.config import CFG


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    eps = 1e-6
    e_perc = np.clip(e_perc, eps, 1)
    a_perc = np.clip(a_perc, eps, 1)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def detect_drift(train_df: pd.DataFrame, batch_df: pd.DataFrame) -> dict:
    
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    scores = {}

    for c in num_cols:
        scores[c] = _psi(train_df[c].to_numpy(dtype=float), batch_df[c].to_numpy(dtype=float))

    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    drifted = overall >= CFG.drift_threshold

    return {
        "drift_score": overall,
        "threshold": CFG.drift_threshold,
        "drifted": drifted,
        "feature_psi": scores,
    }
