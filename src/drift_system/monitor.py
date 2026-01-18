import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib


DEFAULT_N_BINS = 10


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_bins(n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _hist_proportions(values: np.ndarray, bins: np.ndarray) -> List[float]:
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return [0.0] * (len(bins) - 1)
    return (counts / total).astype(float).tolist()


def build_feature_baseline(
    X: pd.DataFrame,
    n_bins: int = DEFAULT_N_BINS,
) -> Dict[str, Any]:
   
    baseline = {}
    for col in X.columns:
        col_vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        col_vals = col_vals[~np.isnan(col_vals)]

        if col_vals.size == 0:
            bins = np.array([0.0, 1.0], dtype=float)
            props = [1.0]
            mu, sd = 0.0, 0.0
        else:
            mu = float(np.mean(col_vals))
            sd = float(np.std(col_vals))

            qs = np.linspace(0, 1, n_bins + 1)
            bins = np.quantile(col_vals, qs)

            bins = np.unique(bins)

            if bins.size < 2:
                bins = np.array([bins[0] - 1.0, bins[0] + 1.0], dtype=float)

            props = _hist_proportions(col_vals, bins)

        baseline[col] = {
            "bins": bins.astype(float).tolist(),
            "proportions": props,
            "mean": mu,
            "std": sd,
        }
    return baseline


def build_prediction_baseline(
    probs: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> Dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    bins = _make_bins(n_bins)
    props = _hist_proportions(probs, bins)
    return {
        "bins": bins.astype(float).tolist(),
        "proportions": props,
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
    }


def save_baseline_snapshot(
    X_reference: pd.DataFrame,
    probs_reference: np.ndarray,
    save_path: str = "artifacts/baseline.joblib",
    n_bins: int = DEFAULT_N_BINS,
) -> str:
    
    _safe_mkdir(os.path.dirname(save_path) or ".")

    snapshot = {
        "created_at": int(time.time()),
        "n_rows": int(len(X_reference)),
        "n_features": int(X_reference.shape[1]),
        "feature_baseline": build_feature_baseline(X_reference, n_bins=n_bins),
        "prediction_baseline": build_prediction_baseline(probs_reference, n_bins=n_bins),
    }

    joblib.dump(snapshot, save_path)
    return save_path


def load_baseline_snapshot(path: str = "artifacts/baseline.joblib") -> Dict[str, Any]:
    return joblib.load(path)
