import joblib
import pandas as pd

from drift_system.config import CFG


def load_artifacts(path=None):
    path = path or (CFG.artifacts_dir / "model.joblib")
    return joblib.load(path)


def predict_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    art = load_artifacts()
    pipe = art["pipeline"]
    schema = art["train_schema"]

    X = df_batch.copy()
    for c in schema:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[schema]

    probs = pipe.predict_proba(X)[:, 1]
    out = df_batch.copy()
    out["prob_positive"] = probs
    return out
