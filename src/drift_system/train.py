import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from drift_system.config import CFG
from drift_system.data import load_data, split_train_test
from drift_system.monitor import save_baseline_snapshot


def build_pipeline(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipe, num_cols, cat_cols


def main():
    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_train_test(df)

    pipe, num_cols, cat_cols = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Evaluate on test
    probs_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)

    # ---- Baseline snapshot for drift monitoring (built from TRAIN) ----
    probs_train = pipe.predict_proba(X_train)[:, 1]
    baseline_path = save_baseline_snapshot(
        X_reference=X_train,
        probs_reference=probs_train,
        save_path=str(CFG.artifacts_dir / "baseline.joblib"),
        n_bins=10,
    )

    artifacts = {
        "pipeline": pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "train_schema": list(X_train.columns),
        "metrics": {"roc_auc_test": float(auc)},
        "version": "v1",
        "baseline_path": baseline_path,
    }

    out = CFG.artifacts_dir / "model.joblib"
    joblib.dump(artifacts, out)

    print("Saved:", out)
    print("Saved baseline snapshot:", baseline_path)
    print("ROC AUC (test):", auc)


if __name__ == "__main__":
    main()
