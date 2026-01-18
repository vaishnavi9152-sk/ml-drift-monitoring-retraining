import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from drift_system.config import CFG
from drift_system.data import load_data, split_train_test


def build_pipeline(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ]
)


    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    return pipe, num_cols, cat_cols


def main():
    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_train_test(df)

    pipe, num_cols, cat_cols = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    artifacts = {
        "pipeline": pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "train_schema": list(X_train.columns),
        "metrics": {"roc_auc_test": float(auc)},
        "version": "v1",
    }

    out = CFG.artifacts_dir / "model.joblib"
    joblib.dump(artifacts, out)

    print("Saved:", out)
    print("ROC AUC (test):", auc)


if __name__ == "__main__":
    main()
