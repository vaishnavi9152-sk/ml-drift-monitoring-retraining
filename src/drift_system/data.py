import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from drift_system.config import CFG


def load_data() -> pd.DataFrame:
   
    ds = fetch_openml(name="adult", version=2, as_frame=True)  
    df = ds.frame.copy()
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    # target column is 'class' in Adult dataset: <=50K or >50K
    y = (df["class"] == ">50K").astype(int)
    return y


def split_train_test(df: pd.DataFrame):
    y = make_target(df)
    X = df.drop(columns=["class"]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
