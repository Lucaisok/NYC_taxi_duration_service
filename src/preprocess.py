import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DURATION_RANGE, FEATURES, TARGET


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return df


def filter_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df[df[TARGET].between(*DURATION_RANGE)]


def split_data(df: pd.DataFrame):
    df = df.copy()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)
    return X_train, X_test, y_train, y_test
