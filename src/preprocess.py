from typing import cast

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
    return df[df[TARGET].between(*DURATION_RANGE)]


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # cast: sklearn's return type is too loose for type checkers
    train, test = cast(list[pd.DataFrame], train_test_split(df, random_state=25))
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]
