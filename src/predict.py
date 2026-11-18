from functools import lru_cache

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from src.config import FEATURES, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, MODEL_ALIAS


def get_champion_version() -> str:
    """Ask the registry which version the alias currently points to (cheap call)."""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MODEL_ALIAS).version


@lru_cache(maxsize=1)  # keep only the current model in memory
def _load_version(version: str) -> Pipeline:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow_sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/{version}")
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(model).__name__}")
    return model


def load_model() -> tuple[Pipeline, str]:
    """Load the current champion. Downloads only when the alias has moved."""
    version = get_champion_version()
    return _load_version(version), version


def predict(model: Pipeline, trip: dict) -> float:
    X = pd.DataFrame([trip])[FEATURES]
    X = X.astype({"pulocationid": int, "dolocationid": int})
    X["tpep_pickup_datetime"] = pd.to_datetime(X["tpep_pickup_datetime"])
    prediction = model.predict(X)[0]
    return float(prediction)
