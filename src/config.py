import os
from typing import Literal

# Raw model inputs (X): only info known at pickup; also what the API must receive.
FEATURES = ["pulocationid", "dolocationid", "tpep_pickup_datetime"]

TARGET = "duration"

# minutes
DURATION_RANGE = (3, 120)

TARGET_ENCODED_COLS = ["pulocationid", "dolocationid", "route"]
NUM_COLS = ["hour", "day_of_week", "is_weekend"]

ModelName = Literal["linear", "forest", "hgb"]

MODEL_PARAMS = {
    "linear": {},
    "forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 20,
        "max_samples": 0.1,
        "n_jobs": -1,
        "random_state": 25,
    },
    "hgb": {
        # tuned via RandomizedSearchCV in notebooks/eda.ipynb (test RMSE 5.04)
        "min_samples_leaf": 100,
        "max_leaf_nodes": 255,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
        # fixed
        "max_iter": 300,
        "early_stopping": True,
        "random_state": 25,
    },
}

MLFLOW_MODEL_NAME = "NYC_taxi_model"
MODEL_ALIAS = "champion"
MLFLOW_EXPERIMENT_NAME = "NYC_taxi_duration"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
