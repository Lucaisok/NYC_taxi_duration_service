from typing import Literal

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from src.config import NUM_COLS, TARGET_ENCODED_COLS
from src.features import add_features

ModelName = Literal["linear", "forest", "hgb"]
MODELS: dict[ModelName, type[RegressorMixin]] = {
    "linear": LinearRegression,
    "forest": RandomForestRegressor,
    "hgb": HistGradientBoostingRegressor,
}

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


def build_pipeline(model_name: ModelName) -> Pipeline:
    params = MODEL_PARAMS[model_name]
    model = MODELS[model_name](**params)
    preprocess = ColumnTransformer(
        [
            ("target", TargetEncoder(), TARGET_ENCODED_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    model_pipeline = Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return model_pipeline
