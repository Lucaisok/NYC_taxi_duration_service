from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from src.config import MODEL_PARAMS, NUM_COLS, TARGET_ENCODED_COLS, ModelName
from src.features import add_features

MODELS: dict[ModelName, type[RegressorMixin]] = {
    "linear": LinearRegression,
    "forest": RandomForestRegressor,
    "hgb": HistGradientBoostingRegressor,
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
