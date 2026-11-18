import argparse

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import root_mean_squared_error

from src.config import (
    DURATION_RANGE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    ModelName,
)
from src.load_data import load_data
from src.pipeline import MODELS, build_pipeline
from src.preprocess import create_target, filter_target_outliers, split_data


def promote_if_better(version: int | None, rmse: float) -> None:
    if version is None:  # model wasn't registered
        return
    client = MlflowClient()
    champion_rmse = float("inf")  # no champion yet -> always promote
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        if champion.run_id is not None:  # versions can exist without a run
            champion_rmse = client.get_run(champion.run_id).data.metrics["rmse"]
    except MlflowException:
        pass

    if rmse < champion_rmse:
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", str(version))
        print(f"Promoted v{version} (rmse {rmse:.3f} < {champion_rmse:.3f})")


def main(model_name: ModelName):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_data()
    df_with_target = create_target(df)
    filtered_df = filter_target_outliers(df_with_target)
    X_train, X_test, y_train, y_test = split_data(filtered_df)
    model_pipeline = build_pipeline(model_name)

    with mlflow.start_run(run_name=f"{model_name} run") as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(MODEL_PARAMS[model_name])
        mlflow.log_param("duration_min", DURATION_RANGE[0])
        mlflow.log_param("duration_max", DURATION_RANGE[1])
        mlflow.log_param("train_rows", len(X_train))

        model_pipeline.fit(X_train, y_train)
        model_predictions = model_pipeline.predict(X_test)
        model_rmse = root_mean_squared_error(y_test, model_predictions)
        mlflow.log_metric("rmse", float(model_rmse))
        model_info = mlflow_sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X_train.head(),
            # skops only loads listed types; safe since we create these files ourselves
            skops_trusted_types=[
                "src.features.add_features",
                "sklearn.tree._tree.Tree",  # random forest trees
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )
        promote_if_better(model_info.registered_model_version, model_rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="hgb")
    main(parser.parse_args().model)
