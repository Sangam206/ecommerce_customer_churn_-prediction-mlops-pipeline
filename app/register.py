import os
import mlflow
from mlflow.tracking import MlflowClient


def register_best_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("customer_churn_prediction")
    if not experiment:
        raise Exception("Experiment 'customer_churn_prediction' not found in MLflow")

    # Find best XGBoost run (f1_macro > 0)
    xgb_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.xgb_f1_macro > 0",
        order_by=["metrics.xgb_f1_macro DESC"],
        max_results=1
    )

    # Find best RandomForest run (f1_macro > 0)
    rf_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.rf_f1_macro > 0",
        order_by=["metrics.rf_f1_macro DESC"],
        max_results=1
    )

    if not xgb_runs and not rf_runs:
        raise Exception("No valid runs found with f1_macro > 0")

    # Pick overall best model across both
    best_run = None
    best_f1 = -1
    artifact_path = None

    if xgb_runs:
        xgb_f1 = xgb_runs[0].data.metrics.get("xgb_f1_macro", 0)
        if xgb_f1 > best_f1:
            best_f1 = xgb_f1
            best_run = xgb_runs[0]
            artifact_path = "xgboost_model"

    if rf_runs:
        rf_f1 = rf_runs[0].data.metrics.get("rf_f1_macro", 0)
        if rf_f1 > best_f1:
            best_f1 = rf_f1
            best_run = rf_runs[0]
            artifact_path = "random_forest_model"

    run_id = best_run.info.run_id
    print(f"Best run: {run_id} | artifact: {artifact_path} | F1: {best_f1:.4f}")

    # Register model
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name="churn_model")
    version = result.version

    # Set as champion
    client.set_registered_model_alias(
        name="churn_model",
        alias="champion",
        version=version
    )

    print(f"Model v{version} registered as champion ✅")
    return run_id