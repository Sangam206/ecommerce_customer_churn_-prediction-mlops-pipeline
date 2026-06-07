import os
import mlflow
from mlflow.tracking import MlflowClient


def register_best_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.18.0.2:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get best run by f1_macro
    experiment = client.get_experiment_by_name("customer_churn_prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=1
    )

    if not runs:
        raise Exception("No runs found to register")

    best_run = runs[0]
    run_id = best_run.info.run_id
    f1 = best_run.data.metrics.get("f1_macro", 0)
    print(f"Best run: {run_id} | F1: {f1}")

    # Register model
    model_uri = f"runs:/{run_id}/model"
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