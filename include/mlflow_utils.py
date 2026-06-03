import os
import mlflow

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def setup_mlflow(experiment_name="customer_churn_prediction"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow connected → {tracking_uri} | Experiment: {experiment_name}")