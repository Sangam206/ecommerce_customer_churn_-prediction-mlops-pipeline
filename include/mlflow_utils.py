import mlflow
import os

def setup_mlflow(experiment_name="default_experiment"):
    """
    Setup MLflow tracking URI + experiment
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"Experiment: {experiment_name}")


def log_model(model, params=None, metrics=None, model_name="model"):
    """
    Logs model + params + metrics to MLflow
    """

    with mlflow.start_run():

        if params:
            mlflow.log_params(params)

        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, model_name)