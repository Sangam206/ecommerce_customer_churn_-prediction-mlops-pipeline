import os
import mlflow


def setup_mlflow(experiment_name="customer_churn_prediction"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    print("=" * 60)
    print("NEW MLFLOW CONFIG LOADED")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment Name: {experiment_name}")
    print("=" * 60)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


# def log_model(model, params=None, metrics=None, model_name="model"):
#     with mlflow.start_run():

#         if params:
#             mlflow.log_params(params)

#         if metrics:
#             mlflow.log_metrics(metrics)

#         try:
#             import xgboost
#             if isinstance(model, xgboost.XGBModel):
#                 mlflow.xgboost.log_model(
#                     model,
#                     name=model_name        # ✅ MLflow v3 API
#                 )
#             else:
#                 mlflow.sklearn.log_model(
#                     model,
#                     name=model_name
#                 )
#         except ImportError:
#             mlflow.sklearn.log_model(model, name=model_name)

#         run_id = mlflow.active_run().info.run_id
#         print(f"Model '{model_name}' logged successfully.")
#         print(f"Run ID: {run_id}")
#         return run_id