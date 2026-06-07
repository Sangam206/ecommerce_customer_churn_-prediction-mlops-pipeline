import os
import pickle
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


def hyperparameter_tuning():
    import optuna
    from xgboost import XGBClassifier

    x_train = pd.read_csv('after encoding/x_train.csv')
    y_train = pd.read_csv('splitting data/y_train.csv').values.ravel()

    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
            reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=50)
    return study.best_params


def model_training():
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBClassifier
    from include.mlflow_utils import setup_mlflow

    x_train = pd.read_csv('after encoding/x_train.csv')
    y_train = pd.read_csv('splitting data/y_train.csv').values.ravel()
    best_par = hyperparameter_tuning()

    setup_mlflow("customer_churn_prediction")

    with mlflow.start_run():
        xgb = XGBClassifier(
            **best_par,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        xgb.fit(x_train, y_train)

        preds = xgb.predict(x_train)
        report = classification_report(y_train, preds, output_dict=True)
        report_str = classification_report(y_train, preds)

        # Save metrics locally
        os.makedirs("train_metrics", exist_ok=True)
        with open("train_metrics/train_model_metrics.txt", "a") as f:
            f.write(report_str + "\n\n")

        # Log to MLflow
        mlflow.log_params(best_par)
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("weighted_avg_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("weighted_avg_recall", report["weighted avg"]["recall"])
        mlflow.log_metric("weighted_avg_f1", report["weighted avg"]["f1-score"])

        # MLflow v3 API
        mlflow.xgboost.log_model(xgb, name="model")

        return xgb


def save_model():
    model = model_training()
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)