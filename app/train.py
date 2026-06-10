import os
import pickle
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from database.redis_client import load_encoded_data_from_redis


def compute_scale_pos_weight(y):
    """Compute scale_pos_weight for XGBoost (neg_count / pos_count)."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos


def hyperparameter_tuning():
    import optuna
    from xgboost import XGBClassifier

    # x_train = pd.read_csv('after encoding/x_train.csv')
    # y_train = pd.read_csv('splitting data/y_train.csv').values.ravel()
    x_train, _, y_train, _ = load_encoded_data_from_redis("processed_data")
    y_train = y_train.values.ravel()


    scale_pos_weight = compute_scale_pos_weight(y_train)

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
            scale_pos_weight=scale_pos_weight,  # Fixed: class imbalance correction
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

    # Attach scale_pos_weight so it carries into training
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    return best_params


def model_training():
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    from include.mlflow_utils import setup_mlflow

    # x_train = pd.read_csv('after encoding/x_train.csv')
    # y_train = pd.read_csv('splitting data/y_train.csv').values.ravel()
    x_train, _, y_train, _ = load_encoded_data_from_redis("processed_data")
    y_train = y_train.values.ravel()
    best_par = hyperparameter_tuning()

    # Random Forest class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    rf_class_weight_dict = dict(zip(classes, class_weights))

    setup_mlflow("customer_churn_prediction")

    with mlflow.start_run():

        # ---------------- XGBoost ----------------
        xgb = XGBClassifier(
            **best_par,           #scale_pos_weight ofor balincing imbalance 
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        xgb.fit(x_train, y_train)

        preds = xgb.predict(x_train)
        report = classification_report(y_train, preds, output_dict=True)
        report_str = classification_report(y_train, preds)

        os.makedirs("train_metrics", exist_ok=True)
        with open("train_metrics/train_model_metrics.txt", "a") as f:
            f.write("XGBoost Results\n")
            f.write(report_str + "\n\n")

        mlflow.log_params(best_par)
        mlflow.log_metric("xgb_f1_macro", report["macro avg"]["f1-score"])
        mlflow.log_metric("xgb_accuracy", report["accuracy"])
        mlflow.log_metric("xgb_weighted_avg_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("xgb_weighted_avg_recall", report["weighted avg"]["recall"])
        mlflow.log_metric("xgb_weighted_avg_f1", report["weighted avg"]["f1-score"])

        mlflow.xgboost.log_model(xgb, name="xgboost_model")

        # ---------------- Random Forest ----------------
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight=rf_class_weight_dict,  # class imbalance correction
            random_state=42,
            n_jobs=-1
        )
        rf.fit(x_train, y_train)

        rf_preds = rf.predict(x_train)
        rf_report = classification_report(y_train, rf_preds, output_dict=True)
        rf_report_str = classification_report(y_train, rf_preds)

        with open("train_metrics/train_model_metrics.txt", "a") as f:
            f.write("Random Forest Results\n")
            f.write(rf_report_str + "\n\n")

        mlflow.log_metric("rf_f1_macro", rf_report["macro avg"]["f1-score"])
        mlflow.log_metric("rf_accuracy", rf_report["accuracy"])
        mlflow.log_metric("rf_weighted_avg_precision", rf_report["weighted avg"]["precision"])
        mlflow.log_metric("rf_weighted_avg_recall", rf_report["weighted avg"]["recall"])
        mlflow.log_metric("rf_weighted_avg_f1", rf_report["weighted avg"]["f1-score"])
        mlflow.log_param("rf_class_weight", rf_class_weight_dict)

        mlflow.sklearn.log_model(rf, "random_forest_model")

        return xgb, rf


def save_model():
    print("Saving models...")
    model = model_training()

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Done")