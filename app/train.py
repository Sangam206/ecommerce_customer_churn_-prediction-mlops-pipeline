import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pickle
import pandas as pd


def hyperparameter_tuning():
    import optuna
    from xgboost import XGBClassifier

    x_train=pd.read_csv('after encoding/x_train.csv')
    y_train=pd.read_csv('splitting data/y_train.csv')
    try:
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

            scores = cross_val_score(
                model,
                x_train,
                y_train,
                cv=5,
                scoring="f1_macro"
            )

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50)

        best_par = study.best_params
        # logger.debug("Hyperparameter tuning for XGBoost is done")

        return best_par

    except Exception as e:
        # logger.error(f"error occurred: {e}")
        raise


def model_training():
    from xgboost import XGBClassifier

    x_train=pd.read_csv('after encoding/x_train.csv')
    y_train=pd.read_csv('splitting data/y_train.csv')
    best_par=hyperparameter_tuning()

    try:
        xgb = XGBClassifier(
            n_estimators=best_par["n_estimators"],
            max_depth=best_par["max_depth"],
            learning_rate=best_par["learning_rate"],
            subsample=best_par["subsample"],
            colsample_bytree=best_par["colsample_bytree"],
            gamma=best_par["gamma"],
            min_child_weight=best_par["min_child_weight"],
            reg_alpha=best_par["reg_alpha"],
            reg_lambda=best_par["reg_lambda"],
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

        xgb.fit(x_train, y_train)

        
        mk_dir = "train_metrics"
        os.makedirs(mk_dir, exist_ok=True)

        report = classification_report(y_train, xgb.predict(x_train))

        file_path = os.path.join(mk_dir, "train_model_metrics.txt")

        with open(file_path, "a") as file:
            file.write(report)
            file.write("\n\n")

        return xgb

    except Exception as e:
        # logger.error(f"training error: {e}")
        raise
# def model_training():
#     from xgboost import XGBClassifier

#     import mlflow
#     import mlflow.xgboost
#     from include.mlflow_utils import setup_mlflow

#     x_train = pd.read_csv('after encoding/x_train.csv')
#     y_train = pd.read_csv('splitting data/y_train.csv')

#     best_par = hyperparameter_tuning()

#     setup_mlflow("customer_bev_prediction")

#     try:
#         with mlflow.start_run():

#             xgb = XGBClassifier(
#                 n_estimators=best_par["n_estimators"],
#                 max_depth=best_par["max_depth"],
#                 learning_rate=best_par["learning_rate"],
#                 subsample=best_par["subsample"],
#                 colsample_bytree=best_par["colsample_bytree"],
#                 gamma=best_par["gamma"],
#                 min_child_weight=best_par["min_child_weight"],
#                 reg_alpha=best_par["reg_alpha"],
#                 reg_lambda=best_par["reg_lambda"],
#                 random_state=42,
#                 n_jobs=-1,
#                 eval_metric="logloss"
#             )

#             xgb.fit(x_train, y_train)

#             preds = xgb.predict(x_train)

#             report = classification_report(y_train, preds, output_dict=True)

#             # 🔥 LOG TO MLFLOW
#             mlflow.log_params(best_par)
#             mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

#             mlflow.xgboost.log_model(xgb, "model")

#             return xgb

#     except Exception as e:
#         raise

def save_model():
    model=model_training()
    folder_path='models'
    try:
        os.makedirs(folder_path,exist_ok=True)

        file_path=os.path.join(folder_path,'model.pkl')
        
        with open (file_path,'wb') as file:
            pickle.dump(model,file)

    except Exception as e:
        # logger.error(f"error occured{e}")
        raise

# save_model()


