from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests


# ── Wrapper functions — imports happen INSIDE task, not at DAG load time ──

def reload_fastapi_model():
    try:
        response = requests.get("http://fastapi:8000/reload-model")
        print(response.json())
    except Exception as e:
        print(f"Reload failed: {e}")


def run_load_data():
    from database.load_data import load_data
    load_data()


def run_data_cleaning():
    from app.cleandata import data_cleaning
    data_cleaning()


def run_data_splitting():
    from app.preprocess import data_splitting
    data_splitting()


def run_preprocess_data():
    from app.preprocess import preprocess_data
    preprocess_data()


def run_save_model():
    from app.train import save_model
    save_model()


def run_register_best_model():
    from app.register import register_best_model
    register_best_model()


# ── DAG Definition ────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",  # retrain every week automatically
    catchup=False,
    default_args={
        "retries": 1,
    }
) as dag:

    upload_task = PythonOperator(
        task_id="upload_csv",
        python_callable=run_load_data
    )

    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=run_data_cleaning
    )

    split_task = PythonOperator(
        task_id="splitting_data",
        python_callable=run_data_splitting
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocess_data
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_save_model
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=run_register_best_model
    )

    reload_task = PythonOperator(
        task_id="reload_model",
        python_callable=reload_fastapi_model
    )

    upload_task >> clean_task >> split_task >> preprocess_task >> train_task >> register_task >> reload_task