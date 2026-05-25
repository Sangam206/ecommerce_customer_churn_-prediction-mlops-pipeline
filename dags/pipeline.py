from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime

from database.load_data import load_data
from app.cleandata import data_cleaning
# from app.
from app.preprocess import data_splitting, preprocess_data
from app.train import hyperparameter_tuning,model_training,save_model
with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    upload_task = PythonOperator(
        task_id="upload_csv",
        python_callable=load_data
    )

    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=data_cleaning
    )

    split_task=PythonOperator(
        task_id="splitting_data",
        python_callable=data_splitting
    )

    preprocess_task= PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data

    )
    train_task= PythonOperator(
        task_id="train_model",
        python_callable=save_model
    )

    

    upload_task >> clean_task >> split_task >> preprocess_task >> train_task


