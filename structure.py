# import os 

# mk_dir='app'
# os.makedirs(mk_dir,exist_ok=True)
# mk_dir1='airflow'
# mkdir1_1='dags'
# os.makedirs(mkdir1_1,exist_ok=True)
# os.makedirs(mk_dir1,exist_ok=True)
# mk_dir2='models'
# os.makedirs(mk_dir2,exist_ok=True)
# mk_dir3='data'
# os.makedirs(mk_dir3,exist_ok=True)

# app_file=['main.py','pipeline.py','preprocess.py','train.py','predict.py','db.py']

# for file in app_file:
#    path= os.path.join(mk_dir,file)
#    with open (path,'w') as f:
#       pass








# # mlops_project/
# # │
# # ├── app/
# # │   ├── main.py
# # │   ├── pipeline.py
# # │   ├── preprocess.py
# # │   ├── train.py
# # │   ├── predict.py
# # │   ├── db.py
# # │   └── schemas.py
# # │
# # ├── airflow/
# # │   └── dags/
# # │       └── ml_pipeline.py
# # │
# # ├── models/
# # │   └── model.pkl
# # │
# # ├── data/
# # │   └── dataset.csv
# # │
# # ├── requirements.txt
# # ├── Dockerfile
# # ├── docker-compose.yml
# # └── README.md




# Great. Since you already created the ML functions, now you need to turn the project into a complete automated MLOps system.

# I’ll teach you the full workflow from beginning → production deployment in the correct industry-style order.

# ---

# # FINAL GOAL

# You want this:

# ```text id="jlwmm1"
# CSV
#  ↓
# MariaDB
#  ↓
# Airflow Automation
#  ↓
# Data Cleaning
#  ↓
# Transformation
#  ↓
# Model Training
#  ↓
# Save Model
#  ↓
# FastAPI API
#  ↓
# Docker
#  ↓
# Anyone Can Run It
# ```

# ---

# # PHASE 1 — PROJECT STRUCTURE

# First create a clean structure.

# ```bash id="aewbyz"
# mlops_project/
# │
# ├── app/
# │   ├── main.py
# │   ├── pipeline.py
# │   ├── preprocess.py
# │   ├── train.py
# │   ├── predict.py
# │   ├── db.py
# │   └── schemas.py
# │
# ├── airflow/
# │   └── dags/
# │       └── ml_pipeline.py
# │
# ├── models/
# │   └── model.pkl
# │
# ├── data/
# │   └── dataset.csv
# │
# ├── requirements.txt
# ├── Dockerfile
# ├── docker-compose.yml
# └── README.md
# ```

# ---
# curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml'

# # PHASE 2 — CREATE VIRTUAL ENVIRONMENT

# Inside Visual Studio Code terminal:

# ```bash id="q4xv6g"
# python -m venv venv
# ```

# Activate:

# ## Windows

# ```bash id="3t7vs4"
# venv\Scripts\activate
# ```

# ## Linux/Mac

# ```bash id="ojb1mr"
# source venv/bin/activate
# ```

# ---

# # PHASE 3 — INSTALL PACKAGES

# Install everything:

# ```bash id="uxzxzg"
# pip install pandas numpy scikit-learn sqlalchemy pymysql fastapi uvicorn apache-airflow joblib python-dotenv
# ```

# Save:

# ```bash id="17a4fq"
# pip freeze > requirements.txt
# ```

# ---

# # PHASE 4 — CONNECT MARIA DB

# Create `app/db.py`

# ```python id="mjlwmz"
# from sqlalchemy import create_engine

# DATABASE_URL = "mysql+pymysql://root:password@localhost/ml_db"

# engine = create_engine(DATABASE_URL)
# ```

# ---

# # PHASE 5 — LOAD CSV INTO MARIA DB

# Create `app/load_data.py`

# ```python id="gwhrm6"
# import pandas as pd
# from db import engine

# def upload_csv():

#     df = pd.read_csv("data/dataset.csv")

#     df.to_sql(
#         "raw_data",
#         con=engine,
#         if_exists="replace",
#         index=False
#     )

#     print("CSV uploaded")
# ```

# ---

# # PHASE 6 — CLEAN DATA

# Create `app/preprocess.py`

# Example:

# ```python id="6t2zlt"
# import pandas as pd
# from db import engine

# def clean_data():

#     query = "SELECT * FROM raw_data"

#     df = pd.read_sql(query, engine)

#     # cleaning
#     df.dropna(inplace=True)

#     # transformation
#     df["salary"] = df["salary"] * 1.1

#     # save transformed table
#     df.to_sql(
#         "transformed_data",
#         con=engine,
#         if_exists="replace",
#         index=False
#     )

#     print("Data transformed")
# ```

# ---

# # PHASE 7 — TRAIN MODEL

# Create `app/train.py`

# Example:

# ```python id="v7zwsm"
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# from db import engine

# def train_model():

#     query = "SELECT * FROM transformed_data"

#     df = pd.read_sql(query, engine)

#     X = df[["experience"]]
#     y = df["salary"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2
#     )

#     model = LinearRegression()

#     model.fit(X_train, y_train)

#     joblib.dump(model, "models/model.pkl")

#  ```

# Code:

# ```python id="m0cn0v"
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime

# from app.load_data import upload_csv
# from app.preprocess import clean_data
# from app.train import train_model

# with DAG(
#     dag_id="ml_pipeline",
#     start_date=datetime(2025, 1, 1),
#     schedule_interval="@daily",
#     catchup=False
# ) as dag:

#     upload_task = PythonOperator(
#         task_id="upload_csv",
#         python_callable=upload_csv
#     )

#     clean_task = PythonOperator(
#         task_id="clean_data",
#         python_callable=clean_datax
#     )

#     train_task = PythonOperator(
#         task_id="train_model",
#         python_callable=train_model
#     )

#     upload_task >> clean_task >> train_task
# ```

# ---

# # PHASE 9 — CREATE FASTAPI API

# Now serve predictions.

# Create `app/pipeline.py`

# ```python id="bzhg8q"
# import joblib

# model = joblib.load("models/model.pkl")

# def predict_salary(experience):

#     prediction = model.predict([[experience]])

#     return prediction[0]
# ```

# ---

# # PHASE 10 — FASTAPI MAIN FILE

# Create `app/main.py`

# ```python id="jw8nq0"
# from fastapi import FastAPI
# from pipeline import predict_salary

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"status": "running"}

# @app.get("/predict/{experience}")
# def predict(experience: int):

#     result = predict_salary(experience)

#     return {
#         "predicted_salary": result
#     }
# ```

# ---

# # PHASE 11 — RUN FASTAPI

# ```bash id="qm1d0f"
# uvicorn app.main:app --reload
# ```

# Open:

# ```text id="z3qecq"
# http://localhost:8000/docs
# ```

# Swagger UI appears automatically.

# ---

# # PHASE 12 — DOCKERIZE PROJECT

# Now make project portable.

# ---

# # Create Dockerfile

# ```dockerfile id="y4v8nk"
# FROM python:3.11

# WORKDIR /app

# COPY . .

# RUN pip install -r requirements.txt

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ```

# ---

# # PHASE 13 — CREATE DOCKER COMPOSE

# This runs:

# * MariaDB
# * Airflow
# * FastAPI

# Create `docker-compose.yml`

# ```yaml id="st47j8"
# version: '3'

# services:

#   mariadb:
#     image: mariadb:11
#     environment:
#       MYSQL_ROOT_PASSWORD: root
#       MYSQL_DATABASE: ml_db
#     ports:
#       - "3306:3306"

#   api:
#     build: .
#     ports:
#       - "8000:8000"

#   airflow:
#     image: apache/airflow:2.8.1
#     ports:
#       - "8080:8080"
# ```

# ---

# # PHASE 14 — RUN EVERYTHING

# ```bash id="k4r8vs"
# docker compose up
# ```

# Now:

# * FastAPI → localhost:8000
# * Airflow → localhost:8080
# * MariaDB → localhost:3306

# ---

# # PHASE 15 — AIRFLOW UI

# Open:

# ```text id="7vxy9m"
# http://localhost:8080
# ```

# Trigger DAG:

# * upload csv
# * transform data
# * train model

# automatically.

# ---

# # PHASE 16 — WHAT HAPPENS NOW

# Your system now works like:

# ```text id="h47jlwm"
# Airflow
#   ↓
# Reads CSV
#   ↓
# Stores Raw Data
#   ↓
# Cleans Data
#   ↓
# Stores Transformed Data
#   ↓
# Trains Model
#   ↓
# Saves model.pkl
#   ↓
# FastAPI Uses Latest Model
#   ↓
# Users Request Predictions
# ```

# This is a REAL MLOps pipeline.

# ---

# # PHASE 17 — SHARE WITH OTHERS

# Now anyone can run:

# ```bash id="wwy5vf"
# docker compose up
# ```

# without:

# * installing packages
# * configuring Python
# * fixing dependency errors

# This is why Docker matters.

# ---

# # PHASE 18 — NEXT PROFESSIONAL IMPROVEMENTS

# After this works:

# | Add                                                                          | Why                 |
# | ---------------------------------------------------------------------------- | ------------------- |
# | [MLflow](https://mlflow.org?utm_source=chatgpt.com)                          | experiment tracking |
# | [Redis](https://redis.io/?utm_source=chatgpt.com)                            | task queue          |
# | [Celery](https://docs.celeryq.dev/en/stable/?utm_source=chatgpt.com)         | async tasks         |
# | [Prometheus](https://prometheus.io/?utm_source=chatgpt.com)                  | monitoring          |
# | [Grafana](https://grafana.com/?utm_source=chatgpt.com)                       | dashboards          |
# | [Kubernetes](https://kubernetes.io/?utm_source=chatgpt.com)                  | scaling             |
# | [GitHub Actions](https://github.com/features/actions?utm_source=chatgpt.com) | CI/CD               |

# ---

# # IMPORTANT INDUSTRY RULE

# Keep these separated:

# | Component | Responsibility |
# | --------- | -------------- |
# | Airflow   | training       |
# | FastAPI   | prediction     |
# | MariaDB   | storage        |
# | Docker    | packaging      |

# This separation is professional architecture.

# ---

# # YOUR NEXT TASK

# Do these in order:

# 1. Create clean folder structure
# 2. Move your existing functions
# 3. Build Airflow DAG
# 4. Save model.pkl
# 5. Create FastAPI API
# 6. Dockerize
# 7. Run docker compose

# Once you finish that, your project becomes:

# * deployable
# * reusable
# * scalable
# * portfolio-level MLOps work.
