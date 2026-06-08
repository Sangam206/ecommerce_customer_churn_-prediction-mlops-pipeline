import os
import pickle
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
CHURN_COUNT = Counter("churn_predictions_total", "Total churn predictions", ["result"])

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "models", "model.pkl")

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("model.pkl not found")
        return None

model = load_model()


class CustomerData(BaseModel):
    Gender: str
    Country: str
    Signup_Quarter: str
    Age: float
    Membership_Years: float
    Login_Frequency: float
    Session_Duration_Avg: float
    Pages_Per_Session: float
    Cart_Abandonment_Rate: float
    Wishlist_Items: float
    Total_Purchases: float
    Average_Order_Value: float
    Days_Since_Last_Purchase: float
    Discount_Usage_Rate: float
    Returns_Rate: float
    Email_Open_Rate: float
    Customer_Service_Calls: float
    Product_Reviews_Written: float
    Social_Media_Engagement_Score: float
    Mobile_App_Usage: float
    Payment_Method_Diversity: float
    Lifetime_Value: float
    Credit_Balance: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

from fastapi.responses import HTMLResponse
import os

@app.get("/monitoring/drift", response_class=HTMLResponse)
def drift_report():
    path = "monitoring_reports/data_drift_report.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run the pipeline first.")
    with open(path) as f:
        return f.read()

@app.get("/monitoring/performance", response_class=HTMLResponse)
def performance_report():
    path = "monitoring_reports/classification_report.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run the pipeline first.")
    with open(path) as f:
        return f.read()

@app.get("/monitoring/dashboard", response_class=HTMLResponse)
def dashboard():
    path = "monitoring_reports/monitoring_dashboard.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run the pipeline first.")
    with open(path) as f:
        return f.read()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/reload-model")
def reload_model():
    global model
    model = load_model()
    return {"status": "model reloaded", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    start = time.time()

    try:
        data = customer.dict()

        row = {
            "Unnamed: 0": 0,
            "cat_cols__Gender_Female": 1 if data["Gender"] == "Female" else 0,
            "cat_cols__Gender_Male": 1 if data["Gender"] == "Male" else 0,
            "cat_cols__Gender_Other": 1 if data["Gender"] == "Other" else 0,
            "cat_cols__Country_Australia": 1 if data["Country"] == "Australia" else 0,
            "cat_cols__Country_Canada": 1 if data["Country"] == "Canada" else 0,
            "cat_cols__Country_France": 1 if data["Country"] == "France" else 0,
            "cat_cols__Country_Germany": 1 if data["Country"] == "Germany" else 0,
            "cat_cols__Country_India": 1 if data["Country"] == "India" else 0,
            "cat_cols__Country_Japan": 1 if data["Country"] == "Japan" else 0,
            "cat_cols__Country_UK": 1 if data["Country"] == "UK" else 0,
            "cat_cols__Country_USA": 1 if data["Country"] == "USA" else 0,
            "cat_cols__Signup_Quarter_Q1": 1 if data["Signup_Quarter"] == "Q1" else 0,
            "cat_cols__Signup_Quarter_Q2": 1 if data["Signup_Quarter"] == "Q2" else 0,
            "cat_cols__Signup_Quarter_Q3": 1 if data["Signup_Quarter"] == "Q3" else 0,
            "cat_cols__Signup_Quarter_Q4": 1 if data["Signup_Quarter"] == "Q4" else 0,
            "num_cols__Age": data["Age"],
            "num_cols__Membership_Years": data["Membership_Years"],
            "num_cols__Login_Frequency": data["Login_Frequency"],
            "num_cols__Session_Duration_Avg": data["Session_Duration_Avg"],
            "num_cols__Pages_Per_Session": data["Pages_Per_Session"],
            "num_cols__Cart_Abandonment_Rate": data["Cart_Abandonment_Rate"],
            "num_cols__Wishlist_Items": data["Wishlist_Items"],
            "num_cols__Total_Purchases": data["Total_Purchases"],
            "num_cols__Average_Order_Value": data["Average_Order_Value"],
            "num_cols__Days_Since_Last_Purchase": data["Days_Since_Last_Purchase"],
            "num_cols__Discount_Usage_Rate": data["Discount_Usage_Rate"],
            "num_cols__Returns_Rate": data["Returns_Rate"],
            "num_cols__Email_Open_Rate": data["Email_Open_Rate"],
            "num_cols__Customer_Service_Calls": data["Customer_Service_Calls"],
            "num_cols__Product_Reviews_Written": data["Product_Reviews_Written"],
            "num_cols__Social_Media_Engagement_Score": data["Social_Media_Engagement_Score"],
            "num_cols__Mobile_App_Usage": data["Mobile_App_Usage"],
            "num_cols__Payment_Method_Diversity": data["Payment_Method_Diversity"],
            "num_cols__Lifetime_Value": data["Lifetime_Value"],
            "num_cols__Credit_Balance": data["Credit_Balance"],
        }

        df = pd.DataFrame([row])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start)
        CHURN_COUNT.labels(result="churn" if pred == 1 else "no_churn").inc()

        return {
            "churn_prediction": int(pred),
            "churn_probability": float(round(prob, 4)),
            "churn_label": "Churn" if pred == 1 else "No Churn"
        }

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))