import os
import time
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Customer Churn Prediction API", version="2.0")

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

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn_model")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
model_version = None


def load_model_from_registry():
    global model, model_version
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print(f"Loading model from MLflow: {model_uri}")
        loaded = mlflow.pyfunc.load_model(model_uri)

        # Get version info for health endpoint
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_version = version_info.version

        model = loaded
        print(f"Model v{model_version} loaded successfully ✅")
        return True
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        model = None
        model_version = None
        return False


# Load on startup
load_model_from_registry()


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


def build_feature_row(data: dict) -> pd.DataFrame:

    row = {
        "cat_cols__Gender_Female": int(data["Gender"] == "Female"),
        "cat_cols__Gender_Male": int(data["Gender"] == "Male"),
        "cat_cols__Gender_Other": int(data["Gender"] == "Other"),

        "cat_cols__Country_Australia": int(data["Country"] == "Australia"),
        "cat_cols__Country_Canada": int(data["Country"] == "Canada"),
        "cat_cols__Country_France": int(data["Country"] == "France"),
        "cat_cols__Country_Germany": int(data["Country"] == "Germany"),
        "cat_cols__Country_India": int(data["Country"] == "India"),
        "cat_cols__Country_Japan": int(data["Country"] == "Japan"),
        "cat_cols__Country_UK": int(data["Country"] == "UK"),
        "cat_cols__Country_USA": int(data["Country"] == "USA"),

        "cat_cols__Signup_Quarter_Q1": int(data["Signup_Quarter"] == "Q1"),
        "cat_cols__Signup_Quarter_Q2": int(data["Signup_Quarter"] == "Q2"),
        "cat_cols__Signup_Quarter_Q3": int(data["Signup_Quarter"] == "Q3"),
        "cat_cols__Signup_Quarter_Q4": int(data["Signup_Quarter"] == "Q4"),

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

    return pd.DataFrame([row])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_version,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/reload-model")
def reload_model():
    success = load_model_from_registry()
    return {
        "status": "reloaded" if success else "failed",
        "model_loaded": model is not None,
        "model_version": model_version,
    }


@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Try /reload-model."
        )

    try:
        df = build_feature_row(customer.dict())

        # Remove accidental index column
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Get model expected feature names
        expected_features = None

        try:
            # sklearn models
            expected_features = list(model._model_impl.feature_names_in_)
        except Exception:
            pass

        try:
            # xgboost models
            if expected_features is None:
                booster = model._model_impl.get_booster()
                expected_features = booster.feature_names
        except Exception:
            pass

        if expected_features is not None:

            # Add missing columns
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            # Remove extra columns
            extra_cols = [c for c in df.columns if c not in expected_features]
            if extra_cols:
                df = df.drop(columns=extra_cols)

            # Exact ordering
            df = df[expected_features]

        print("FINAL FEATURES:")
        print(df.columns.tolist())

        result = model.predict(df)

        pred = int(result[0])

        try:
            prob = float(model.predict_proba(df)[0][1])
        except Exception:
            try:
                underlying = model._model_impl
                prob = float(underlying.predict_proba(df)[0][1])
            except Exception:
                prob = float(pred)

        return {
            "churn_prediction": pred,
            "churn_probability": round(prob, 4),
            "churn_label": "Churn" if pred == 1 else "No Churn",
            "model_version": model_version,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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