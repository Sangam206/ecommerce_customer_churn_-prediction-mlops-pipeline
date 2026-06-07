import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Customer Churn Prediction API")

# ── Load Model ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ model.pkl not found — run the training pipeline first")
        return None

model = load_model()

# ── Schema ────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    Usage_Frequency: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: float

# ── Preprocess ────────────────────────────────────────────────────────
def preprocess(data: CustomerData):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    df = pd.DataFrame([data.dict()])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    ct = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    transformed = ct.fit_transform(df)
    result = pd.DataFrame(transformed, columns=ct.get_feature_names_out())

    # Align with training columns
    try:
        train_path = os.path.join(os.path.dirname(__file__), "..", "after encoding", "x_train.csv")
        train_cols = pd.read_csv(train_path).columns.tolist()
        for col in train_cols:
            if col not in result.columns:
                result[col] = 0
        result = result[train_cols]
    except Exception as e:
        print(f"Column alignment skipped: {e}")

    return result

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Churn Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    try:
        input_df = preprocess(customer)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "churn_label": "Churn" if prediction == 1 else "No Churn"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))