import os
import pickle
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# =========================
# APP
# =========================
app = FastAPI(title="Customer Churn Prediction API")

# =========================
# MODEL LOAD
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "model.pkl"
)

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ model.pkl not found")
        return None

model = load_model()

# =========================
# INPUT SCHEMA
# =========================

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

# =========================
# PREPROCESS
# =========================

def preprocess(data: CustomerData):

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    df = pd.DataFrame([data.dict()])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    transformed = ct.fit_transform(df)

    result = pd.DataFrame(transformed, columns=ct.get_feature_names_out())

    # align with training columns (optional)
    try:
        train_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "after encoding",
            "x_train.csv"
        )

        train_cols = pd.read_csv(train_path).columns.tolist()

        for col in train_cols:
            if col not in result.columns:
                result[col] = 0

        result = result[train_cols]

    except Exception as e:
        print("Column alignment skipped:", e)

    return result

# =========================
# HTML FORM UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Churn Prediction</title>

<style>
body { font-family: Arial; background:#f4f4f4; padding:30px; }
.container { background:white; padding:25px; max-width:700px; margin:auto; border-radius:10px; }
input, select { width:100%; padding:10px; margin:8px 0; }
button { width:100%; padding:12px; background:green; color:white; border:none; }
#result { margin-top:20px; font-size:18px; }
</style>

</head>

<body>

<div class="container">

<h2>Customer Churn Prediction</h2>

<form id="f">

<input id="Age" placeholder="Age" type="number">

<select id="Gender">
<option>Male</option>
<option>Female</option>
</select>

<input id="Tenure" placeholder="Tenure" type="number">
<input id="Usage_Frequency" placeholder="Usage Frequency" type="number">
<input id="Support_Calls" placeholder="Support Calls" type="number">
<input id="Payment_Delay" placeholder="Payment Delay" type="number">

<select id="Subscription_Type">
<option>Basic</option>
<option>Standard</option>
<option>Premium</option>
</select>

<select id="Contract_Length">
<option>Monthly</option>
<option>Quarterly</option>
<option>Annual</option>
</select>

<input id="Total_Spend" placeholder="Total Spend" type="number">
<input id="Last_Interaction" placeholder="Last Interaction" type="number">

<button type="submit">Predict</button>

</form>

<div id="result"></div>

</div>

<script>

document.getElementById("f").addEventListener("submit", async (e) => {
e.preventDefault();

const data = {
Age: +document.getElementById("Age").value,
Gender: document.getElementById("Gender").value,
Tenure: +document.getElementById("Tenure").value,
Usage_Frequency: +document.getElementById("Usage_Frequency").value,
Support_Calls: +document.getElementById("Support_Calls").value,
Payment_Delay: +document.getElementById("Payment_Delay").value,
Subscription_Type: document.getElementById("Subscription_Type").value,
Contract_Length: document.getElementById("Contract_Length").value,
Total_Spend: +document.getElementById("Total_Spend").value,
Last_Interaction: +document.getElementById("Last_Interaction").value
};

const res = await fetch("/predict", {
method: "POST",
headers: {"Content-Type":"application/json"},
body: JSON.stringify(data)
});

const out = await res.json();

document.getElementById("result").innerHTML =
"Prediction: " + out.churn_label +
"<br>Probability: " + out.churn_probability;

});

</script>

</body>
</html>
"""

# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict(customer: CustomerData):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_df = preprocess(customer)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return {
            "churn_prediction": int(pred),
            "churn_probability": float(round(prob, 4)),
            "churn_label": "Churn" if pred == 1 else "No Churn"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))