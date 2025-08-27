# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --------- Load artifacts ----------
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS / "my_model.pkl"     # <- trained model
SCALER_PATH = ARTIFACTS / "scaler.pkl"      # <- saved StandardScaler

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------- FastAPI app + CORS ----------
app = FastAPI(title="PPH Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Input schema ----------
class PPHInput(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float
    BMI: float
    Anaemia: int            # 0 = No, 1 = Yes
    Parity: int
    DeliveryMethod: int     # 0 = Normal, 1 = Caesarean
    HistoryPPH: int         # 0 = No, 1 = Yes

# --------- Categorization helpers ----------
def categorize_sbp(x):
    if x < 90: return "Low"
    elif 90 <= x <= 129: return "Normal"
    elif 130 <= x <= 139: return "Elevated"
    elif 140 <= x <= 159: return "High"
    else: return "Severe"


def categorize_dbp(x):
    if x < 60:
        return "Low"
    elif 60 <= x <= 79:
        return "Normal"
    elif 80 <= x <= 89:
        return "High"
    else:
        return "Severe"

#blood sugar range
def categorize_bs(x):
    if x < 4.0:
        return "Low"
    elif 4.0 <= x <= 5.4:
        return "Normal (Fasting)"
    elif 5.5 <= x <= 7.8:
        return "Normal (Post-meal)"
    elif 7.9 <= x <= 11.0:
        return "Prediabetes"
    else:
        return "Diabetes"


#body temperature category
def categorize_temp(x):
    if x < 95:
        return "Hypothermia"
    elif 95 <= x <= 99.5:
        return "Normal"
    elif 99.6 <= x <= 100.9:
        return "Low-grade fever"
    elif 101 <= x <= 103:
        return "Fever"
    else:
        return "High fever"

#Heart rate
def categorize_hr(x):
    if x < 60:
        return "Bradycardia"
    elif 60 <= x <= 100:
        return "Normal"
    elif 101 <= x <= 120:
        return "Mild Tachycardia"
    else:
        return "Severe Tachycardia"


def categorize_bmi(x):
    if x < 18.5: return "Underweight"
    elif 18.5 <= x <= 24.9: return "Normal"
    elif 25 <= x <= 29.9: return "Overweight"
    elif 30 <= x <= 34.9: return "Obesity I"
    elif 35 <= x <= 39.9: return "Obesity II"
    else: return "Obesity III"

# --------- Ordinal mappings (MATCH training) ----------
ORDINAL = {
    'SBP_Category': {
        'Low': 0, 'Normal': 1, 'Elevated': 2, 'High': 3, 'Severe': 4
    },
    'DBP_Category': {
        'Low': 0, 'Normal': 1, 'High': 2, 'Severe': 3
    },
    'BS_Category': {
        'Low': 0, 'Normal (Fasting)': 1, 'Normal (Post-meal)': 2,
        'Prediabetes': 3, 'Diabetes': 4
    },
    'Temp_Category': {
        'Hypothermia': 0, 'Normal': 1, 'Low-grade fever': 2,
        'Fever': 3, 'High fever': 4
    },
    'HR_Category': {
        'Bradycardia': 0, 'Normal': 1, 'Mild Tachycardia': 2, 'Severe Tachycardia': 3
    },
    'BMI_Category': {
        'Underweight': 0, 'Normal': 1, 'Overweight': 2,
        'Obesity I': 3, 'Obesity II': 4, 'Obesity III': 5
    }
}

# --------- Features ---------
EXPECTED_FEATURES = [
    'Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate','BMI',
    'Anaemia','History_of_Past_PPH','Parity',
    'SBP_Category','DBP_Category','BS_Category','Temp_Category','HR_Category','BMI_Category',
    'Mode_of_delivery_Normal vaginal delivery',
    'Mode_of_delivery_Through operation (Caesarean Section / CS)',
]

NUMERIC_COLS = ['Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate','BMI']

# --------- Preprocessing ----------
def preprocess(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    # Categorize
    df['SBP_Category'] = df['SystolicBP'].apply(categorize_sbp)
    df['DBP_Category'] = df['DiastolicBP'].apply(categorize_dbp)
    df['BS_Category'] = df['BS'].apply(categorize_bs)
    df['Temp_Category'] = df['BodyTemp'].apply(categorize_temp)
    df['HR_Category'] = df['HeartRate'].apply(categorize_hr)
    df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

    # History PPH name alignment
    df['History_of_Past_PPH'] = df['HistoryPPH'].astype(int)

    # One-hot encode delivery
    df['Mode_of_delivery_Normal vaginal delivery'] = (df['DeliveryMethod'] == 0).astype(int)
    df['Mode_of_delivery_Through operation (Caesarean Section / CS)'] = (df['DeliveryMethod'] == 1).astype(int)

    # Drop unused raw cols
    df = df.drop(columns=['DeliveryMethod','HistoryPPH'])

    # Map ordinals
    for col, mapping in ORDINAL.items():
        df[col] = df[col].map(mapping)

    # Fill NaNs if unseen category
    df = df.fillna(0)

    # Scale numerics
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Reorder
    df = df[EXPECTED_FEATURES]

    return df

# --------- Prediction endpoint ----------
@app.post("/predict")
def predict_pph(payload: PPHInput):
    try:
        X = preprocess(payload.dict())
        pred = model.predict(X)[0]

        # Probability if classifier
        y_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            y_prob = float(np.max(proba))

        # Adjust risk map based on training (update if binary!)
        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        risk = risk_map.get(int(pred), "Unknown")

        return {
            "riskLevel": risk,
            "probability": round(y_prob, 3) if y_prob is not None else None,
            "modelVersion": "xgb_reg_v1"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}
