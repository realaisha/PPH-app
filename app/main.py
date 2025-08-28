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
MODEL_PATH = ARTIFACTS / "maternal_risk_xgb_pipeline.pkl"     # new model
SCALER_PATH = ARTIFACTS / "numeric_scaler.pkl"                # numeric scaler

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------- FastAPI app + CORS ----------
app = FastAPI(title="PPH Risk API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend domain
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
    Anaemia: int          # 0 = No, 1 = Yes
    Parity: int
    DeliveryMethod: int   # 0 = Normal, 1 = Cesarean, 2 = Both
    HistoryPPH: int       # 0 = No, 1 = Yes


# --------- Expected features ----------
EXPECTED_FEATURES = [
    'age','systolicbp','diastolicbp','bs','bodytemp','heartrate','bmi','parity',
    'anaemia','history_of_past_pph',
    'mode_of_delivery_Both',
    'mode_of_delivery_Normal vaginal delivery',
    'mode_of_delivery_Through operation (Caesarean Section / CS)'
]

NUMERIC_COLS = ['age','systolicbp','diastolicbp','bs','bodytemp','heartrate','bmi','parity']


# --------- Preprocessing ----------
def preprocess(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    # Rename to match training
    df = df.rename(columns={
        "Age": "age",
        "SystolicBP": "systolicbp",
        "DiastolicBP": "diastolicbp",
        "BS": "bs",
        "BodyTemp": "bodytemp",
        "HeartRate": "heartrate",
        "BMI": "bmi",
        "Anaemia": "anaemia",
        "Parity": "parity",
        "HistoryPPH": "history_of_past_pph"
    })

    # Handle delivery method (one-hot encode 3 categories)
    df['mode_of_delivery_Both'] = 1 if payload["DeliveryMethod"] == 2 else 0
    df['mode_of_delivery_Normal vaginal delivery'] = 1 if payload["DeliveryMethod"] == 0 else 0
    df['mode_of_delivery_Through operation (Caesarean Section / CS)'] = 1 if payload["DeliveryMethod"] == 1 else 0

    # Scale numeric features
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Reorder columns to expected
    df = df[EXPECTED_FEATURES]

    return df


# --------- Prediction endpoint ----------
@app.post("/predict")
def predict_pph(payload: PPHInput):
    try:
        X = preprocess(payload.dict())
        pred = model.predict(X)[0]

        # Probability if classifier supports it
        y_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            y_prob = float(np.max(proba))

        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        risk = risk_map.get(int(pred), "Unknown")

        return {
            "riskLevel": risk,
            "probability": round(y_prob, 3) if y_prob is not None else None,
            "modelVersion": "xgb_classifier_v2"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "PPH Risk API running with new model"}
