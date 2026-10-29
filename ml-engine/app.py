"""
COSMOS ML Engine - FastAPI Microservice
========================================
Exposes the deployed Random Forest Stroke Classifier over REST API.
Endpoints:
  - GET  /health            : Health check & model metadata
  - POST /predict           : Evaluates raw or pre-extracted BSI/ADR features
  - POST /simulate-packet   : Generates an on-demand virtual EEG packet with model inference
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Import the existing feature extractor and virtual hardware
from train_model import extract_spectral_features
from virtual_hardware import CosmosVirtualHardware

# Initialize FastAPI app
app = FastAPI(
    title="COSMOS Stroke Screening & EEG Asymmetry Microservice",
    description="Machine Learning inference engine for real-time stroke prediction using BSI and ADR.",
    version="1.0.0"
)

# Enable CORS for external frontends / microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained Random Forest model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cosmos_stroke_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")

model = joblib.load(MODEL_PATH)
virtual_hw = CosmosVirtualHardware(sampling_rate=256)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class FeatureInferenceRequest(BaseModel):
    bsi: float = Field(..., description="Brain Symmetry Index (BSI), normal < 0.10, pathological > 0.20")
    adr: float = Field(..., description="Alpha-to-Delta Ratio (ADR), normal > 1.20, critical < 0.70")
    alpha_power: Optional[float] = 100.0
    delta_power: Optional[float] = 50.0

class RawSignalInferenceRequest(BaseModel):
    f3_signal: List[float] = Field(..., description="Array of microvolt values from F3 (Left Frontal)")
    f4_signal: List[float] = Field(..., description="Array of microvolt values from F4 (Right Frontal)")
    sampling_rate: Optional[int] = 256

class PredictionResponse(BaseModel):
    status: str
    prediction_code: int
    prediction_label: str
    is_stroke_alert: bool
    confidence_percentage: float
    bsi: float
    adr: float
    clinical_recommendation: str


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "COSMOS ML Inference Microservice",
        "model_type": "RandomForestClassifier (n_estimators=100)",
        "features": ["BSI", "ADR", "Alpha_Power", "Delta_Power"],
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_from_features(req: FeatureInferenceRequest):
    """
    Evaluates pre-extracted BSI & ADR features against the trained Random Forest model.
    """
    feature_vec = np.array([[req.bsi, req.adr, req.alpha_power, req.delta_power]])
    pred_code = int(model.predict(feature_vec)[0])
    probs = model.predict_proba(feature_vec)[0]
    confidence = float(probs[pred_code] * 100.0)
    
    is_stroke = (pred_code == 1)
    label = "ISCHEMIC_STROKE_DETECTED" if is_stroke else "NORMAL_BASELINE"
    recommendation = (
        "CRITICAL: Acute cerebral asymmetry and alpha suppression detected. Initiate emergency stroke triage protocol."
        if is_stroke else
        "Symmetrical brainwave activity confirmed. Continue routine bedside monitoring."
    )
    
    return PredictionResponse(
        status="SUCCESS",
        prediction_code=pred_code,
        prediction_label=label,
        is_stroke_alert=is_stroke,
        confidence_percentage=round(confidence, 2),
        bsi=round(req.bsi, 4),
        adr=round(req.adr, 4),
        clinical_recommendation=recommendation
    )


@app.post("/predict-raw", response_model=PredictionResponse)
def predict_from_raw_signals(req: RawSignalInferenceRequest):
    """
    Ingests raw F3 and F4 microvolt arrays, runs Welch FFT to compute BSI and ADR,
    and returns the model prediction.
    """
    if len(req.f3_signal) < 64 or len(req.f4_signal) < 64:
        raise HTTPException(status_code=400, detail="Signal arrays must contain at least 64 samples.")
        
    features = extract_spectral_features(req.f3_signal, req.f4_signal, fs=req.sampling_rate)
    bsi, adr, alpha_p, delta_p = features
    
    feature_vec = np.array([[bsi, adr, alpha_p, delta_p]])
    pred_code = int(model.predict(feature_vec)[0])
    probs = model.predict_proba(feature_vec)[0]
    confidence = float(probs[pred_code] * 100.0)
    
    is_stroke = (pred_code == 1)
    label = "ISCHEMIC_STROKE_DETECTED" if is_stroke else "NORMAL_BASELINE"
    recommendation = (
        "CRITICAL: Acute cerebral asymmetry detected. Notify stroke team immediately."
        if is_stroke else
        "Physiological symmetry within normal limits."
    )
    
    return PredictionResponse(
        status="SUCCESS",
        prediction_code=pred_code,
        prediction_label=label,
        is_stroke_alert=is_stroke,
        confidence_percentage=round(confidence, 2),
        bsi=round(bsi, 4),
        adr=round(adr, 4),
        clinical_recommendation=recommendation
    )


@app.get("/simulate-packet")
def simulate_live_packet(state: str = "NORMAL"):
    """
    On-demand endpoint to simulate one live EEG packet and return both
    raw voltages and model classification.
    Valid states: NORMAL, STROKE_LEFT, STROKE_RIGHT, STRESS
    """
    virtual_hw.set_state(state.upper())
    pkt = virtual_hw.read_packet(duration_sec=1.0)
    
    # Extract features
    f3_uV = (np.array(pkt["f3_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
    f4_uV = (np.array(pkt["f4_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
    bsi, adr, a_p, d_p = extract_spectral_features(f3_uV, f4_uV, fs=256)
    
    # Model inference
    feature_vec = np.array([[bsi, adr, a_p, d_p]])
    pred_code = int(model.predict(feature_vec)[0])
    confidence = float(np.max(model.predict_proba(feature_vec)) * 100.0)
    
    return {
        "timestamp": pkt["timestamp"],
        "simulated_state": pkt["state"],
        "f3_latest_uV": pkt["f3_uV_instant"],
        "f4_latest_uV": pkt["f4_uV_instant"],
        "bsi": round(bsi, 4),
        "adr": round(adr, 4),
        "prediction_label": "STROKE" if pred_code == 1 else "NORMAL",
        "confidence": round(confidence, 2),
        "battery_level": pkt["hardware_battery_pct"]
    }


if __name__ == "__main__":
    import uvicorn
    print(">>> Starting COSMOS ML Engine Microservice on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
