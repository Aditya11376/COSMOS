"""
COSMOS Virtual Telemetry Streamer & Pipeline Gateway
====================================================
Continuously generates live virtual EEG telemetry, computes BSI/ADR,
runs the Random Forest model for inference, and dispatches the payload
to the Java Spring Boot enterprise backend (http://localhost:8080/api/telemetry/ingest).

Supports:
- Auto-cycling through NORMAL baseline and STROKE events
- Command-line arguments for custom target URL and interval
"""

import sys
import time
import requests
import joblib
import numpy as np

from virtual_hardware import CosmosVirtualHardware
from train_model import extract_spectral_features

# Target Java Spring Boot REST Endpoint
SPRING_BOOT_URL = "http://localhost:8085/api/telemetry/ingest"

# Load the trained ML model
MODEL_PATH = "cosmos_stroke_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    import os
    model = joblib.load(os.path.join(os.path.dirname(__file__), MODEL_PATH))

# Initialize Virtual Hardware
hw = CosmosVirtualHardware(sampling_rate=256)


def stream_telemetry(interval_sec: float = 0.5, patient_id: str = "PT-2026-AKGEC-01"):
    print("=" * 70)
    print("   COSMOS EDGE TELEMETRY STREAMER & PIPELINE GATEWAY")
    print("=" * 70)
    print(f" Target Backend : {SPRING_BOOT_URL}")
    print(f" Sampling Rate  : {hw.fs} Hz (10-Bit ADC emulation)")
    print(f" Stream Interval: {interval_sec}s per telemetry frame")
    print(" Press Ctrl+C to stop streaming.\n")

    cycle_count = 0
    current_state = "NORMAL"

    while True:
        cycle_count += 1

        # Simulate natural medical event timeline:
        # 1-12 cycles: Normal Baseline
        # 13-22 cycles: Acute Stroke Onset
        # 23+ cycles: Medical treatment / back to normal
        if cycle_count == 13:
            current_state = "STROKE_LEFT"
            print("\n>>> [!] SIMULATING EVENT: Acute Ischemic Stroke Onset (Left MCA)! [!]")
        elif cycle_count == 25:
            current_state = "NORMAL"
            print("\n>>> [i] SIMULATING EVENT: Thrombolytic Therapy Administered (Symmetry Restored).")
            cycle_count = 0

        hw.set_state(current_state)

        # 1. Read packet from Virtual Hardware (0.5s window = 128 samples)
        pkt = hw.read_packet(duration_sec=interval_sec)

        # 2. Convert ADC integers to microvolts for DSP
        f3_uV = (np.array(pkt["f3_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
        f4_uV = (np.array(pkt["f4_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0

        # 3. Extract Spectral Features (BSI, ADR, Powers)
        bsi, adr, alpha_power, delta_power = extract_spectral_features(f3_uV, f4_uV, fs=hw.fs)

        # 4. Run Machine Learning Inference
        feat_vec = np.array([[bsi, adr, alpha_power, delta_power]])
        pred_code = int(model.predict(feat_vec)[0])
        confidence = float(np.max(model.predict_proba(feat_vec)) * 100.0)
        pred_label = "STROKE" if pred_code == 1 else "NORMAL"

        # 5. Build Enriched Telemetry Payload for Spring Boot Backend
        payload = {
            "patientId": patient_id,
            "patientName": "Ramesh Kumar (Age 62)",
            "timestamp": pkt["timestamp"],
            "f3Voltage": pkt["f3_uV_instant"],
            "f4Voltage": pkt["f4_uV_instant"],
            "bsi": round(float(bsi), 4),
            "adr": round(float(adr), 4),
            "modelPrediction": pred_label,
            "modelConfidence": round(confidence, 1),
            "hardwareBattery": pkt["hardware_battery_pct"],
            "deviceStatus": "STREAMING",
            "simulatedState": current_state
        }

        # 6. Dispatch to Spring Boot via HTTP POST
        try:
            res = requests.post(SPRING_BOOT_URL, json=payload, timeout=0.8)
            status_str = f"HTTP {res.status_code}"
        except requests.exceptions.RequestException:
            status_str = "BACKEND OFFLINE (Awaiting Spring Boot)"

        # Terminal output indicator
        tag = "[CRITICAL ALERT]" if pred_code == 1 else "[NORMAL STABLE] "
        print(f"[{payload['timestamp']}] {tag} State: {current_state:<11} | BSI: {payload['bsi']:>6.3f} | ADR: {payload['adr']:>5.2f} | ML: {pred_label:<6} ({confidence:>4.1f}%) -> {status_str}")

        time.sleep(interval_sec)


if __name__ == "__main__":
    try:
        stream_telemetry()
    except KeyboardInterrupt:
        print("\n[STOP] Telemetry Streamer stopped by user.")
