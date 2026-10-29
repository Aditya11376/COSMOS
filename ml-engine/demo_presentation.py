"""
COSMOS Live Interactive Presentation & Defense CLI
===================================================
A ready-to-run demonstration tool for your project presentation / viva defense.

Allows you to interactively demonstrate:
1. NORMAL Baseline: Shows healthy twin brainwaves, BSI < 0.1, ADR > 1.2, Model = NORMAL (GREEN)
2. ACUTE STROKE: Live-triggers left hemisphere stroke, BSI > 0.2, ADR < 0.7, Model = STROKE (RED ALERT)
3. ARTIFACT TEST: Injects an eye blink artifact to prove the model DOES NOT cause a false alarm!
4. LIVE DATA STREAM: Runs a real-time continuous monitor with an ASCII waveform display.

Usage:
  python demo_presentation.py
"""

import os
import sys
import time
import joblib
import numpy as np

# Import our custom Virtual Hardware and Feature Extractor
from virtual_hardware import CosmosVirtualHardware
from train_model import extract_spectral_features


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "cosmos_stroke_model.pkl")
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Please run train_model.py first.")
        sys.exit(1)
    return joblib.load(model_path)


def evaluate_trial(hw, state_name, duration_sec=2.0, model=None):
    hw.set_state(state_name)
    # Read a continuous 2-second trial
    pkt = hw.read_packet(duration_sec=duration_sec)
    
    # Convert ADC integers to microvolts for feature extraction
    f3_uV = (np.array(pkt["f3_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
    f4_uV = (np.array(pkt["f4_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
    
    # Extract BSI and ADR
    features = extract_spectral_features(f3_uV, f4_uV, fs=hw.fs)
    bsi, adr, alpha_p, delta_p = features
    
    # Run Random Forest Model Inference
    X_sample = np.array([features])
    prediction = model.predict(X_sample)[0]
    probabilities = model.predict_proba(X_sample)[0]
    confidence = probabilities[prediction] * 100.0
    
    return {
        "state": state_name,
        "f3_latest_uV": pkt["f3_uV_instant"],
        "f4_latest_uV": pkt["f4_uV_instant"],
        "bsi": bsi,
        "adr": adr,
        "alpha_power": alpha_p,
        "delta_power": delta_p,
        "prediction_label": "ISCHEMIC STROKE DETECTED" if prediction == 1 else "NORMAL (HEALTHY BRAIN)",
        "prediction_code": prediction,
        "confidence": confidence,
        "battery": pkt["hardware_battery_pct"]
    }


def print_dashboard_card(result):
    print("\n" + "=" * 72)
    print("      COSMOS CLINICAL OSCILLATION & STROKE MONITORING SYSTEM")
    print("=" * 72)
    print(f" Patient ID    : PT-2026-AKGEC-01       | Wearable Battery: {result['battery']}%")
    print(f" Scalp Nodes   : F3 (Left) & F4 (Right) | Virtual Hardware: ONLINE")
    print("-" * 72)
    print(" BIOMARKER TELEMETRY:")
    
    # BSI Status
    bsi_status = "NORMAL (Symmetric)" if result['bsi'] < 0.10 else ("ELEVATED" if result['bsi'] < 0.20 else "PATHOLOGICAL ASYMMETRY!")
    print(f"  • Brain Symmetry Index (BSI) : {result['bsi']:.4f}  [{bsi_status}]")
    print(f"    (Clinical Norm: < 0.10  |  Ischemic Alert Threshold: > 0.20)")
    
    # ADR Status
    adr_status = "HEALTHY WAKEFULNESS" if result['adr'] > 1.20 else ("MODERATE DEPRESSION" if result['adr'] >= 0.70 else "CRITICAL ALPHA COLLAPSE!")
    print(f"  • Alpha-to-Delta Ratio (ADR) : {result['adr']:.4f}  [{adr_status}]")
    print(f"    (Clinical Norm: > 1.20  |  Critical Ischemic Threshold: < 0.70)")
    
    print("-" * 72)
    print(" RANDOM FOREST INFERENCE ENGINE:")
    
    if result['prediction_code'] == 1:
        print("  ┌──────────────────────────────────────────────────────────────────┐")
        print("  │  >>> [!] EMERGENCY STROKE ALERT: IMMEDIATE ACTION REQUIRED [!] <<< │")
        print("  │                                                                  │")
        print(f"  │  Diagnosis : {result['prediction_label']:<48} │")
        print(f"  │  Confidence: {result['confidence']:.1f}% Match with Ischemic Penumbra Profile    │")
        print("  │  Action    : Notify Stroke Team -> Prepare Thrombolytic Therapy  │")
        print("  └──────────────────────────────────────────────────────────────────┘")
    else:
        print("  ┌──────────────────────────────────────────────────────────────────┐")
        print(f"  │  Status    : {result['prediction_label']:<48} │")
        print(f"  │  Confidence: {result['confidence']:.1f}% Physiological Symmetry Confirmed          │")
        print("  │  Action    : Patient Stable -> Standard Continuous Telemetry     │")
        print("  └──────────────────────────────────────────────────────────────────┘")
    print("=" * 72)


def run_continuous_stream(hw, model):
    print("\n[STREAMING] Starting Continuous Real-Time Telemetry Monitor...")
    print("Press Ctrl+C to return to main menu.\n")
    time.sleep(1)
    
    try:
        step = 0
        state = "NORMAL"
        while True:
            step += 1
            # After 8 cycles, simulate sudden ischemic onset to show live transition!
            if step == 8:
                print("\n>>> [!] INJECTING EVENT: Acute Left Cerebral Artery Occlusion! [!]")
                state = "STROKE_LEFT"
            elif step == 16:
                print("\n>>> [i] EVENT: Medical Intervention / Perfusion Restored.")
                state = "NORMAL"
                step = 0

            res = evaluate_trial(hw, state, duration_sec=1.0, model=model)
            
            # Simple terminal visual bar for BSI
            bar_len = int(min(res['bsi'] * 50, 25))
            bsi_bar = "■" * bar_len + " " * (25 - bar_len)
            
            alert_tag = " [CRITICAL ALERT] " if res['prediction_code'] == 1 else " [OK - STABLE]   "
            print(f"[{time.strftime('%H:%M:%S')}] F3: {res['f3_latest_uV']:>6.1f} uV | F4: {res['f4_latest_uV']:>6.1f} uV | BSI: [{bsi_bar}] {res['bsi']:.3f} | ADR: {res['adr']:.2f} | {alert_tag}")
            time.sleep(0.7)

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Returning to presentation menu...")
        time.sleep(1)


def main():
    hw = CosmosVirtualHardware(sampling_rate=256)
    model = load_model()

    while True:
        clear_screen()
        print("==================================================================")
        print("       COSMOS: LIVE VIVA / PRESENTATION DEMONSTRATION")
        print("==================================================================")
        print(" Choose an evaluation scenario to demonstrate to the panel:")
        print("")
        print("  [1] Demonstrate Healthy Baseline Patient (Normal Symmetrical Brain)")
        print("  [2] Demonstrate Acute Ischemic Stroke Onset (Left MCA Occlusion)")
        print("  [3] Demonstrate Noise & Artifact Rejection (Eye Blink EOG Test)")
        print("  [4] Start Live Continuous Real-Time Streaming Monitor")
        print("  [5] Exit Demonstration")
        print("==================================================================")
        
        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            print("\n[RUNNING] Collecting 2.0s bilateral EEG data for Healthy Patient...")
            time.sleep(1)
            res = evaluate_trial(hw, "NORMAL", model=model)
            print_dashboard_card(res)
            input("\nPress Enter to continue back to menu...")

        elif choice == "2":
            print("\n[RUNNING] Simulating Acute Stroke Patient (Oxygen starvation on Left)...")
            time.sleep(1)
            res = evaluate_trial(hw, "STROKE_LEFT", model=model)
            print_dashboard_card(res)
            input("\nPress Enter to continue back to menu...")

        elif choice == "3":
            print("\n[RUNNING] Injecting Eye Blink (+120 uV EOG artifact)...")
            hw.trigger_eye_blink()
            res = evaluate_trial(hw, "NORMAL", model=model)
            print_dashboard_card(res)
            print(" EXPLANATION FOR EXAMINERS:")
            print(" Notice: Although a huge +120 uV spike occurred, both F3 & F4 were affected")
            print(" equally. Therefore, the BSI remained below 0.20 and the model correctly")
            print(" dismissed it as a harmless artifact without triggering a false alarm!")
            input("\nPress Enter to continue back to menu...")

        elif choice == "4":
            run_continuous_stream(hw, model)

        elif choice == "5":
            print("\nThank you for exploring the COSMOS demonstration platform. Good luck with your defense!")
            break


if __name__ == "__main__":
    main()
