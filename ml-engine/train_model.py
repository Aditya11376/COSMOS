"""
COSMOS ML Training & Feature Pipeline
======================================
1. Uses the Virtual Hardware Generator to create a balanced dataset of 1,000 trials
   (500 Healthy Baseline trials, 500 Acute Ischemic Stroke trials).
2. Extracts clinical features from raw F3 & F4 voltages:
   - Brain Symmetry Index (BSI)
   - Alpha-to-Delta Ratio (ADR)
   - Alpha Band Power (8 - 13 Hz)
   - Delta Band Power (0.5 - 4 Hz)
3. Trains an explainable Machine Learning Classifier (Random Forest).
4. Evaluates accuracy, precision, recall, and feature importances.
5. Exports the trained model to 'cosmos_stroke_model.pkl'.
"""

import os
import joblib
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from virtual_hardware import CosmosVirtualHardware


# 1. CLINICAL FEATURE EXTRACTOR (BSI & ADR via WELCH FFT)
def extract_spectral_features(f3_signal, f4_signal, fs=256):
    """
    Transforms raw microvolt time-series from F3 and F4 nodes into
    the clinical biomarkers BSI and ADR using Welch's Power Spectral Density.
    """
    f3_arr = np.array(f3_signal, dtype=float)
    f4_arr = np.array(f4_signal, dtype=float)
    
    # Compute Power Spectral Density (PSD) using Welch method
    freqs, psd_f3 = welch(f3_arr, fs=fs, nperseg=min(len(f3_arr), fs))
    _, psd_f4 = welch(f4_arr, fs=fs, nperseg=min(len(f4_arr), fs))
    
    # Frequency band masks
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)   # Delta: 0.5 - 4.0 Hz (Ischemia / slow wave)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)  # Alpha: 8.0 - 13.0 Hz (Healthy conscious rhythm)
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)  # Total relevant spectrum: 0.5 - 30.0 Hz
    
    # Calculate integrated band powers (Area under PSD curve)
    p_delta_f3 = np.sum(psd_f3[delta_mask])
    p_alpha_f3 = np.sum(psd_f3[alpha_mask])
    p_total_f3 = np.sum(psd_f3[total_mask])
    
    p_delta_f4 = np.sum(psd_f4[delta_mask])
    p_alpha_f4 = np.sum(psd_f4[alpha_mask])
    p_total_f4 = np.sum(psd_f4[total_mask])
    
    # Formula 1: Brain Symmetry Index (BSI)
    # Measures normalized hemispheric power disparity
    bsi = abs(p_total_f4 - p_total_f3) / (p_total_f4 + p_total_f3 + 1e-9)
    
    # Formula 2: Alpha-to-Delta Ratio (ADR)
    # Measures the ratio of healthy awake rhythm to pathological slowing
    total_alpha = p_alpha_f3 + p_alpha_f4
    total_delta = p_delta_f3 + p_delta_f4
    adr = total_alpha / (total_delta + 1e-9)
    
    # Feature vector: [BSI, ADR, Total_Alpha_Power, Total_Delta_Power]
    return [float(bsi), float(adr), float(total_alpha), float(total_delta)]


# 2. DATASET GENERATION FROM VIRTUAL HARDWARE
def generate_training_dataset(num_trials_per_class=500, trial_duration_sec=2.0, fs=256):
    """
    Generates synthetic patient trials using the Virtual Hardware Digital Twin.
    Class 0 = Normal Baseline
    Class 1 = Acute Ischemic Stroke
    """
    print(f"[DATASET] Generating {num_trials_per_class * 2} clinical EEG trials ({trial_duration_sec}s windows)...")
    
    hw = CosmosVirtualHardware(sampling_rate=fs)
    X = []
    y = []
    
    # Generate Normal Baseline Trials (Class 0)
    hw.set_state("NORMAL")
    for _ in range(num_trials_per_class):
        # Read a 2-second continuous window
        pkt = hw.read_packet(duration_sec=trial_duration_sec)
        # Convert ADC counts back to microvolts for DSP
        f3_uV = (np.array(pkt["f3_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
        f4_uV = (np.array(pkt["f4_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
        
        features = extract_spectral_features(f3_uV, f4_uV, fs=fs)
        X.append(features)
        y.append(0)  # 0 = Normal
        
    # Generate Ischemic Stroke Trials (Class 1)
    hw.set_state("STROKE_LEFT")
    for _ in range(num_trials_per_class):
        pkt = hw.read_packet(duration_sec=trial_duration_sec)
        f3_uV = (np.array(pkt["f3_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
        f4_uV = (np.array(pkt["f4_adc_raw"]) * (3.3 / 1024.0) - 1.65) * 200.0
        
        features = extract_spectral_features(f3_uV, f4_uV, fs=fs)
        X.append(features)
        y.append(1)  # 1 = Stroke
        
    return np.array(X), np.array(y)


# 3. MODEL TRAINING AND EVALUATION PIPELINE
def train_and_export_model():
    fs = 256
    X, y = generate_training_dataset(num_trials_per_class=500, trial_duration_sec=2.0, fs=fs)
    
    # Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\n[TRAIN] Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Test predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100.0
    
    print("\n" + "=" * 60)
    print(f"      COSMOS ML STROKE CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Model Overall Accuracy: {accuracy:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal Baseline", "Ischemic Stroke"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Display Feature Importance (Why BSI and ADR matter!)
    feature_names = ["Brain Symmetry Index (BSI)", "Alpha-to-Delta Ratio (ADR)", "Alpha Power", "Delta Power"]
    importances = clf.feature_importances_
    print("\nFeature Importances (Deterministic Explainability):")
    for name, imp in zip(feature_names, importances):
        print(f"  - {name:<30}: {imp * 100:.2f}%")
        
    # Save the trained model to disk
    model_path = os.path.join(os.path.dirname(__file__), "cosmos_stroke_model.pkl")
    joblib.dump(clf, model_path)
    print(f"\n[EXPORT] Model successfully serialized to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_export_model()
