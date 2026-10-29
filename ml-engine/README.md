# COSMOS: ML Engine & Virtual Hardware Microservice

Biological Telemetry & Machine Learning Layer for **COSMOS (Clinical Oscillation and Stroke Monitoring Operating System)**.

🌐 **Live Cloud Deployment:** [https://cosmos-ec3s.onrender.com](https://cosmos-ec3s.onrender.com)  
📖 **Interactive Swagger API Docs:** [https://cosmos-ec3s.onrender.com/docs](https://cosmos-ec3s.onrender.com/docs)

---

## Core Capabilities
* **Virtual Hardware Digital Twin:** Emulates a dual-channel frontal EEG headband (F3 & F4 nodes) with 10-bit ADC quantization and AD8232 analog bandpass filtering.
* **Deterministic Feature Extraction:** Computes **Brain Symmetry Index (BSI)** and **Alpha-to-Delta Ratio (ADR)** via Welch Fast Fourier Transform (FFT).
* **Explainable AI:** Uses a Random Forest Classifier (100 decision trees) for real-time acute ischemic stroke prediction with zero black-box opacity.
* **Cloud & Edge Ready:** Runs locally or in the cloud via FastAPI, streaming real-time telemetry to the Java Spring Boot enterprise backend (`port 8085`).

---

## Repository Structure

```
ml-engine/
├── virtual_hardware.py       # Hardware Digital Twin (F3/F4 waveforms & ADC simulation)
├── train_model.py            # Dataset synthesizer & Random Forest training pipeline
├── cosmos_stroke_model.pkl   # Pre-trained serialized Random Forest model
├── app.py                    # FastAPI REST microservice (POST /predict)
├── pipeline_streamer.py      # Telemetry client pushing data to Spring Boot (:8085)
├── demo_presentation.py      # Interactive CLI demo for viva/project presentations
├── Dockerfile                # Production container deployment definition
└── requirements.txt          # Minimal Python dependencies
```

---

## Quick Start

### 1. Setup Environment
```bash
cd ml-engine
pip install -r requirements.txt
```

### 2. Run the REST API Locally
```bash
python app.py
```
*API will run at `http://localhost:8000`. Swagger documentation: `http://localhost:8000/docs`.*

### 3. Stream Telemetry to Spring Boot
Ensure the Java Spring Boot backend is active on port `8085`, then run:
```bash
python pipeline_streamer.py
```
*Streams biopotential telemetry frames every 500ms to `http://localhost:8085/api/telemetry/ingest`.*

### 4. Interactive Viva / Presentation Demo
```bash
python demo_presentation.py
```
*Console tool to demonstrate healthy baseline, acute stroke alert triggering, and artifact rejection during presentations.*
