# COSMOS: ML Engine & Virtual Hardware Microservice

Biological Telemetry & Machine Learning Layer for **COSMOS (Clinical Oscillation and Stroke Monitoring Operating System)**.

🌐 **Production Cloud Endpoint:** [https://cosmos-ec3s.onrender.com](https://cosmos-ec3s.onrender.com)  
📖 **Interactive Swagger API Documentation:** [https://cosmos-ec3s.onrender.com/docs](https://cosmos-ec3s.onrender.com/docs)

---

## Core Capabilities
* **Virtual Hardware Digital Twin:** Emulates a dual-channel frontal EEG headband (F3 & F4 nodes) with 10-bit ADC quantization and AD8232 analog bandpass filtering.
* **Deterministic Feature Extraction:** Computes **Brain Symmetry Index (BSI)** and **Alpha-to-Delta Ratio (ADR)** via Welch Fast Fourier Transform (FFT).
* **Explainable AI:** Random Forest Classifier (100 decision trees) for real-time acute ischemic stroke prediction with zero black-box opacity.
* **Cloud-Native Deployment:** Deployed on Render via FastAPI and Docker, ready to stream telemetry to the Java Spring Boot enterprise backend (`port 8085`).

---

## Directory Architecture

```
ml-engine/
├── virtual_hardware.py       # Hardware Digital Twin (F3/F4 waveforms & ADC simulation)
├── train_model.py            # Dataset synthesizer & Random Forest training pipeline
├── cosmos_stroke_model.pkl   # Serialized trained Random Forest classifier
├── app.py                    # FastAPI Cloud Microservice (POST /predict)
├── pipeline_streamer.py      # Telemetry client pushing data to Spring Boot (:8085)
├── Dockerfile                # Production container deployment definition
└── requirements.txt          # Python dependencies
```

---

## Usage

### 1. Cloud REST API
The microservice is live and can be queried directly from any client or backend:
```bash
# Health check
curl -X GET "https://cosmos-ec3s.onrender.com/health"

# Stroke prediction
curl -X POST "https://cosmos-ec3s.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"bsi": 0.28, "adr": 0.42, "alpha_power": 20.0, "delta_power": 300.0}'
```

### 2. Stream Telemetry to Spring Boot
Once the Java Spring Boot backend is active on port `8085`, start the edge telemetry streamer:
```bash
python pipeline_streamer.py
```
*Streams biopotential frames every 500ms to `http://localhost:8085/api/telemetry/ingest`.*
