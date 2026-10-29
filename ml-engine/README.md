# COSMOS ML Engine & Virtual Hardware Microservice

This microservice forms the **Biological Telemetry & Machine Learning Layer** of the **COSMOS (Clinical Oscillation and Stroke Monitoring Operating System)** platform.

It is responsible for:
1. Emulating the dual-channel frontal EEG headband hardware (F3 and F4 nodes).
2. Processing raw microvolt biosignals through an analog front-end (AFE) bandpass filter and 10-bit ADC quantizer.
3. Extracting clinical biomarkers: **Brain Symmetry Index (BSI)** and **Alpha-to-Delta Ratio (ADR)** via Welch Fast Fourier Transform (FFT).
4. Running a trained **Random Forest Classifier** for real-time acute ischemic stroke prediction.
5. Exposing REST inference endpoints and streaming live telemetry to the **Java Spring Boot Enterprise Backend**.

---

## Directory Architecture

```
ml-engine/
├── virtual_hardware.py       # Hardware Digital Twin (F3/F4 waveforms, AD8232 filter, 10-bit ADC)
├── train_model.py            # Dataset synthesizer, feature extraction, and model training
├── cosmos_stroke_model.pkl   # Serialized Random Forest classifier
├── app.py                    # FastAPI REST microservice (POST /predict, POST /predict-raw)
├── pipeline_streamer.py      # Live streaming client dispatching telemetry to Spring Boot
├── demo_presentation.py      # Interactive menu CLI tool for viva/presentations
├── requirements.txt          # Python dependencies
└── README.md                 # Service documentation
```

---

## Quick Start Guide

### 1. Install Dependencies
```bash
cd ml-engine
pip install -r requirements.txt
```

### 2. Train the Model (Optional - pre-trained model included)
```bash
python train_model.py
```
*Generates 1,000 biological EEG trials, extracts BSI & ADR features, evaluates accuracy, and exports `cosmos_stroke_model.pkl`.*

### 3. Run the FastAPI REST Microservice
```bash
python app.py
# Or via uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: **http://localhost:8000/docs**

#### Key Endpoints:
* `GET  /health` - Check microservice health and model specs.
* `POST /predict` - Send BSI and ADR values to receive an immediate diagnosis.
* `POST /predict-raw` - Send raw F3 & F4 microvolt arrays; the API computes BSI/ADR internally.
* `GET  /simulate-packet?state=NORMAL` - Generate an on-demand simulated EEG telemetry frame.

### 4. Start the Live Telemetry Streamer (Connects to Spring Boot)
Once your Java Spring Boot backend is running on port 8085:
```bash
python pipeline_streamer.py
```
*Streams telemetry frames every 500ms to `http://localhost:8085/api/telemetry/ingest`, automatically cycling through normal baseline and acute stroke onset to test emergency triage alerts.*

### 5. Run the Interactive Viva / Presentation Demo
```bash
python demo_presentation.py
```
*Provides an interactive console dashboard to showcase normal patients, stroke patients, and artifact rejection during project defense.*
