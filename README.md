# ML Backend for Pest Risk Prediction

This FastAPI backend serves as the "Brain" of the Agri Pest-Risk Prediction System.
It connects to Firebase, monitors live sensor data, archives it for history, and runs an Autoencoder ML model to predict pest risk based on climate anomalies.

## Setup

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Firebase Credentials**:
    Ensure `serviceAccountKey.json` is present in this directory.

3.  **ML Models**:
    Ensure the following files are in `models/`:
    - `global_climate_autoencoder.h5`
    - `global_climate_scaler.pkl`
    - `global_climate_threshold.txt`

## Running

Run the server:
```bash
python app.py
```
Or with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## How it works

1.  **Monitors** `users/{uid}/live_status/{field_id}/env` for new sensor data (Frequency: 30 mins).
2.  **Archives** valid data to `users/{uid}/fields/{field_id}/data/{timestamp}`.
3.  **Fetches** Solar Radiation data from NASA Power API.
4.  **Checks** if 48 historical data points exist.
5.  **Predicts** Risk (NORMAL/HIGH) using the Autoencoder.
6.  **Writes** results to `users/{uid}/fields/{field_id}/prediction`.

## API Endpoints

-   `GET /health`: Health check.
