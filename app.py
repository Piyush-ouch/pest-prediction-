from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from firebase_admin import credentials, db
import firebase_admin
from datetime import datetime, timedelta
import requests
import time
import threading

import os
import json

app = FastAPI()

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# -----------------------
# STARTUP INITIALIZATION
# -----------------------

# Handle Firebase Credentials (Env Var for Cloud, File for Local)
firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
if firebase_creds:
    # Parse JSON string from env var
    cred_dict = json.loads(firebase_creds)
    cred = credentials.Certificate(cred_dict)
else:
    # Fallback to local file
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    else:
        raise FileNotFoundError("No FIREBASE_CREDENTIALS env var or serviceAccountKey.json found.")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
})

autoencoder = load_model(
    "models/global_climate_autoencoder.h5",
    compile=False
)
scaler = joblib.load("models/global_climate_scaler.pkl")

with open("models/global_climate_threshold.txt") as f:
    THRESHOLD = float(f.read())

# -----------------------
# UTILITY FUNCTIONS
# -----------------------

def dew_point(temp, rh):
    a, b = 17.27, 237.7
    # Ensure RH is not 0 to avoid log error
    if rh <= 0: rh = 0.1
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100)
    return (b * alpha) / (a - alpha)

def get_latest_solar(lat, lon):
    # Default fallback if no lat/lon
    if not lat or not lon:
        return 0.0
        
    for i in range(0, 5): # Try last 5 days
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y%m%d")
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN"
            f"&latitude={lat}&longitude={lon}"
            f"&start={date}&end={date}&format=JSON"
        )
        try:
            r = requests.get(url, timeout=5).json()
            val = list(
                r["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].values()
            )[0]
            if val != -999.0: # NASA error code
                return val
        except:
            continue
    return 5.0 # Fallback average solar

def summarize(values):
    return np.mean(values), np.min(values), np.max(values)

# -------------------------------------------------------------------
# Soil‑moisture helper functions (grid‑level risk weighting)
# -------------------------------------------------------------------

def soil_category(soil):
    """Return DRY / MODERATE / WET based on percentage value."""
    if soil < 30:
        return "DRY"
    elif soil < 60:
        return "MODERATE"
    else:
        return "WET"


def grid_risk(farm_risk, soil):
    """Combine farm‑level climate risk with soil moisture category.
    Returns one of: NORMAL, WATCH, WARNING, HIGH.
    """
    soil_cat = soil_category(soil)

    if farm_risk == "NORMAL":
        return "NORMAL"

    if farm_risk == "WATCH":
        if soil_cat == "WET":
            return "WARNING"
        elif soil_cat == "MODERATE":
            return "WATCH"
        else:
            return "NORMAL"

    if farm_risk == "WARNING":
        if soil_cat == "WET":
            return "HIGH"
        elif soil_cat == "MODERATE":
            return "WARNING"
        else:
            return "WATCH"

    if farm_risk == "HIGH":
        if soil_cat == "DRY":
            return "WARNING"
        else:
            return "HIGH"


# -----------------------
# CORE LOGIC
# -----------------------

def process_field(user_id, field_id, env_data):
    """
    1. Archive live env data to history
    2. Read history
    3. Predict if enough data
    """
    if "temp" not in env_data or "hum" not in env_data:
        return

    # 1. Archive Data
    timestamp = str(int(time.time() * 1000))
    data_entry = {
        "temperature": float(env_data["temp"]),
        "humidity": float(env_data["hum"]),
        "timestamp": timestamp
    }
    
    # Check last entry to prevent duplicates (spam)
    # Target: users/{user_id}/history/{field_id}
    history_root = db.reference(f"users/{user_id}/history/{field_id}")
    last_entry = history_root.order_by_key().limit_to_last(1).get()
    
    if last_entry:
        last_ts = int(list(last_entry.keys())[0])
        current_ts = int(timestamp)
        # If less than 25 minutes (1500000 ms) has passed, skip archiving
        diff = current_ts - last_ts
        if diff < 1500000:
            print(f"Skipping archive for {field_id}: Last update was {diff/1000:.1f}s ago (Need 1500s)")
            # Do NOT return here; proceed to prediction logic
        else:
            # Save to history bucket only if time gap is met
            history_ref = db.reference(f"users/{user_id}/history/{field_id}")
            history_ref.child(timestamp).set(data_entry)
            print(f"✅ Archived data to users/{user_id}/history/{field_id}/{timestamp}")

    else:
        # Case where no history exists yet - create first entry
        history_ref = db.reference(f"users/{user_id}/history/{field_id}")
        history_ref.child(timestamp).set(data_entry)
        print(f"✅ Archived data to users/{user_id}/history/{field_id}/{timestamp}")
    
    # Fetch last 47 entries from historical_logs (we add current live data as the 48th)
    logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
    logs = logs_ref.order_by_key().limit_to_last(47).get()
    
    if not logs:
        print(f"No historical logs for {field_id}")
        return
    
    # Extract temperature and humidity from historical logs
    records = []
    for entry in logs.values():
        env = entry.get("env", {})
        temp = env.get("temp")
        hum = env.get("hum")
        if temp is None or hum is None:
            continue
        records.append({"temperature": float(temp), "humidity": float(hum)})

    # Append the CURRENT live reading as the latest record
    records.append({
        "temperature": float(env_data["temp"]),
        "humidity": float(env_data["hum"])
    })
    
    if len(records) < 48:
        print(f"Insufficient valid env data for {field_id}: {len(records)}/48")
        return

    # 2. Prepare for Prediction using records from historical_logs
    temps = [r["temperature"] for r in records]
    hums = [r["humidity"] for r in records]
    dews = [dew_point(t, h) for t, h in zip(temps, hums)]
    dpdiffs = [t - d for t, d in zip(temps, dews)]

    # Hardcoded Lat/Lon or fetch if available
    lat, lon = 20.5937, 78.9629
    if "lat" in env_data and "lon" in env_data:
         lat, lon = env_data["lat"], env_data["lon"]
    
    solar = get_latest_solar(lat, lon)

    features = []
    for arr in [temps, hums, dews, dpdiffs]:
        features.extend(summarize(arr))

    features.extend([solar, solar, solar])

    # 3. Predict
    X = scaler.transform([features])
    recon = autoencoder.predict(X)
    error = float(np.mean((X - recon) ** 2))

    risk = "NORMAL"
    if error > THRESHOLD:
        risk = "HIGH"
    elif error > (THRESHOLD * 0.7):
        risk = "WATCH"

    # 4. Write Prediction
    # OLD: users/{user_id}/fields/{field_id}/prediction
    # NEW: users/{user_id}/live_status/{field_id}/prediction
    
    reason = "Normal conditions"
    if risk == "HIGH":
        reason = "High humidity and low solar drying"
    elif risk == "WATCH":
        reason = "Elevated risk parameters detected"

    db.reference(
        f"users/{user_id}/live_status/{field_id}/prediction"
    ).set({
        "risk": risk,
        "anomaly_score": error,
        "confidence": round(error / THRESHOLD, 2),
        "reason": reason,
        "lastUpdated": datetime.utcnow().isoformat()
    })
    print(f"Prediction for {field_id}: {risk} (Err: {error:.4f}) Written to live_status")
    return risk


def prediction_loop():
    print("--- Starting Prediction Loop ---")
    while True:
        try:
            print("Fetching users...")
            users = db.reference("users").get(shallow=True) or {}
            print(f"Found {len(users)} users.")
            
            for user_id in users.keys():
                print(f"Checking user: {user_id}")
                # Check live_status
                ls_ref = db.reference(f"users/{user_id}/live_status")
                live_status = ls_ref.get()
                
                if not live_status:
                    print(f"No live_status for {user_id}")
                    continue
                
                for field_id, field_data in live_status.items():
                    print(f"Checking field: {field_id}")
                    if isinstance(field_data, dict) and "env" in field_data:
                        print(f"Processing env data for {field_id}...")
                        farm_risk = process_field(user_id, field_id, field_data["env"])
                        # Apply soil‑moisture based grid weighting for specific probes
                        probes = field_data.get("probes", {})
                        if probes:
                            for probe_id, probe_data in probes.items():
                                if not isinstance(probe_data, dict): continue
                                
                                raw_moisture = probe_data.get("raw_moisture")
                                if raw_moisture is not None:
                                    try:
                                        moisture_val = float(raw_moisture)
                                        probe_risk_level = grid_risk(farm_risk, moisture_val)
                                        
                                        # Write probe specific prediction
                                        db.reference(
                                            f"users/{user_id}/live_status/{field_id}/probes/{probe_id}/prediction"
                                        ).set(probe_risk_level)
                                        print(f"Probe {probe_id}: Moisture {moisture_val}% -> {probe_risk_level}")
                                    except ValueError as e:
                                        print(f"Invalid moisture for probe {probe_id}: {repr(raw_moisture)} Error: {e}")
                                    except Exception as e:
                                        print(f"Unexpected error for probe {probe_id}: {e}")
                    else:
                        print(f"Skipping {field_id} (No env data)")
                        
        except Exception as e:
            print(f"Global Loop Error: {e}")
            import traceback
            traceback.print_exc()
            
        print("Sleeping for 30 min...")
        time.sleep(1800)  # Run every 30 mins

@app.on_event("startup")
def start_background_tasks():
    print("Backend Starting - Launching Prediction Loop...")
    thread = threading.Thread(target=prediction_loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
