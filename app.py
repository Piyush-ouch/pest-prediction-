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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

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

import keras

# Workaround for keras 3 incompatibilities with older model saves
class CustomDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

class CustomInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

autoencoder = load_model(
    "models/global_climate_autoencoder (2).h5",
    compile=False,
    custom_objects={"Dense": CustomDense, "InputLayer": CustomInputLayer}
)
scaler = joblib.load("models/global_climate_scaler (2).pkl")

with open("models/global_climate_threshold (2).txt") as f:
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

    if farm_risk in ("NORMAL", "LOW"):
        return farm_risk

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
        if soil_cat == "WET":
            return "HIGH"
        else:
            # Both DRY and MODERATE reduce to WARNING
            return "WARNING"


# -----------------------
# CORE LOGIC
# -----------------------

def process_field(user_id, field_id, env_data=None, crop_type=None):
    """
    1. (Optional) Archive live env data to history
    2. Read date-wise historical_logs (latest date)
    3. Predict if enough data
    Returns: (risk_level, latest_probes) tuple or (None, {})
    """
    # 1. Archive Data (only if live env data is provided)
    if env_data and "temp" in env_data and "hum" in env_data:
        timestamp = str(int(time.time() * 1000))
        data_entry = {
            "temperature": float(env_data["temp"]),
            "humidity": float(env_data["hum"]),
            "timestamp": timestamp
        }
        
        history_root = db.reference(f"users/{user_id}/history/{field_id}")
        last_entry = history_root.order_by_key().limit_to_last(1).get()
        
        if last_entry:
            last_ts = int(list(last_entry.keys())[0])
            current_ts = int(timestamp)
            diff = current_ts - last_ts
            if diff < 1500000:
                print(f"Skipping archive for {field_id}: Last update was {diff/1000:.1f}s ago (Need 1500s)")
            else:
                history_ref = db.reference(f"users/{user_id}/history/{field_id}")
                history_ref.child(timestamp).set(data_entry)
                print(f"✅ Archived data to users/{user_id}/history/{field_id}/{timestamp}")
        else:
            history_ref = db.reference(f"users/{user_id}/history/{field_id}")
            history_ref.child(timestamp).set(data_entry)
            print(f"✅ Archived data to users/{user_id}/history/{field_id}/{timestamp}")
    
    # 2. Fetch from historical_logs - date-wise structure
    # Path: users/{uid}/historical_logs/{field_id}/{date}/{push_key}/env/temp|hum
    dates_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
    dates = dates_ref.get(shallow=True)
    
    if not dates:
        print(f"No historical logs dates for {field_id}")
        return None, {}
    
    # Sort dates descending (newest first)
    sorted_dates = sorted(dates.keys(), reverse=True)
    
    records = []
    latest_probes = {}
    found_latest_probes = False
    dates_checked = 0
    
    import re
    
    # Iterate backwards through dates until we have 48 records
    for date_str in sorted_dates:
        if len(records) >= 48:
            break
            
        # Ensure that date_str is actually a date (YYYY-MM-DD) and not a legacy push key
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            continue
            
        logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}/{date_str}")
        # Fetch up to 48 per date to be safe, ordered by key (timestamp)
        # We need descending order to get the most recent ones first
        logs_query = logs_ref.order_by_key().limit_to_last(48).get()
        
        if not logs_query:
            continue
            
        # Parse this date's entries (they come ordered oldest -> newest for this date)
        # We process them in reverse (newest -> oldest) to get the most recent first
        date_entries = []
        for push_key, entry in reversed(list(logs_query.items())):
            if len(records) + len(date_entries) >= 48:
                break
                
            temp = None
            hum = None
            
            # Extract env data
            if isinstance(entry, dict) and "env" in entry:
                env_entry = entry["env"]
                temp = env_entry.get("temp")
                hum = env_entry.get("hum")
            
            if not found_latest_probes and isinstance(entry, dict) and "probes" in entry:
                latest_probes = entry["probes"]
                found_latest_probes = True
            
            if temp is not None and hum is not None:
                date_entries.append({"temperature": float(temp), "humidity": float(hum)})

        records.extend(date_entries)
        dates_checked += 1
    
    if len(records) < 47:
        print(f"Insufficient valid env data for {field_id}: {len(records)}/48 (Checked {dates_checked} dates)")
        return None, {}
        
    print(f"📅 Collected {len(records)} records for {field_id} across {dates_checked} dates.")
    
    # Reverse the records list so it goes oldest -> newest for the prediction logic
    records.reverse()

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

    # Helper to find longest sustained duration (streak in half-hour ticks)
    def max_streak(cond_func):
        max_s = 0
        curr_s = 0
        for t, h in zip(temps, hums):
            if cond_func(t, h):
                curr_s += 1
                max_s = max(max_s, curr_s)
            else:
                curr_s = 0
        return max_s

    if crop_type == "tomato":
        # New tomato pest logic based on sustained conditions (4 hours = 8 ticks, 6 hours = 12 ticks, 12 hours = 24 ticks)
        thrips_streak = max_streak(lambda t, h: 12 <= t <= 25)
        aphids_streak = max_streak(lambda t, h: h >= 65)
        mites_streak = max_streak(lambda t, h: t > 27 and h < 50)
        heat_streak = max_streak(lambda t, h: t >= 32)
        dew_streak = max_streak(lambda t, h: 20 <= t <= 27 and h >= 80)
        
        risk = "NORMAL"
        reasons = []
        pests = []
        diseases = []
        
        # Aphids & Whiteflies (High Humidity)
        if aphids_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Aphids/Whiteflies (High Humidity sustained for >12hrs)")
            pests.extend(["Aphids", "Whiteflies"])
        elif aphids_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Aphids/Whiteflies conditions sustained for >6hrs")
            pests.extend(["Aphids", "Whiteflies"])
            
        # Red Spider Mites (Hot & Dry)
        if mites_streak >= 16: # 8+ hours
            risk = "HIGH"
            reasons.append("High risk of Red Spider Mites (Hot & Dry sustained for >8hrs)")
            pests.append("Red Spider Mites")
        elif mites_streak >= 8: # 4+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Red Spider Mite conditions sustained for >4hrs")
            pests.append("Red Spider Mites")
            
        # Diseases (Persistent Dew)
        if dew_streak >= 12: # 6+ hours
            risk = "HIGH"
            reasons.append("High risk of Diseases (Persistent Moisture/Dew sustained for >6hrs)")
            diseases.append("Fungal/Bacterial Diseases")
            
        # Plant stress
        if heat_streak >= 8: # 4+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Plant stress due to extreme heat sustained for >4hrs")
            
        if thrips_streak >= 24: # 12+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Favorable temperatures for Thrips sustained for >12hrs")
            pests.append("Thrips")
            
        if not reasons:
            reasons.append("Conditions are currently not favorable for major tomato pests")
            
        reason = " | ".join(set(reasons))
        # Ensure lists are unique while maintaining basic order
        pests = list(set(pests))
        diseases = list(set(diseases))
        error = 0.0  # Bypass model anomaly score
        confidence = 1.0
    elif crop_type == "cotton":
        # Cotton pest logic based on sustained conditions
        hot_dry_streak = max_streak(lambda t, h: 30 <= t <= 35 and h < 70)
        warm_humid_streak = max_streak(lambda t, h: 32 <= t <= 34 and h >= 80)
        mod_humid_streak = max_streak(lambda t, h: 20 <= t <= 32 and h >= 80)
        
        risk = "NORMAL"
        reasons = []
        pests = []
        diseases = []
        
        # Hot-Dry: Whitefly, Mealybug, Bollworms, CLCuD
        if hot_dry_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Whitefly/Bollworms (Hot & Dry sustained for >12hrs)")
            pests.extend(["Cotton Whitefly", "Cotton Mealybug", "Pink Bollworm", "American Bollworm"])
            diseases.append("Cotton Leaf Curl Disease (CLCuD)")
        elif hot_dry_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Hot & Dry conditions developing (Whitefly/Bollworms)")
            pests.extend(["Cotton Whitefly", "Cotton Mealybug", "Pink Bollworm"])
            
        # Warm-Humid: Jassids, Thrips
        if warm_humid_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Jassids/Thrips (Warm & Humid sustained for >12hrs)")
            pests.extend(["Jassids", "Thrips"])
        elif warm_humid_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Warm & Humid conditions developing (Jassids/Thrips)")
            pests.extend(["Jassids", "Thrips"])
            
        # Moderate-Humid: Aphids
        if mod_humid_streak >= 24: # 12+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Favorable conditions for Aphids (Moderate & Humid sustained for >12hrs)")
            pests.append("Aphids")
            
        if not reasons:
            reasons.append("Conditions are currently not favorable for major cotton pests")
            
        reason = " | ".join(set(reasons))
        pests = list(set(pests))
        diseases = list(set(diseases))
        error = 0.0
        confidence = 1.0
    elif crop_type == "wheat":
        # Wheat pest logic based on sustained conditions
        aphid_streak = max_streak(lambda t, h: 10 <= t <= 26 and h >= 60)
        termite_streak = max_streak(lambda t, h: t > 26 and h < 55)
        armyworm_streak = max_streak(lambda t, h: 24 <= t <= 30 and h >= 70)
        
        risk = "NORMAL"
        reasons = []
        pests = []
        diseases = []
        
        # Cool & Humid: Aphids
        if aphid_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Aphids (Cool-Moderate & Humid sustained for >12hrs)")
            pests.append("Wheat Aphid")
        elif aphid_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Cool-Moderate & Humid conditions developing (Aphids)")
            pests.append("Wheat Aphid")
            
        # Warm & Dry: Termites
        if termite_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Termites (Warm & Dry sustained for >12hrs)")
            pests.append("Termites")
        elif termite_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Warm & Dry conditions developing (Termites)")
            pests.append("Termites")
            
        # Warm & Humid: Oriental Armyworm
        if armyworm_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Oriental Armyworm (Warm & Humid sustained for >12hrs)")
            pests.append("Oriental Armyworm")
        elif armyworm_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Warm & Humid conditions developing (Armyworm)")
            pests.append("Oriental Armyworm")
            
        if not reasons:
            reasons.append("Conditions are currently not favorable for major wheat pests")
            
        reason = " | ".join(set(reasons))
        pests = list(set(pests))
        diseases = list(set(diseases))
        error = 0.0
        confidence = 1.0
    elif crop_type == "rice":
        # Rice pest logic based on sustained conditions
        warm_humid_streak = max_streak(lambda t, h: 25 <= t <= 34 and h >= 80)
        sunny_dry_streak = max_streak(lambda t, h: 29 <= t <= 35 and h < 75)
        
        risk = "NORMAL"
        reasons = []
        pests = []
        diseases = []
        
        # Warm & Humid: Yellow Stem Borer, BPH, Gall Midge
        if warm_humid_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Stem Borer/BPH/Gall Midge (Warm & Humid sustained for >12hrs)")
            pests.extend(["Yellow Stem Borer", "Brown Planthopper (BPH)", "Rice Gall Midge"])
        elif warm_humid_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Warm & Humid conditions developing (Borer/BPH/Midge)")
            pests.extend(["Yellow Stem Borer", "Brown Planthopper (BPH)", "Rice Gall Midge"])
            
        # Sunny & Dry: Rice Leaf Folder
        if sunny_dry_streak >= 24: # 12+ hours
            risk = "HIGH"
            reasons.append("High risk of Rice Leaf Folder (Sunny & Dry sustained for >12hrs)")
            pests.append("Rice Leaf Folder")
        elif sunny_dry_streak >= 12: # 6+ hours
            if risk != "HIGH": risk = "WARNING"
            reasons.append("Warning: Sunny & Dry conditions developing (Leaf Folder)")
            pests.append("Rice Leaf Folder")
            
        if not reasons:
            reasons.append("Conditions are currently not favorable for major rice pests")
            
        reason = " | ".join(set(reasons))
        pests = list(set(pests))
        diseases = list(set(diseases))
        error = 0.0
        confidence = 1.0
    else:
        # Generic Model Logic
        features = []
        for arr in [temps, hums, dews, dpdiffs]:
            features.extend(summarize(arr))

        features.extend([solar, solar, solar])

        # 3. Predict
        X = scaler.transform([features])
        recon = autoencoder.predict(X)
        error = float(np.mean((X - recon) ** 2))

        risk = "NORMAL" # default instead of NORMAL
        if error > THRESHOLD:
            risk = "HIGH"
        elif error > (THRESHOLD * 0.7):
            risk = "WARNING" # match typical WARNING rather than WATCH if user wants WARNING

        # 4. Write Prediction
        # OLD: users/{user_id}/fields/{field_id}/prediction
        # NEW: users/{user_id}/live_status/{field_id}/prediction
        
        reason = "Normal conditions"
        if risk == "HIGH":
            reason = "High humidity and low solar drying"
        elif risk == "WARNING":
            reason = "Elevated risk parameters detected"
            
        confidence = round(error / THRESHOLD, 2)
        pests = []
        diseases = []

    db.reference(
        f"users/{user_id}/live_status/{field_id}/prediction"
    ).set({
        "risk": risk.upper(),
        "anomaly_score": error,
        "confidence": confidence,
        "reason": reason,
        "pests": pests,
        "diseases": diseases,
        "lastUpdated": datetime.utcnow().isoformat()
    })
    print(f"Prediction for {field_id} (Crop: {crop_type}): {risk.upper()} (Err: {error:.4f}) Written to live_status")
    return risk.upper(), latest_probes


def prediction_loop():
    print("--- Starting Prediction Loop ---")
    while True:
        try:
            print("Fetching users...")
            users = db.reference("users").get(shallow=True) or {}
            print(f"Found {len(users)} users.")
            
            for user_id in users.keys():
                print(f"Checking user: {user_id}")
                
                # Get fields from historical_logs (date-wise structure)
                hist_ref = db.reference(f"users/{user_id}/historical_logs")
                hist_fields = hist_ref.get(shallow=True) or {}
                
                if not hist_fields:
                    print(f"No historical_logs for {user_id}")
                    continue
                
                # Get live_status for crop type and other metadata
                ls_ref = db.reference(f"users/{user_id}/live_status")
                live_status = ls_ref.get() or {}
                
                for field_id in hist_fields.keys():
                    # Skip non-field entries (e.g. field_A_events)
                    if field_id.endswith("_events"):
                        continue
                    print(f"Processing field: {field_id}")
                    
                    # Get crop type from live_status if available
                    crop_type = ""
                    if field_id in live_status and isinstance(live_status[field_id], dict):
                        crop_type = live_status[field_id].get("crop", "").lower()
                    
                    # Get live env data if available (for optional archiving)
                    env_data = None
                    if field_id in live_status and isinstance(live_status[field_id], dict):
                        if "env" in live_status[field_id]:
                            env_data = live_status[field_id]["env"]
                    
                    # Process field - reads from date-wise historical_logs
                    result = process_field(
                        user_id, field_id, env_data=env_data, crop_type=crop_type
                    )
                    
                    if result is None or result[0] is None:
                        print(f"Skipping probes for {field_id} (No farm_risk available yet)")
                        continue
                    
                    farm_risk, latest_probes = result
                    
                    # Apply soil-moisture based grid weighting using only currently active probes from live_status
                    if field_id in live_status and "probes" in live_status[field_id]:
                        active_probes = live_status[field_id]["probes"]
                        for probe_id, probe_data in active_probes.items():
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
    port = int(os.getenv("PORT", 8008))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
