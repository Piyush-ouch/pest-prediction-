from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from firebase_admin import credentials, db
import firebase_admin
from datetime import datetime, timedelta
import requests
import time
import threading
import os
import json
import re

app = FastAPI(title="Pest Prediction GNN Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# PATH RESOLUTION HELPER
# -----------------------
def resolve_path(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

# -----------------------
# FIREBASE INITIALIZATION
# -----------------------
firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
if firebase_creds:
    cred_dict = json.loads(firebase_creds)
    cred = credentials.Certificate(cred_dict)
else:
    key_path = resolve_path("serviceAccountKey.json")
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
    else:
        raise FileNotFoundError(f"No FIREBASE_CREDENTIALS env var or serviceAccountKey.json found at {key_path}")

# Prevent duplicate app initialization
try:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
    })
    print("[INFO] Firebase initialized successfully.")
except ValueError:
    print("[WARN] Firebase already initialized.")

# -----------------------
# CUSTOM GCN LAYER & MODEL DEFINITION
# -----------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class GCNLayer(layers.Layer):
    """
    Custom Graph Convolution Network (GCN) layer in Keras.
    Implements: H_new = Activation( A_norm * H * W + b )
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self._activation_arg = activation
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape is a list of two shapes: [X_shape, A_shape]
        # X shape: (batch_size, num_nodes, in_features)
        # A shape: (batch_size, num_nodes, num_nodes)
        in_features = input_shape[0][-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_features, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        x, a = inputs
        # 1. Compute A * X (Batch Matrix Multiplication)
        ax = tf.matmul(a, x)
        # 2. Multiply by Kernel (W)
        out = tf.matmul(ax, self.kernel)
        # 3. Add Bias
        out = tf.nn.bias_add(out, self.bias)
        # 4. Apply Activation
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = super(GCNLayer, self).get_config()
        config.update({
            "units": self.units,
            "activation": self._activation_arg
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TemporalReshape(layers.Layer):
    def __init__(self, time_steps=48, num_features=4, **kwargs):
        super(TemporalReshape, self).__init__(**kwargs)
        self.time_steps = time_steps
        self.num_features = num_features

    def call(self, x):
        shape = tf.shape(x)
        batch = shape[0]
        nodes = shape[1]
        return tf.reshape(x, (batch * nodes, self.time_steps, self.num_features))

    def get_config(self):
        config = super(TemporalReshape, self).get_config()
        config.update({
            "time_steps": self.time_steps,
            "num_features": self.num_features
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class SpatialReshape(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialReshape, self).__init__(**kwargs)

    def call(self, inputs):
        lstm_out, original_in = inputs
        shape = tf.shape(original_in)
        batch = shape[0]
        nodes = shape[1]
        return tf.reshape(lstm_out, (batch, nodes, 64))

    def get_config(self):
        return super(SpatialReshape, self).get_config()


def build_dynamic_gnn_model(time_steps=48, num_features=4, num_classes=4):
    """
    Builds a Spatio-Temporal GNN.
    - Input: Node Features shape (batch, nodes, time_steps, features)
    - Input: Normalized Adjacency shape (batch, nodes, nodes)
    - Output: Risk prediction probabilities shape (batch, nodes, num_classes)
    """
    x_in = layers.Input(shape=(None, time_steps, num_features), name="node_features")
    a_in = layers.Input(shape=(None, None), name="adjacency")
    
    # 1. Temporal sequence encoder: Process the time series for each probe individually
    x_reshaped = TemporalReshape(time_steps=time_steps, num_features=num_features, name="temporal_reshape")(x_in)
    
    # LSTM layer
    lstm_out = layers.LSTM(64, activation='tanh', return_sequences=False)(x_reshaped)
    # Shape: (batch * nodes, 64)
    
    # 2. Reshape back to spatial representation: (batch, nodes, 64)
    x_node_embeds = SpatialReshape(name="spatial_reshape")([lstm_out, x_in])
    
    # 3. GCN Layers
    h = GCNLayer(units=64, activation='relu', name="gcn_1")([x_node_embeds, a_in])
    h = GCNLayer(units=32, activation='relu', name="gcn_2")([h, a_in])
    
    # 4. Dense Classifier Output
    predictions = layers.Dense(num_classes, activation='softmax', name="output")(h)
    
    model = models.Model(inputs=[x_in, a_in], outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Load or initialize GNN
GNN_MODEL_PATH = resolve_path("models/gnn_pest_model.h5")
gnn_model = None
model_lock = threading.Lock()

def load_or_init_gnn():
    global gnn_model
    with model_lock:
        models_dir = os.path.dirname(GNN_MODEL_PATH)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        if os.path.exists(GNN_MODEL_PATH):
            try:
                gnn_model = tf.keras.models.load_model(
                    GNN_MODEL_PATH,
                    custom_objects={
                        "GCNLayer": GCNLayer,
                        "TemporalReshape": TemporalReshape,
                        "SpatialReshape": SpatialReshape
                    }
                )
                print(f"[INFO] Loaded existing GNN model from {GNN_MODEL_PATH}")
                return
            except Exception as e:
                print(f"[WARN] Failed to load GNN model: {e}. Re-initializing...")
                
        gnn_model = build_dynamic_gnn_model()
        gnn_model.save(GNN_MODEL_PATH)
        print(f"[NEW] Created and saved new GNN model to {GNN_MODEL_PATH}")

load_or_init_gnn()


# -----------------------
# FALLBACK LEGACY MODELS (For Anomaly detection & rules baseline)
# -----------------------
import keras
class CustomDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

class CustomInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

autoencoder = None
scaler = None
THRESHOLD = 0.5

ae_path = resolve_path("models/global_climate_autoencoder (2).h5")
scaler_path = resolve_path("models/global_climate_scaler (2).pkl")
threshold_path = resolve_path("models/global_climate_threshold (2).txt")

if os.path.exists(ae_path):
    try:
        autoencoder = tf.keras.models.load_model(
            ae_path,
            compile=False,
            custom_objects={"Dense": CustomDense, "InputLayer": CustomInputLayer}
        )
        scaler = joblib.load(scaler_path)
        with open(threshold_path) as f:
            THRESHOLD = float(f.read())
        print("[INFO] Loaded fallback global climate autoencoder model successfully.")
    except Exception as e:
        print(f"[WARN] Could not load fallback autoencoder: {e}")

# -----------------------
# UTILITY FUNCTIONS
# -----------------------

def dew_point(temp, rh):
    a, b = 17.27, 237.7
    if rh <= 0: rh = 0.1
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100)
    return (b * alpha) / (a - alpha)

def get_latest_solar(lat, lon):
    if not lat or not lon:
        return 0.0
    for i in range(0, 5):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y%m%d")
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN"
            f"&latitude={lat}&longitude={lon}"
            f"&start={date}&end={date}&format=JSON"
        )
        try:
            r = requests.get(url, timeout=5).json()
            val = list(r["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].values())[0]
            if val != -999.0:
                return val
        except:
            continue
    return 5.0

def summarize(values):
    return np.mean(values), np.min(values), np.max(values)

def soil_category(soil):
    if soil < 30:
        return "DRY"
    elif soil < 60:
        return "MODERATE"
    else:
        return "WET"

def grid_risk(farm_risk, soil):
    soil_cat = soil_category(soil)
    if farm_risk in ("NORMAL", "LOW"):
        return farm_risk
    if farm_risk == "WATCH":
        return "WARNING" if soil_cat == "WET" else ("WATCH" if soil_cat == "MODERATE" else "NORMAL")
    if farm_risk == "WARNING":
        return "HIGH" if soil_cat == "WET" else ("WARNING" if soil_cat == "MODERATE" else "WATCH")
    if farm_risk == "HIGH":
        return "HIGH" if soil_cat == "WET" else "WARNING"
    return "NORMAL"

# -----------------------
# GRAPH & SPATIAL UTILITIES
# -----------------------

def compute_normalized_adjacency(coords, sigma=0.01, dist_threshold=0.1):
    """
    Computes a symmetric normalized adjacency matrix using an RBF kernel.
    A_norm = D^{-1/2} * (A + I) * D^{-1/2}
    """
    num_nodes = len(coords)
    if num_nodes == 0:
        return np.empty((0, 0))
    
    A = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                A[i, j] = 0.0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= dist_threshold:
                    A[i, j] = np.exp(- (dist ** 2) / (2 * (sigma ** 2)))
                else:
                    A[i, j] = 0.0
                    
    # Self-loops
    A_tilde = A + np.eye(num_nodes)
    
    # Normalize
    degrees = np.sum(A_tilde, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_norm

# -----------------------
# BASELINE RISK RULES (Used for training targets / distillation)
# -----------------------

def calculate_baseline_risk(crop_type, temps, hums, dews, dpdiffs, solar, moisture_val):
    """Runs the legacy app.py decision rule logic to assign baseline label."""
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

    risk = "NORMAL"
    
    if crop_type == "tomato":
        aphids_streak = max_streak(lambda t, h: h >= 65)
        mites_streak = max_streak(lambda t, h: t > 27 and h < 50)
        dew_streak = max_streak(lambda t, h: 20 <= t <= 27 and h >= 80)
        heat_streak = max_streak(lambda t, h: t >= 32)
        thrips_streak = max_streak(lambda t, h: 12 <= t <= 25)
        
        if aphids_streak >= 24 or mites_streak >= 16 or dew_streak >= 12:
            risk = "HIGH"
        elif aphids_streak >= 12 or mites_streak >= 8 or heat_streak >= 8 or thrips_streak >= 24:
            risk = "WARNING"
            
    elif crop_type == "cotton":
        hot_dry_streak = max_streak(lambda t, h: 30 <= t <= 35 and h < 70)
        warm_humid_streak = max_streak(lambda t, h: 32 <= t <= 34 and h >= 80)
        mod_humid_streak = max_streak(lambda t, h: 20 <= t <= 32 and h >= 80)
        
        if hot_dry_streak >= 24 or warm_humid_streak >= 24:
            risk = "HIGH"
        elif hot_dry_streak >= 12 or warm_humid_streak >= 12 or mod_humid_streak >= 24:
            risk = "WARNING"
            
    elif crop_type == "wheat":
        aphid_streak = max_streak(lambda t, h: 10 <= t <= 26 and h >= 60)
        termite_streak = max_streak(lambda t, h: t > 26 and h < 55)
        armyworm_streak = max_streak(lambda t, h: 24 <= t <= 30 and h >= 70)
        
        if aphid_streak >= 24 or termite_streak >= 24 or armyworm_streak >= 24:
            risk = "HIGH"
        elif aphid_streak >= 12 or termite_streak >= 12 or armyworm_streak >= 12:
            risk = "WARNING"
            
    elif crop_type == "rice":
        warm_humid_streak = max_streak(lambda t, h: 25 <= t <= 34 and h >= 80)
        sunny_dry_streak = max_streak(lambda t, h: 29 <= t <= 35 and h < 75)
        
        if warm_humid_streak >= 24 or sunny_dry_streak >= 24:
            risk = "HIGH"
        elif warm_humid_streak >= 12 or sunny_dry_streak >= 12:
            risk = "WARNING"
            
    else:
        # Fallback to autoencoder anomaly scoring
        if autoencoder and scaler:
            features = []
            for arr in [temps, hums, dews, dpdiffs]:
                features.extend(summarize(arr))
            features.extend([solar, solar, solar])
            X_scale = scaler.transform([features])
            recon = autoencoder.predict(X_scale)
            error = float(np.mean((X_scale - recon) ** 2))
            if error > THRESHOLD:
                risk = "HIGH"
            elif error > (THRESHOLD * 0.7):
                risk = "WARNING"
        else:
            risk = "NORMAL"

    # Weigh by node soil moisture
    node_risk = grid_risk(risk, moisture_val)
    return node_risk

# -----------------------
# DATA ASSEMBLY
# -----------------------

def fetch_field_data(user_id, field_id, live_status, num_steps=48):
    """
    Fetches past N entries, lines up env parameters and local probe values,
    and returns features, adjacency, node metadata.
    """
    # 1. Fetch historical logs
    dates_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
    dates = dates_ref.get(shallow=True)
    if not dates:
        return None
    
    sorted_dates = sorted(dates.keys(), reverse=True)
    records = []
    
    for date_str in sorted_dates:
        if len(records) >= num_steps:
            break
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            continue
            
        logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}/{date_str}")
        logs_query = logs_ref.order_by_key().limit_to_last(num_steps).get()
        if not logs_query:
            continue
            
        for push_key, entry in reversed(list(logs_query.items())):
            if len(records) >= num_steps:
                break
            
            temp, hum, probes = None, None, {}
            if isinstance(entry, dict):
                if "env" in entry:
                    temp = entry["env"].get("temp")
                    hum = entry["env"].get("hum")
                if "probes" in entry:
                    probes = entry["probes"]
            
            if temp is not None and hum is not None:
                records.append({
                    "temperature": float(temp),
                    "humidity": float(hum),
                    "probes": probes
                })
                
    if len(records) < num_steps:
        return None
        
    records.reverse() # Order oldest -> newest
    
    # 2. Extract active probes list from live_status
    active_probes_dict = live_status.get(field_id, {}).get("probes", {})
    if not active_probes_dict:
        return None
        
    probe_ids = list(active_probes_dict.keys())
    num_nodes = len(probe_ids)
    
    # Coordinates mapping
    coords = []
    for idx, pid in enumerate(probe_ids):
        pdata = active_probes_dict[pid]
        lat = pdata.get("lat")
        lon = pdata.get("lon")
        if lat is not None and lon is not None:
            coords.append([float(lat), float(lon)])
        else:
            # Fallback layout: arrange probes in a virtual layout
            coords.append([idx * 0.02, 0.0])
            
    coords = np.array(coords)
    A_norm = compute_normalized_adjacency(coords)
    
    # 3. Construct features for each node/probe
    # Shape of Node features: (num_nodes, num_steps, 4)
    # Features: [temp, hum, moisture, solar]
    field_lat = float(live_status.get(field_id, {}).get("lat", 20.5937))
    field_lon = float(live_status.get(field_id, {}).get("lon", 78.9629))
    solar_val = get_latest_solar(field_lat, field_lon)
    
    X = np.zeros((num_nodes, num_steps, 4))
    
    # Keep track of latest known moisture per probe to handle missing logs
    last_known_moisture = {pid: 50.0 for pid in probe_ids}
    
    for t in range(num_steps):
        rec = records[t]
        temp = rec["temperature"]
        hum = rec["humidity"]
        env_probes = rec["probes"]
        
        for n_idx, pid in enumerate(probe_ids):
            # Extract probe moisture
            raw_m = None
            if pid in env_probes and isinstance(env_probes[pid], dict):
                raw_m = env_probes[pid].get("raw_moisture")
            
            if raw_m is not None:
                last_known_moisture[pid] = float(raw_m)
                
            X[n_idx, t, 0] = temp
            X[n_idx, t, 1] = hum
            X[n_idx, t, 2] = last_known_moisture[pid]
            X[n_idx, t, 3] = solar_val
            
    return {
        "X": X,
        "A": A_norm,
        "probe_ids": probe_ids,
        "records": records,
        "solar": solar_val
    }

# -----------------------
# INFERENCE PIPELINE
# -----------------------

def process_field_gnn(user_id, field_id, live_status, crop_type):
    """Computes pest prediction for each probe in a field using GNN."""
    data = fetch_field_data(user_id, field_id, live_status)
    if data is None:
        print(f"[WARN] Insufficient data to construct graph for {field_id}")
        return
        
    X, A, probe_ids = data["X"], data["A"], data["probe_ids"]
    
    # Reshape features to fit batch dimension (batch_size=1)
    X_batch = np.expand_dims(X, axis=0) # (1, num_nodes, 48, 4)
    A_batch = np.expand_dims(A, axis=0) # (1, num_nodes, num_nodes)
    
    # Run GNN Prediction
    with model_lock:
        predictions = gnn_model.predict([X_batch, A_batch], verbose=0)
        
    # predictions shape: (1, num_nodes, 4)
    pred_classes = np.argmax(predictions[0], axis=1)
    confidence_scores = np.max(predictions[0], axis=1)
    
    risk_labels = ["NORMAL", "WATCH", "WARNING", "HIGH"]
    
    # Write output to live status per probe
    for idx, pid in enumerate(probe_ids):
        risk = risk_labels[pred_classes[idx]]
        confidence = float(confidence_scores[idx])
        
        # Calculate reason based on temporal patterns
        avg_moisture = np.mean(X[idx, :, 2])
        reason = f"GNN prediction with {confidence*100:.1f}% confidence. Avg local moisture: {avg_moisture:.1f}%."
        
        # Write probe specific prediction
        db.reference(
            f"users/{user_id}/live_status/{field_id}/probes/{pid}/prediction"
        ).set(risk)
        
        db.reference(
            f"users/{user_id}/live_status/{field_id}/probes/{pid}/gnn_metadata"
        ).set({
            "risk": risk,
            "confidence": confidence,
            "reason": reason,
            "lastUpdated": datetime.utcnow().isoformat()
        })
        print(f"[NODE] Node {pid}: GNN Risk -> {risk} (Conf: {confidence:.2f})")
        
    # Summarize field level risk (max of all nodes)
    highest_class = np.max(pred_classes)
    field_risk = risk_labels[highest_class]
    
    db.reference(
        f"users/{user_id}/live_status/{field_id}/prediction"
    ).update({
        "risk": field_risk,
        "reason": f"GNN spatial consensus: Outbreak level is {field_risk}.",
        "lastUpdated": datetime.utcnow().isoformat()
    })
    print(f"[FIELD] Field {field_id}: Consensus Risk -> {field_risk}")

# -----------------------
# BACKGROUND PREDICTION LOOP
# -----------------------

def prediction_loop():
    print("--- Starting GNN Prediction Loop ---")
    while True:
        try:
            print("Fetching users...")
            users = db.reference("users").get(shallow=True) or {}
            print(f"Found {len(users)} users.")
            
            for user_id in users.keys():
                hist_fields = db.reference(f"users/{user_id}/historical_logs").get(shallow=True) or {}
                live_status = db.reference(f"users/{user_id}/live_status").get() or {}
                
                for field_id in hist_fields.keys():
                    if field_id.endswith("_events"):
                        continue
                        
                    crop_type = ""
                    if field_id in live_status and isinstance(live_status[field_id], dict):
                        crop_type = live_status[field_id].get("crop", "").lower()
                        
                    print(f"[RUN] Running GNN prediction on {field_id} (Crop: {crop_type})...")
                    process_field_gnn(user_id, field_id, live_status, crop_type)
                    
        except Exception as e:
            print(f"[ERROR] Error in prediction loop: {e}")
            
        print("Sleeping for 30 min...")
        time.sleep(1800)

# -----------------------
# SELF-SUPERVISED GNN TRAINING
# -----------------------

def train_gnn_model_task():
    """
    Self-supervised training loop.
    Extracts history logs, generates "silver labels" using legacy baseline rules,
    and fine-tunes the GNN model parameters to approximate them.
    """
    print("[TRAIN] Starting Self-Supervised GNN Training...")
    try:
        users = db.reference("users").get(shallow=True) or {}
        
        train_X = []
        train_A = []
        train_Y = []
        
        for user_id in users.keys():
            hist_fields = db.reference(f"users/{user_id}/historical_logs").get(shallow=True) or {}
            live_status = db.reference(f"users/{user_id}/live_status").get() or {}
            
            for field_id in hist_fields.keys():
                if field_id.endswith("_events"): continue
                
                # Fetch maximum history length for better training samples
                data = fetch_field_data(user_id, field_id, live_status, num_steps=48)
                if data is None: continue
                
                X, A, probe_ids = data["X"], data["A"], data["probe_ids"]
                records, solar = data["records"], data["solar"]
                
                crop_type = ""
                if field_id in live_status and isinstance(live_status[field_id], dict):
                    crop_type = live_status[field_id].get("crop", "").lower()
                
                # We build features and targets for each node
                node_targets = []
                for n_idx, pid in enumerate(probe_ids):
                    temps = [r["temperature"] for r in records]
                    hums = [r["humidity"] for r in records]
                    dews = [dew_point(t, h) for t, h in zip(temps, hums)]
                    dpdiffs = [t - d for t, d in zip(temps, dews)]
                    moisture_val = X[n_idx, -1, 2] # final step moisture
                    
                    # Generate silver target
                    target_risk = calculate_baseline_risk(
                        crop_type, temps, hums, dews, dpdiffs, solar, moisture_val
                    )
                    
                    # One-hot encoding mapping
                    y_idx = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "HIGH": 3}.get(target_risk, 0)
                    y_onehot = np.zeros(4)
                    y_onehot[y_idx] = 1.0
                    node_targets.append(y_onehot)
                
                train_X.append(X)
                train_A.append(A)
                train_Y.append(np.array(node_targets))
                
        if len(train_X) == 0:
            print("[WARN] No valid training samples found in Firebase database history logs.")
            return
            
        print(f"[DATA] Assembled {len(train_X)} graph training samples.")
        
        # Train on each sample graph individually since they can have a variable number of nodes
        epochs = 10
        with model_lock:
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(len(train_X)):
                    # Add batch dimensions
                    X_sample = np.expand_dims(train_X[i], axis=0) # (1, nodes, 48, 4)
                    A_sample = np.expand_dims(train_A[i], axis=0) # (1, nodes, nodes)
                    Y_sample = np.expand_dims(train_Y[i], axis=0) # (1, nodes, 4)
                    
                    history = gnn_model.fit(
                        [X_sample, A_sample], Y_sample,
                        epochs=1, verbose=0
                    )
                    epoch_loss += history.history['loss'][0]
                print(f"[EPOCH] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_X):.4f}")
            
            # Save trained parameters
            gnn_model.save(GNN_MODEL_PATH)
            print(f"[SAVE] Trained GNN model parameters saved to {GNN_MODEL_PATH}")
            
    except Exception as e:
        print(f"[ERROR] GNN training failed: {e}")
        import traceback
        traceback.print_exc()

# -----------------------
# FASTAPI ENDPOINTS
# -----------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Pest Prediction GNN Service",
        "tensorflow_version": tf.__version__,
        "model_loaded": gnn_model is not None
    }

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """Triggers self-supervised distillation training in the background."""
    background_tasks.add_task(train_gnn_model_task)
    return {"status": "Training initiated in the background."}

@app.post("/predict/{user_id}/{field_id}")
def run_manual_prediction(user_id: str, field_id: str):
    """Manually triggers GNN inference for a specific user field."""
    live_status = db.reference(f"users/{user_id}/live_status").get() or {}
    if field_id not in live_status:
        raise HTTPException(status_code=404, detail="Field not found in live status.")
        
    crop_type = live_status[field_id].get("crop", "").lower()
    process_field_gnn(user_id, field_id, live_status, crop_type)
    return {"status": "GNN prediction complete."}


# Start background prediction loop
@app.on_event("startup")
def start_background_tasks():
    print("Backend Starting - Launching GNN Prediction Loop...")
    thread = threading.Thread(target=prediction_loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8088))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
