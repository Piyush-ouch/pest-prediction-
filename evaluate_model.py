import numpy as np
import joblib
from tensorflow.keras.models import load_model
import keras
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Workaround for keras 3 incompatibilities
class CustomDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

class CustomInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

print("Loading models...")
autoencoder = load_model(
    "models/global_climate_autoencoder (2).h5",
    compile=False,
    custom_objects={"Dense": CustomDense, "InputLayer": CustomInputLayer}
)
scaler = joblib.load("models/global_climate_scaler (2).pkl")

with open("models/global_climate_threshold (2).txt") as f:
    THRESHOLD = float(f.read())

print(f"Model Threshold: {THRESHOLD}")

# Generate Synthetic Test Dataset
# The model expects 15 features:
# Temp(mean, min, max), Hum(mean, min, max), Dew(mean, min, max), DPDiff(mean, min, max), Solar(mean, min, max)

def generate_normal_data(num_samples=500):
    # Normal climate (e.g. Temp 15-28C, Hum 40-70%, Dew 10-20C, DPDiff 5-15, Solar 4-6)
    data = []
    for _ in range(num_samples):
        t_mean = np.random.uniform(15, 28)
        t_min = t_mean - np.random.uniform(2, 5)
        t_max = t_mean + np.random.uniform(2, 5)
        
        h_mean = np.random.uniform(40, 70)
        h_min = h_mean - np.random.uniform(5, 10)
        h_max = h_mean + np.random.uniform(5, 10)
        
        d_mean = np.random.uniform(10, 20)
        d_min = d_mean - np.random.uniform(2, 4)
        d_max = d_mean + np.random.uniform(2, 4)
        
        dp_mean = t_mean - d_mean
        dp_min = t_min - d_min
        dp_max = t_max - d_max
        
        s_mean = np.random.uniform(4, 6)
        s_min = s_mean - np.random.uniform(0.5, 1)
        s_max = s_mean + np.random.uniform(0.5, 1)
        
        features = [t_mean, t_min, t_max, h_mean, h_min, h_max, d_mean, d_min, d_max, dp_mean, dp_min, dp_max, s_mean, s_min, s_max]
        data.append(features)
    return np.array(data)

def generate_anomalous_data(num_samples=500):
    # Anomalous climate (e.g. extremely hot/humid favoring pests, or extreme heatwaves)
    data = []
    for _ in range(num_samples):
        # High Pest Risk (e.g. Very High Temp + Very High Humidity)
        t_mean = np.random.uniform(30, 40)
        t_min = t_mean - np.random.uniform(1, 3)
        t_max = t_mean + np.random.uniform(1, 3)
        
        h_mean = np.random.uniform(80, 95)
        h_min = h_mean - np.random.uniform(2, 5)
        h_max = h_mean + np.random.uniform(2, 5)
        
        d_mean = np.random.uniform(25, 30)
        d_min = d_mean - np.random.uniform(1, 2)
        d_max = d_mean + np.random.uniform(1, 2)
        
        dp_mean = t_mean - d_mean
        dp_min = t_min - d_min
        dp_max = t_max - d_max
        
        s_mean = np.random.uniform(2, 3) # Low solar/cloudy
        s_min = s_mean - np.random.uniform(0.5, 1)
        s_max = s_mean + np.random.uniform(0.5, 1)
        
        features = [t_mean, t_min, t_max, h_mean, h_min, h_max, d_mean, d_min, d_max, dp_mean, dp_min, dp_max, s_mean, s_min, s_max]
        data.append(features)
    return np.array(data)

print("Generating synthetic test dataset (1000 samples)...")
normal_X = generate_normal_data(500)
anomalous_X = generate_anomalous_data(500)

# Combine and create labels (0 = Normal, 1 = Anomaly/Pest Risk)
X = np.vstack([normal_X, anomalous_X])
y_true = np.array([0]*500 + [1]*500)

# Scale
X_scaled = scaler.transform(X)

# Predict reconstruction
print("Running model predictions...")
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# Classify based on threshold
y_pred = (mse > THRESHOLD).astype(int)

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*40)
print("MODEL PERFORMANCE (SYNTHETIC TEST SET)")
print("="*40)
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print("="*40)
print(f"Threshold Used: {THRESHOLD}")
print(f"Average Normal MSE: {np.mean(mse[:500]):.5f}")
print(f"Average Anomalous MSE: {np.mean(mse[500:]):.5f}")
