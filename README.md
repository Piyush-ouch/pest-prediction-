# Agri-Pest Risk Prediction Backend (The Brain)

This directory serves as the **Intelligence Engine** for the Agri-Pest Risk Prediction System. It features two backends:
1. **Heuristic & Anomaly Autoencoder Backend (`app.py`)**: Runs traditional rules and generic climate anomaly detection.
2. **Spatio-Temporal Graph Neural Network Backend (`app_gnn.py`)**: Models fields as spatial graphs of sensor nodes and runs collaborative risk predictions.

---

## 🚀 Key Features

*   **Spatio-Temporal Graph Modeling**: Model localized sensors as nodes in a graph and pass messages between them to capture spatial correlations (e.g., disease spread, moisture leakage).
*   **Dual Temporal-Spatial Architecture**: A hybrid neural design utilizing `LSTM` sequential feature encoding followed by custom `GCN` layers.
*   **Self-Supervised Distillation Training**: Train the GNN model directly on historical Firebase logs using baseline heuristic classifications as weak labels.
*   **Dynamic Adjacency Construction**: Automatically calculates physical node adjacency matrices using coordinates (Latitude/Longitude) via a Radial Basis Function (RBF) kernel.
*   **NASA Power API Integration**: Dynamic, location-based solar radiation archiving to evaluate field drying rates.

---

## 🛠️ Graph Neural Network (GNN) Technical Details

Instead of predicting risk for each sensor probe independently, the **GNN Backend** treats the field as a spatial network.

```
       [Global Climate Context: Temp, Hum, Solar]
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │ Probe 1 │ │ Probe 2 │ │ Probe 3 │   <-- LSTM Sequence Encoder
        └────┬────┘ └────┬────┘ └────┬────┘
             │           │           │
             └───────────┼───────────┘
                         ▼
             ┌───────────────────────┐
             │ Graph Message Passing  │   <-- Custom GCN Adjacency Layers
             └───────────┬───────────┘
                         ▼
         [Collaborative Risk Predictions]
```

### 1. Graph Representation
*   **Nodes ($V$)**: Soil moisture probes in the field.
*   **Node Features ($X_i$)**: A sequence tensor of shape `(48, 4)` for each probe representing a 24-hour sequence (half-hour logs) of:
    $$\text{Features} = [\text{Temperature}, \text{Humidity}, \text{Soil Moisture}, \text{Solar Radiation}]$$
*   **Edge Weights ($A_{ij}$)**: Physical proximity connections calculated using the distance ($d$) between probes' GPS coordinates:
    $$A_{ij} = \exp\left(-\frac{d(i, j)^2}{2\sigma^2}\right)$$
    *(Connected if distance is within a spatial threshold)*

### 2. Model Architecture
1.  **Temporal Processing (LSTM)**: Each node's time-series sequence is fed into a shared **LSTM layer** ($64$ units) to extract localized climate trends:
    $$Z_i = \text{LSTM}(X_i) \in \mathbb{R}^{64}$$
2.  **Spatial Message Passing (GCN)**: Node representation embeddings are aggregated with physical neighbors using a custom Keras Graph Convolutional layer:
    $$H^{(l+1)} = \sigma\left(\hat{A} H^{(l)} W^{(l)} + b\right)$$
    where $\hat{A} = D^{-1/2}(A + I_N)D^{-1/2}$ is the symmetric normalized adjacency matrix.
3.  **Classification Head**: A dense layer applies a softmax activation over each node embedding to predict one of four categorical risk levels:
    $$\text{Risks} = [\text{NORMAL}, \text{WATCH}, \text{WARNING}, \text{HIGH}]$$

---

## 📂 Project Structure

```
ml_backend/
│
├── models/
│   ├── gnn_pest_model.h5                   # Trained Spatio-Temporal GNN weights
│   ├── global_climate_autoencoder (2).h5    # Anomaly detection fallback model
│   ├── global_climate_scaler (2).pkl        # Climate feature normalizer
│   └── global_climate_threshold (2).txt     # Anomaly threshold value
│
├── app.py                                  # Standard backend
├── app_gnn.py                              # GNN-based backend (Dynamic & Spatio-temporal)
├── add_mock_data.py                        # Populates Firebase with historical mock data
├── evaluate_model.py                       # Evaluates models offline
├── check_firebase.py                       # Verification check for database connections
└── requirements.txt                        # Python dependencies
```

---

## ⚡ Setup & Installation

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Firebase Credentials
Place your `serviceAccountKey.json` credentials file inside the `ml_backend/` folder (or set the `FIREBASE_CREDENTIALS` environment variable with your JSON string).

---

## 💻 Running the Servers

### Run the Standard Backend (Port 8008)
Uses the climate Autoencoder + rule heuristic predictions:
```bash
python ml_backend/app.py
```

### Run the GNN-based Backend (Port 8088)
Runs dynamic graph convolutional spatial message-passing predictions:
```bash
python ml_backend/app_gnn.py
```

---

## 📡 API Endpoints (GNN Backend)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | Health check & loads TensorFlow metadata. |
| **POST** | `/predict/{user_id}/{field_id}` | Manually triggers spatial GNN inference for a specific field. |
| **POST** | `/train` | Triggers background self-supervised distillation GNN training using database logs. |

---

## 📊 Database Schema (Realtime DB)

Predictions are updated under the user path in the following structure:
```json
{
  "users": {
    "{user_id}": {
      "live_status": {
        "{field_id}": {
          "crop": "wheat",
          "lat": 20.5937,
          "lon": 78.9629,
          "prediction": {
            "risk": "HIGH",
            "reason": "GNN spatial consensus: Outbreak level is HIGH."
          },
          "probes": {
            "{probe_id}": {
              "lat": 20.5938,
              "lon": 78.9630,
              "raw_moisture": 62,
              "prediction": "HIGH",
              "gnn_metadata": {
                "risk": "HIGH",
                "confidence": 0.88,
                "reason": "GNN prediction with 88% confidence. Avg local moisture: 62%."
              }
            }
          }
        }
      }
    }
  }
}
```
