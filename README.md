# Traffic-Based Route Guidance System (TBRGS)

**COS30019 Introduction to AI — Assignment 2 Part B**  
Swinburne University of Technology

---

## Overview

TBRGS is a congestion-aware intelligent navigation system for the City of Boroondara. It trains three machine learning models on historical VicRoads **SCATS** (Sydney Coordinated Adaptive Traffic System) traffic volume data, predicts future traffic flow, converts the predictions into road segment travel times, and feeds those costs into a modified A\* search engine to recommend the top-5 fastest routes between any two SCATS intersections.

---

## Features

- **Three ML models** for 15-minute traffic flow prediction: LSTM, GRU, and Random Forest
- **Parabolic traffic flow model** converts predicted vehicle counts to average speed, then to travel time in seconds
- **30-second fixed delay** applied per controlled intersection
- **Modified A\*** with a time-admissible heuristic (straight-line distance ÷ 60 km/h)
- **Yen's K-Shortest Paths** returns up to 5 ranked alternative routes
- **Flask + Folium web interface** renders routes on a live OpenStreetMap basemap with colour-coded polylines
- **Tkinter desktop GUI** as a standalone alternative to the web app
- **Coordinate correction system** fixes known map-drift in raw SCATS latitude/longitude data

---

## ML Model Performance (Test Set)

Evaluated on the held-out 15% test split of the October 2006 Boroondara dataset.

| Model | MSE | RMSE | NRMSE | MAE |
|---|---|---|---|---|
| LSTM | 646.21 | 25.42 | 0.1314 | 19.24 |
| GRU | **635.21** | **25.20** | **0.1302** | 19.02 |
| Random Forest | 643.63 | 25.37 | 0.1311 | **18.66** |

GRU achieves the lowest RMSE overall; Random Forest achieves the lowest MAE. All three models perform within 2% of each other, suggesting the 1-hour (4 × 15-min) sliding window captures the dominant temporal pattern regardless of model architecture.

---

## Project Structure

```
codespace/
├── web_app.py              # Flask web application (main interface)
├── gui_app.py              # Tkinter desktop GUI (alternative interface)
├── route_guidance.py       # Modified A*, Yen's K-paths, travel-time formula
├── build_real_network.py   # SCATS graph construction and coordinate correction
├── data_preprocessing.py   # Data loading, cleaning, normalisation, sliding window
├── ml_models.py            # LSTM, GRU, Random Forest architecture definitions
├── train_evaluate.py       # Model training, evaluation metrics, chart generation
├── map_visualization.py    # Standalone static network topology plot
├── test_tbrgs.py           # 22-case unittest suite
├── config.json             # Runtime configuration (defaults, hyperparameters)
├── requirements.txt        # Python dependency list
├── install_dependencies.bat  # Windows one-click dependency installer
├── start_server.bat          # Windows one-click web server launcher
├── models/
│   ├── lstm_model.keras      # Pre-trained LSTM weights
│   ├── gru_model.keras       # Pre-trained GRU weights
│   └── random_forest_model.pkl
├── data/
│   ├── Scats Data October 2006.xls
│   ├── SCATSSiteListingSpreadsheet_VicRoads.xls
│   └── Traffic_Count_Locations_with_LONG_LAT.csv
├── outputs/
│   ├── predictions_comparison.png  # Actual vs predicted (first 100 test steps)
│   ├── metrics_comparison.png      # MSE/RMSE/NRMSE/MAE bar charts
│   ├── loss_curves.png             # LSTM & GRU training vs validation loss
│   ├── real_network_topology.png   # Static SCATS road network plot
│   └── route_map.html              # Interactive Folium route map
└── notebooks/
    └── regression_tree_2B.ipynb    # Exploratory Decision Tree experiments (reference only)
```

---

## Installation

**Prerequisite:** Python 3.8 or higher must be installed and available on `PATH`.

### Option A — Windows (one-click)

1. Double-click **`install_dependencies.bat`** to install all required packages.
2. Double-click **`start_server.bat`** to launch the Flask server.
3. Open `http://127.0.0.1:5000` in your browser.

### Option B — Manual (any OS)

```bash
pip install -r requirements.txt
python web_app.py
```

Then open `http://127.0.0.1:5000`.

### Option C — Desktop GUI

```bash
python gui_app.py
```

---

## Usage

### Web Interface

1. Select **Origin** and **Destination** from the SCATS site dropdowns (e.g. `0970` → `4821`).
2. Choose an **ML Model** (LSTM / GRU / Random Forest).
3. Click **Find Routes**.
4. The map displays up to 5 colour-coded routes ranked by estimated travel time. Route costs and intersection names are shown in the results panel.

### Configuration

Edit `config.json` to change runtime defaults without modifying source code:

```json
{
    "default_origin": "0970",
    "default_destination": "4821",
    "speed_kmh": 60,
    "intersection_delay_seconds": 30,
    "default_ml_model": "GRU",
    "available_models": ["LSTM", "GRU", "Random Forest"],
    "top_k": 5,
    "window_size": 4,
    "data_file": "data/Scats Data October 2006.xls"
}
```

---

## How It Works

1. **Graph construction** — `build_real_network.py` reads SCATS site coordinates from the Excel dataset, applies per-site and global coordinate corrections, and connects each intersection to its 3 nearest neighbours by Haversine distance to produce a weighted directed graph.

2. **Traffic prediction** — The selected ML model receives the most recent 4 traffic volume readings (1 hour of history) as a sliding window and predicts the next 15-minute vehicle count.

3. **Travel time conversion** — Predicted flow is mapped to average speed using the parabolic fundamental diagram:

   ```
   flow = -1.4648375 × speed² + 93.75 × speed
   travel_time = (distance / speed) × 3600 + 30  [seconds]
   ```

   Speed is clamped to [32, 60] km/h. The +30 s term accounts for the average intersection signal delay.

4. **Pathfinding** — Modified A\* (time-admissible heuristic: `euclidean_km / 60 × 3600`) finds the optimal route. Yen's algorithm then enumerates up to 4 alternative paths.

5. **Rendering** — The Flask app returns an embedded Folium map with colour-coded polylines (red = best, green = 2nd, blue = 3rd, orange = 4th, purple = 5th) and a summary table of travel times.

---

## Re-training Models

Pre-trained weights are included in `models/`. To retrain from scratch:

```bash
python train_evaluate.py
```

This will:
- Reload and preprocess the dataset
- Train LSTM, GRU, and Random Forest
- Print test-set metrics to console
- Save updated weights to `models/`
- Save updated charts to `outputs/` (including the new `loss_curves.png`)

Training runs for up to 50 epochs with `EarlyStopping(patience=10)`.

---

## Testing

```bash
python -m unittest test_tbrgs -v
```

The 22 test cases cover:

- Travel time formula correctness (free-flow, congested, zero-distance, extreme flow)
- Edge weight update structure
- A\* search (normal, same O=D, unreachable, invalid node)
- Yen's K-paths (multiple paths, ascending cost order, result structure, unreachable pair)
- Heuristic admissibility
- `config.json` structure and valid default SCATS IDs
- Sliding window shapes and values
- Real graph structural integrity
- Numeric node sort-key ordering

All 22 tests pass on a clean installation with pre-trained model files present.

---
