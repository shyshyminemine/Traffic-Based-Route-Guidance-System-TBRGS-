"""
web_app.py
==========
Interactive Flask web application for the Intelligent Route Guidance System.
Users can pick Origin, Destination, ML Model from dropdowns and get
Top-5 routes rendered live on an OpenStreetMap via Folium.

Run:  python web_app.py
Open: http://127.0.0.1:5000
"""
import os
import sys
import warnings
import json

# Ensure imports and file paths resolve relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import subprocess
import folium
from folium import plugins
import joblib
from flask import Flask, render_template_string, request
from tensorflow.keras.models import load_model

from build_real_network import build_real_boroondara_graph
from data_preprocessing import prepare_dataloaders
from route_guidance import (yens_k_shortest_paths, build_time_weighted_edges,
                             run_parta_search)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Load configuration from config.json
# ═══════════════════════════════════════════════════════════════════════
CONFIG = {}
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
    print(f"Loaded config.json: {CONFIG}")
else:
    print("Warning: config.json not found, using built-in defaults.")

DATA_FILE        = CONFIG.get("data_file", "data/Scats Data October 2006.xls")
DEFAULT_ORIGIN   = CONFIG.get("default_origin", "0970")
DEFAULT_DEST     = CONFIG.get("default_destination", "4821")
DEFAULT_MODEL    = CONFIG.get("default_ml_model", "GRU")
TOP_K            = CONFIG.get("top_k", 5)
WINDOW_SIZE      = CONFIG.get("window_size", 4)

# ═══════════════════════════════════════════════════════════════════════
#  Startup: load everything once so each request is fast
# ═══════════════════════════════════════════════════════════════════════
print("Loading road network...")
NODES, EDGES_KM, SITE_NAMES = build_real_boroondara_graph(DATA_FILE, K=3)
SCATS_LIST = sorted(NODES.keys())

print("Loading ML models...")
MODELS = {}
if os.path.exists("models/lstm_model.keras"):
    MODELS["LSTM"] = load_model("models/lstm_model.keras")
if os.path.exists("models/gru_model.keras"):
    MODELS["GRU"] = load_model("models/gru_model.keras")
if os.path.exists("models/random_forest_model.pkl"):
    MODELS["Random Forest"] = joblib.load("models/random_forest_model.pkl")

print("Preparing sample prediction data...")
_, _, _, SCALER, arrays = prepare_dataloaders(DATA_FILE, window_size=WINDOW_SIZE)
SAMPLE_SEQ = arrays[4][0:1]   # first test window

print(f"Ready — {len(NODES)} SCATS sites, {len(MODELS)} models loaded.\n")

# ═══════════════════════════════════════════════════════════════════════
ROUTE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
ROUTE_NAMES  = ["Rank 1 (Best)", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]

app = Flask(__name__)


PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Intelligent Route Guidance System</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'Inter',sans-serif; background:#0f172a; color:#e2e8f0; }

    /* ── Top bar ───────────────────────────────── */
    .topbar {
      background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
      padding:14px 28px; display:flex; align-items:center; gap:18px;
      flex-wrap:wrap; box-shadow:0 2px 12px rgba(0,0,0,.4);
    }
    .topbar h1 {
      font-size:18px; font-weight:700; margin-right:auto;
      background:linear-gradient(90deg,#38bdf8,#818cf8);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .topbar label { font-size:13px; color:#94a3b8; }
    .topbar select, .topbar button {
      padding:7px 12px; border-radius:6px; font-size:13px; border:1px solid #475569;
      background:#1e293b; color:#e2e8f0; outline:none;
    }
    .topbar select:focus { border-color:#38bdf8; }
    .topbar button {
      background:linear-gradient(135deg,#2563eb,#7c3aed);
      border:none; color:#fff; font-weight:600; cursor:pointer;
      padding:8px 22px; transition:transform .15s;
    }
    .topbar button:hover { transform:scale(1.04); }

    /* ── Main area ─────────────────────────────── */
    .container { display:flex; height:calc(100vh - 56px); }

    /* sidebar */
    .sidebar {
      width:320px; min-width:320px; background:#1e293b;
      padding:16px; overflow-y:auto; border-right:1px solid #334155;
    }
    .sidebar h2 { font-size:15px; margin-bottom:12px; color:#94a3b8; }
    .route-card {
      background:#0f172a; border:1px solid #334155; border-radius:8px;
      padding:12px; margin-bottom:10px; transition:border-color .2s;
    }
    .route-card:hover { border-color:#38bdf8; }
    .route-rank { font-weight:700; font-size:14px; }
    .route-time { font-size:13px; color:#94a3b8; margin:4px 0; }
    .route-path { font-size:12px; color:#64748b; word-break:break-all; }

    .flow-badge {
      display:inline-block; background:#312e81; color:#a5b4fc; font-size:12px;
      padding:4px 10px; border-radius:20px; margin-top:8px;
    }

    /* map */
    .map-frame { flex:1; }
    .map-frame iframe { width:100%; height:100%; border:none; }

    .no-route { color:#f87171; margin-top:20px; font-size:14px; }
  </style>
</head>
<body>
  <form class="topbar" method="GET" action="/">
    <h1>Intelligent Route Guidance System</h1>
    <label>Origin
      <select name="origin">
        {% for s in scats_list %}
        <option value="{{s}}" {{'selected' if s==origin}}>{{s}} — {{names[s]}}</option>
        {% endfor %}
      </select>
    </label>
    <label>Destination
      <select name="dest">
        {% for s in scats_list %}
        <option value="{{s}}" {{'selected' if s==dest}}>{{s}} — {{names[s]}}</option>
        {% endfor %}
      </select>
    </label>
    <label>Model
      <select name="model">
        {% for m in models %}
        <option value="{{m}}" {{'selected' if m==model_name}}>{{m}}</option>
        {% endfor %}
      </select>
    </label>
    <button type="submit">Find Routes</button>
  </form>

  <div class="container">
    <div class="sidebar">
      <h2>Top-5 Recommended Routes</h2>
      <div class="flow-badge">Predicted flow: {{ "%.0f"|format(flow) }} veh / 15 min</div>
      <br><br>
      {% if routes %}
        {% for r in routes %}
        <div class="route-card" style="border-left:4px solid {{r.color}}">
          <div class="route-rank" style="color:{{r.color}}">{{r.label}}</div>
          <div class="route-time">Est. travel time: <b>{{r.time}}</b> min</div>
          <div class="route-path">{{r.path_str}}</div>
        </div>
        {% endfor %}
      {% else %}
        <p class="no-route">No routes found between the selected nodes.</p>
      {% endif %}
    </div>
    <div class="map-frame">
      <iframe srcdoc="{{map_html}}"></iframe>
    </div>
  </div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
def predict_flow(model, model_name):
    if model_name == "Random Forest":
        X = SAMPLE_SEQ.reshape(SAMPLE_SEQ.shape[0], -1)
        ps = model.predict(X)[0]
    else:
        ps = model.predict(SAMPLE_SEQ, verbose=0)[0][0]
    return max(0.0, SCALER.inverse_transform([[ps]])[0][0])


def build_folium_map(top_k, predicted_flow, model_name, origin, dest):
    """Return the Folium map HTML as a string."""
    lats = [p[1] for p in NODES.values()]
    lons = [p[0] for p in NODES.values()]
    m = folium.Map(
        location=[sum(lats)/len(lats), sum(lons)/len(lons)],
        zoom_start=14, tiles="OpenStreetMap",
    )

    # Background edges
    bg = folium.FeatureGroup(name="Network", show=True)
    for u, nbrs in EDGES_KM.items():
        if u not in NODES: continue
        lu, lo = NODES[u][1], NODES[u][0]
        for v, _ in nbrs:
            if v not in NODES: continue
            lv, lov = NODES[v][1], NODES[v][0]
            folium.PolyLine([[lu,lo],[lv,lov]], color="#BBB", weight=2, opacity=.4).add_to(bg)
    bg.add_to(m)

    # Background nodes
    bn = folium.FeatureGroup(name="All SCATS Sites", show=True)
    for sid, (lon, lat) in NODES.items():
        site_label = SITE_NAMES.get(sid, sid)
        folium.CircleMarker(
            [lat, lon], radius=4, color="#999", fill=True,
            fill_color="#CCC", fill_opacity=.7,
            tooltip=f"{sid} — {site_label}",
        ).add_to(bn)
    bn.add_to(m)

    # Route layers
    for rank, pinfo in enumerate(top_k):
        path = pinfo["path"]
        cost_min = pinfo["cost"] / 60.0
        color = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
        label = ROUTE_NAMES[rank] if rank < len(ROUTE_NAMES) else f"Rank {rank+1}"

        fg = folium.FeatureGroup(name=f"{label} ({cost_min:.1f} min)", show=(rank == 0))
        coords = [[NODES[n][1], NODES[n][0]] for n in path if n in NODES]

        named_route = [SITE_NAMES.get(n, n) for n in path]
        popup_html = (
            f"<b style='color:{color}'>{label}</b><br>"
            f"<b>Route:</b> {' &rarr; '.join(named_route)}<br>"
            f"<b>Time:</b> {cost_min:.2f} min<br>"
            f"<b>Flow:</b> {predicted_flow:.0f} veh/15min<br>"
            f"<b>Model:</b> {model_name}"
        )
        folium.PolyLine(
            coords, color=color, weight=5 if rank==0 else 4, opacity=.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{label}: {cost_min:.1f} min",
        ).add_to(fg)

        # Waypoint markers
        for i, nid in enumerate(path):
            if nid not in NODES: continue
            lon, lat = NODES[nid]
            site_label = SITE_NAMES.get(nid, nid)
            if rank == 0:
                if i == 0:
                    icon = folium.Icon(color="green", icon="play", prefix="fa")
                elif i == len(path)-1:
                    icon = folium.Icon(color="red", icon="flag-checkered", prefix="fa")
                else:
                    icon = folium.Icon(color="blue", icon="circle", prefix="fa")
                folium.Marker([lat, lon], icon=icon, tooltip=f"{nid} — {site_label}").add_to(fg)
            else:
                folium.CircleMarker(
                    [lat, lon], radius=5, color=color, fill=True,
                    fill_color=color, fill_opacity=.9, tooltip=f"{nid} — {site_label}",
                ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    return m._repr_html_()



# ═══════════════════════════════════════════════════════════════════════
#  Flask route
# ═══════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    origin     = request.args.get("origin", SCATS_LIST[0])
    dest       = request.args.get("dest",   SCATS_LIST[-1])
    model_name = request.args.get("model",  "GRU")

    if origin == dest:
        model = MODELS.get(model_name)
        flow = predict_flow(model, model_name) if model else 0
        
        route_cards = [{
            "label": "Arrival",
            "color": "#3cb44b",
            "time": "0.00",
            "path_str": SITE_NAMES.get(origin, origin) + " (Already at destination)"
        }]
        
        # Use a dummy path with the same node twice to render the marker on the map
        dummy_top_k = [{"cost": 0.0, "path": [origin, origin]}]
        map_html = build_folium_map(dummy_top_k, flow, model_name, origin, dest)

        return render_template_string(PAGE_HTML,
            scats_list=SCATS_LIST, models=list(MODELS.keys()),
            names=SITE_NAMES,
            origin=origin, dest=dest, model_name=model_name,
            routes=route_cards, flow=flow, map_html=map_html)

    model = MODELS.get(model_name)
    if model is None:
        return render_template_string(PAGE_HTML,
            scats_list=SCATS_LIST, models=list(MODELS.keys()),
            names=SITE_NAMES,
            origin=origin, dest=dest, model_name=model_name,
            routes=[], flow=0, map_html=f"<p>Model '{model_name}' not available.</p>")

    # Predict flow (Part B)
    flow = predict_flow(model, model_name)

    # Build time-weighted edges (Part A+B bridge)
    time_edges = build_time_weighted_edges(EDGES_KM, flow)

    # A* single best path
    single_path, single_cost, nodes_created = run_parta_search(
        NODES, time_edges, origin, dest)

    # Yen's Top-5 shortest paths
    top_k = yens_k_shortest_paths(NODES, time_edges, origin, dest, K=5)


    route_cards = []
    for i, p in enumerate(top_k):
        t = p["cost"] / 60.0

        named_path = [SITE_NAMES.get(nid, nid) for nid in p["path"]]
        path_display = " \u2192 ".join(named_path)
        route_cards.append({
            "label": ROUTE_NAMES[i] if i < len(ROUTE_NAMES) else f"Rank {i+1}",
            "color": ROUTE_COLORS[i % len(ROUTE_COLORS)],
            "time":  f"{t:.2f}",
            "path_str": path_display,
        })

    map_html = build_folium_map(top_k, flow, model_name, origin, dest)

    return render_template_string(PAGE_HTML,
        scats_list=SCATS_LIST, models=list(MODELS.keys()),
        names=SITE_NAMES,
        origin=origin, dest=dest, model_name=model_name,
        routes=route_cards, flow=flow, map_html=map_html)


def kill_existing_server(port):
    """Find and kill any process using the specified port on Windows."""
    try:
        result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        lines = result.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and f":{port}" in parts[1]:
                pid = parts[-1]
                if pid != "0":
                    pids.add(pid)

        for pid in pids:
            print(f"Stopping existing server (PID: {pid}) on port {port}...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"Warning: Could not check/kill existing server on port {port}: {e}")

if __name__ == "__main__":
    kill_existing_server(5000)
    print("Starting web server at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
