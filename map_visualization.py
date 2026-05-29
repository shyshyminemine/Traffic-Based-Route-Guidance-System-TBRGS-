"""
map_visualization.py
====================
Generates an interactive OpenStreetMap-based HTML visualization of the
Top-K predicted routes across the Boroondara SCATS network.

Output: route_map.html
"""

import os
import sys
import warnings

# Ensure imports and file paths resolve relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import numpy as np
import folium
from folium import plugins
import joblib
from tensorflow.keras.models import load_model

from build_real_network import build_real_boroondara_graph
from data_preprocessing import prepare_dataloaders
from route_guidance import yens_k_shortest_paths, calculate_travel_time

warnings.filterwarnings("ignore")

# Colour palette for routes
ROUTE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
ROUTE_LABELS = ["Rank 1 (Best)", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]


def predict_flow(model, model_name, scaler, sample_sequence):
    """Return the predicted traffic flow (vehicles / 15-min) in original scale."""
    if model_name == "Random Forest":
        X_rf = sample_sequence.reshape(sample_sequence.shape[0], -1)
        pred_scaled = model.predict(X_rf)[0]
    else:
        pred_scaled = model.predict(sample_sequence, verbose=0)[0][0]
    flow = scaler.inverse_transform([[pred_scaled]])[0][0]
    return max(0.0, flow)


def build_updated_edges(edges_km, predicted_flow):
    """Convert distance-based edges into time-based edges using predicted flow."""
    updated = {}
    for u, nbrs in edges_km.items():
        updated[u] = []
        for v, dist_km in nbrs:
            tt = calculate_travel_time(predicted_flow, dist_km)
            updated[u].append((v, tt))
    return updated


def create_route_map(
    nodes,
    edges_km,
    top_k_paths,
    predicted_flow,
    model_name,
    origin,
    destination,
    output_file="outputs/route_map.html",
):
    """
    Build a Folium interactive map with:
      - All SCATS nodes as circle markers (grey background layer)
      - All network edges as thin grey lines (background layer)
      - Top-K routes drawn in distinct colours with popup info
    """


    lats = [pos[1] for pos in nodes.values()]
    lons = [pos[0] for pos in nodes.values()]
    centre_lat = sum(lats) / len(lats)
    centre_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Background: all edges
    bg_edges_group = folium.FeatureGroup(name="Background Network", show=True)
    for u, nbrs in edges_km.items():
        if u not in nodes:
            continue
        lon_u, lat_u = nodes[u]
        for v, dist_km in nbrs:
            if v not in nodes:
                continue
            lon_v, lat_v = nodes[v]
            folium.PolyLine(
                locations=[[lat_u, lon_u], [lat_v, lon_v]],
                color="#BBBBBB",
                weight=2,
                opacity=0.5,
            ).add_to(bg_edges_group)
    bg_edges_group.add_to(m)

    # Background: all SCATS nodes
    bg_nodes_group = folium.FeatureGroup(name="All SCATS Sites", show=True)
    for sid, (lon, lat) in nodes.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#999999",
            fill=True,
            fill_color="#CCCCCC",
            fill_opacity=0.8,
            popup=folium.Popup(f"<b>SCATS {sid}</b><br>Lat: {lat:.5f}<br>Lon: {lon:.5f}", max_width=200),
            tooltip=f"SCATS {sid}",
        ).add_to(bg_nodes_group)
    bg_nodes_group.add_to(m)

    # Route layers: one per Top-K path
    for rank, path_info in enumerate(top_k_paths):
        path = path_info["path"]
        cost_sec = path_info["cost"]
        cost_min = cost_sec / 60.0
        color = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
        label = ROUTE_LABELS[rank] if rank < len(ROUTE_LABELS) else f"Rank {rank+1}"

        route_group = folium.FeatureGroup(name=f"{label}  ({cost_min:.1f} min)", show=(rank == 0))


        coords = []
        for nid in path:
            if nid in nodes:
                lon, lat = nodes[nid]
                coords.append([lat, lon])

        popup_html = (
            f"<div style='font-family:Arial;font-size:13px;'>"
            f"<b style='color:{color}'>{label}</b><br>"
            f"<b>Route:</b> {' &rarr; '.join(path)}<br>"
            f"<b>Est. Travel Time:</b> {cost_min:.2f} min<br>"
            f"<b>Predicted Flow:</b> {predicted_flow:.0f} veh/15min<br>"
            f"<b>Model:</b> {model_name}"
            f"</div>"
        )

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=5 if rank == 0 else 4,
            opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{label}: {cost_min:.1f} min",
        ).add_to(route_group)

        # Mark waypoint nodes along this route
        for idx, nid in enumerate(path):
            if nid not in nodes:
                continue
            lon, lat = nodes[nid]

            if idx == 0:
                icon = folium.Icon(color="green", icon="play", prefix="fa")
                label_text = f"START: SCATS {nid}"
            elif idx == len(path) - 1:
                icon = folium.Icon(color="red", icon="flag-checkered", prefix="fa")
                label_text = f"END: SCATS {nid}"
            else:
                icon = folium.Icon(color="blue", icon="circle", prefix="fa")
                label_text = f"Via: SCATS {nid}"

            # Only add full markers for Rank 1 to keep the map clean
            if rank == 0:
                folium.Marker(
                    location=[lat, lon],
                    icon=icon,
                    popup=label_text,
                    tooltip=label_text,
                ).add_to(route_group)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=f"SCATS {nid}",
                ).add_to(route_group)

        route_group.add_to(m)


    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)

    # Legend (custom HTML)
    legend_items = ""
    for rank, path_info in enumerate(top_k_paths):
        color = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
        label = ROUTE_LABELS[rank] if rank < len(ROUTE_LABELS) else f"Rank {rank+1}"
        t = path_info["cost"] / 60.0
        legend_items += (
            f'<li><span style="background:{color};width:14px;height:14px;'
            f'display:inline-block;margin-right:6px;border-radius:3px;"></span>'
            f'{label} &mdash; {t:.1f} min</li>'
        )

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
        background:white;padding:12px 16px;border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,.3);font-family:Arial;font-size:13px;
        max-width:280px;">
      <b>Route Legend</b><br>
      <span style="font-size:11px;color:#666;">
        Model: {model_name} &nbsp;|&nbsp; Flow: {predicted_flow:.0f} veh/15min
      </span>
      <ul style="list-style:none;padding-left:0;margin-top:6px;">
        {legend_items}
      </ul>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_file)
    print(f"Interactive map saved to: {os.path.abspath(output_file)}")
    return m


# ════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Build real road network ──────────────────────────────────────
    print("1. Building real Boroondara SCATS network...")
    nodes, edges_km, site_names = build_real_boroondara_graph("data/Scats Data October 2006.xls", K=3)
    scats_list = list(nodes.keys())

    # ── 2. Load ML model + data ─────────────────────────────────────────
    print("2. Loading GRU model and preparing data...")
    model_name = "GRU"
    model = load_model("models/gru_model.keras")
    _, _, _, scaler, arrays = prepare_dataloaders(
        "data/Scats Data October 2006.xls", window_size=4
    )
    X_test = arrays[4]
    sample_seq = X_test[0:1]

    # ── 3. Predict traffic + update edge costs ──────────────────────────
    print("3. Predicting traffic flow...")
    predicted_flow = predict_flow(model, model_name, scaler, sample_seq)
    print(f"   => Predicted flow: {predicted_flow:.1f} vehicles / 15-min")
    updated_edges = build_updated_edges(edges_km, predicted_flow)

    # ── 4. Choose origin / destination ──────────────────────────────────
    origin = scats_list[0]
    destination = scats_list[-1]
    print(f"4. Finding Top-5 routes from SCATS {origin} to SCATS {destination}...")

    top_k = yens_k_shortest_paths(nodes, updated_edges, origin, destination, K=5)

    if not top_k:
        print("   No routes found. Try different origin/destination.")
        return

    for i, p in enumerate(top_k):
        route_str = " -> ".join(p["path"])
        print(f"   Rank {i+1}: {p['cost']/60:.2f} min | {route_str}")

    # ── 5. Generate interactive map ─────────────────────────────────────
    print("5. Generating interactive Folium map...")
    create_route_map(
        nodes,
        edges_km,
        top_k,
        predicted_flow,
        model_name,
        origin,
        destination,
        output_file="outputs/route_map.html",
    )
    print("Done!")


if __name__ == "__main__":
    main()
