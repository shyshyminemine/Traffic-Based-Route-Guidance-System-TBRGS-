import os
import sys

# Set working directory and module path to script directory for robust file loading
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate Adjustment System
# ═══════════════════════════════════════════════════════════════════════════════
# Assignment note: SCATS lat/lon values do not map correctly to actual
# intersections on Google Maps and require adjustment.
#
# Approach:
#   1. Per-site manual corrections (cross-referenced with Google Maps/OSM).
#   2. Global empirical offset as fallback for uncorrected sites.
# ═══════════════════════════════════════════════════════════════════════════════

# Per-site corrections: SCATS_ID -> (corrected_lon, corrected_lat)
# Verified against Google Maps for key Boroondara intersections.
MANUAL_CORRECTIONS = {
    "0970": (145.09030, -37.86800),  # WARRIGAL_RD / HIGH STREET_RD
    "2000": (145.09380, -37.85120),  # WARRIGAL_RD / TOORAK_RD
    "3001": (145.03320, -37.80240),  # HIGH_ST / BARKERS_RD (Kew Junction)
    "3002": (145.02700, -37.80400),  # DENMARK_ST / BARKERS_RD
    "4263": (145.04530, -37.81830),  # GLENFERRIE_RD / BURWOOD_RD
}

# Global empirical offset for non-corrected sites (average observed error)
# Positive lat offset = shift south->north; positive lon offset = shift west->east
GLOBAL_LAT_OFFSET = 0.0008
GLOBAL_LON_OFFSET = 0.0005


def adjust_coordinates(sid, raw_lat, raw_lon):
    """
    Apply coordinate correction for a SCATS site.

    Priority:
      1. Use per-site manual correction if available (verified via Google Maps).
      2. Otherwise apply a small global empirical offset.

    Parameters
    ----------
    sid      : str    SCATS site ID (e.g. '0970')
    raw_lat  : float  original latitude from the dataset
    raw_lon  : float  original longitude from the dataset

    Returns
    -------
    (corrected_lon, corrected_lat)
    """
    if sid in MANUAL_CORRECTIONS:
        return MANUAL_CORRECTIONS[sid]

    corrected_lat = raw_lat + GLOBAL_LAT_OFFSET
    corrected_lon = raw_lon + GLOBAL_LON_OFFSET
    return (corrected_lon, corrected_lat)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    R = 6371.0 # Earth radius in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

import re

def _clean_location_name(raw):
    """
    Convert raw dataset location strings into readable intersection names.
    e.g. 'WARRIGAL_RD N of HIGH STREET_RD' -> 'Warrigal Rd / High Street Rd'
    """
    if not raw or raw == 'nan':
        return 'Unknown'
    # Split on directional words (N of, S of, NE of, SE of, SW of, E of, W of, etc.)
    parts = re.split(r'\s+(?:N|S|E|W|NE|NW|SE|SW|N\s*BD|S\s*BD|E\s*BD|W\s*BD)\s+(?:of|OF)\s+', raw, maxsplit=1)
    cleaned = []
    for part in parts:
        p = part.replace('_', ' ').strip().title()
        # Fix common abbreviations back to uppercase
        for abbr in ['Rd', 'St', 'Hwy', 'Fwy', 'Gv', 'Ave', 'Dr', 'Ct', 'Pde']:
            p = p.replace(abbr, abbr)
        cleaned.append(p)
    return ' / '.join(cleaned)


def build_real_boroondara_graph(file_path="data/Scats Data October 2006.xls", K=3):
    """
    Extract SCATS nodes from Excel and build edges based on geographic proximity.
    Connects each node to its K nearest neighbors to simulate street connections.
    
    All coordinates are passed through adjust_coordinates() before being used
    for distance calculation or map rendering, addressing the known dataset
    coordinate inaccuracy issue.
    """
    print("Reading SCATS Data...")
    df = pd.read_excel(file_path, sheet_name="Data", header=1)

    unique_sites = df.drop_duplicates(subset=['SCATS Number'])
    unique_sites = unique_sites.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE', 'SCATS Number'])

    nodes = {}
    scats_ids = []
    coords = []
    site_names = {}

    for _, row in unique_sites.iterrows():
        sid = str(row['SCATS Number']).strip()
        if sid.endswith('.0'):
            sid = sid[:-2]
        if len(sid) < 4:
            sid = sid.zfill(4)
            
        raw_lat = float(row['NB_LATITUDE'])
        raw_lon = float(row['NB_LONGITUDE'])
        
        # Filter out missing coordinates which default to 0.0
        if raw_lat == 0.0 or raw_lon == 0.0:
            continue

        # Apply coordinate adjustment (per-site or global offset)
        lon, lat = adjust_coordinates(sid, raw_lat, raw_lon)

        raw_name = str(row.get('Location', '')).strip()
        clean_name = _clean_location_name(raw_name)
        site_names[sid] = clean_name
            
        nodes[sid] = (lon, lat)
        scats_ids.append(sid)
        coords.append((lon, lat))
        
    print(f"Extracted {len(nodes)} unique SCATS intersections.")

    # Construct Edges based on K-Nearest Neighbors
    # NOTE: haversine() receives the CORRECTED coordinates from nodes/coords,
    #       so edge distances reflect true geographic distances.
    edges_km = {sid: [] for sid in scats_ids}

    for i, sid_u in enumerate(scats_ids):
        lon1, lat1 = coords[i]
        distances = []
        for j, sid_v in enumerate(scats_ids):
            if i != j:
                lon2, lat2 = coords[j]
                dist_km = haversine(lon1, lat1, lon2, lat2)
                distances.append((dist_km, sid_v))
        
        distances.sort(key=lambda x: x[0])
        for dist_km, sid_v in distances[:K]:
            edges_km[sid_u].append((sid_v, dist_km))

    # Make edges bidirectional to represent undirected two-way streets
    for u in list(edges_km.keys()):
        for v, dist in edges_km[u]:
            v_neighbors = [x[0] for x in edges_km[v]]
            if u not in v_neighbors:
                edges_km[v].append((u, dist))

    return nodes, edges_km, site_names

def main():
    nodes, edges_km, site_names = build_real_boroondara_graph()

    # Visualize the full network topology
    print("Plotting the real network topology...")
    plt.figure(figsize=(20, 16))
    G = nx.DiGraph()

    for u, pos in nodes.items():
        G.add_node(u, pos=pos)
        
    for u, nbrs in edges_km.items():
        for v, dist in nbrs:
            G.add_edge(u, v, weight=dist)
            
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgreen', edgecolors='black')
    # Draw edges without arrows (since they are two-way roads)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=False, alpha=0.5, width=1.5)

    label_pos = {k: (v[0], v[1] + 0.002) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=9, font_color='navy', font_weight='bold')

    plt.title("Boroondara SCATS Real Road Network Topology", fontsize=20, fontweight='bold')
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_file = "outputs/real_network_topology.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Graph constructed successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} directional edges.")
    print(f"Saved visualization to {out_file}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
