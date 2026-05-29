"""
route_guidance.py
=================
Phase 3 Integration Module: Traffic-Based Route Guidance System (TBRGS)

The A* search algorithm below is directly ported and modified from Assignment 2A
(search.py). The key modification for Part B integration is the heuristic function:

  Original Part A heuristic:  h(n) = Euclidean distance in km  (for distance-cost graphs)
  Modified Part B heuristic:  h(n) = Euclidean distance / 60 km/h * 3600 s
                                    = straight-line distance * 60 seconds/km

This modification ensures ADMISSIBILITY: since no road segment can be traversed
faster than 60 km/h (the speed cap in calculate_travel_time), the heuristic
never overestimates the true remaining travel time — guaranteeing A* optimality.

End-to-end pipeline:
    1. Build real SCATS road network (nodes + distance edges)
    2. Predict traffic flow with a trained ML model (LSTM / GRU / RF)
    3. Convert predicted flow -> travel-time edge costs (fundamental diagram)
    4. Run modified A* on the time-weighted graph (single best path)
    5. Yen's K-shortest paths wraps the modified A* to return Top-K routes
"""

import math
import heapq
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: node sort key (ported from Part A search.py — node_sort_key)
# ═══════════════════════════════════════════════════════════════════════════════

def _node_sort_key(node_id):
    """Numeric sort for integer IDs, lexicographic fallback. (from search.py)"""
    try:
        return (int(node_id), '')
    except ValueError:
        return (math.inf, node_id)


def _get_neighbors_sorted(edges, state):
    """Neighbours sorted ascending by node id. (from search.py)"""
    return sorted(edges.get(state, []), key=lambda x: _node_sort_key(x[0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic functions
# ═══════════════════════════════════════════════════════════════════════════════

def _euclidean_km(nodes, node, goals):
    """
    Straight-line distance in km to nearest goal. (from Part A search.py euclidean)
    Node positions are (longitude, latitude); distance is approximate km via
    coordinate differences scaled by the Haversine constant.
    """
    nx_, ny = nodes[node]
    return min(
        math.sqrt((nx_ - nodes[g][0]) ** 2 + (ny - nodes[g][1]) ** 2)
        for g in goals
    )


def _time_heuristic(nodes, node, goals):
    """
    MODIFIED Part B heuristic — admissible for time-weighted graphs.

    Converts Euclidean straight-line distance (degrees, approx km in this
    coordinate space) to theoretical minimum travel time in seconds by
    dividing by the maximum allowed speed (60 km/h).

        h(n) = euclidean_distance(n, nearest_goal) / 60 km/h * 3600 s/h

    Admissibility proof:
      - calculate_travel_time() caps speed at 60 km/h.
      - Straight-line distance <= actual road distance for any segment.
      - Therefore h(n) <= true remaining travel time. ✓
    """
    dist = _euclidean_km(nodes, node, goals)
    MAX_SPEED_KMH = 60.0
    return (dist / MAX_SPEED_KMH) * 3600.0   # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Modified A* (ported from Part A search.py astar(), heuristic changed)
# ═══════════════════════════════════════════════════════════════════════════════

def _astar_timed(nodes, edges, origin, destinations):
    """
    A* Search — ported from Assignment 2A (search.py :: astar) and modified
    to use the time-consistent heuristic _time_heuristic() instead of the
    original Euclidean-distance heuristic.

    All structural logic (heap tuple, ancestor check, counter tie-breaking,
    neighbour sorting) is identical to the Part A implementation.

    Parameters
    ----------
    nodes        : dict  str -> (float, float)   coordinate map
    edges        : dict  str -> [(str, float)]   adjacency list (time in seconds)
    origin       : str   start node id
    destinations : list  goal node ids

    Returns
    -------
    (goal, nodes_created, path)
        goal          : str or None
        nodes_created : int
        path          : list of node ids
    """
    goal_set = set(destinations)
    created = 1          # root node is created immediately
    counter = [0]        # monotonic stamp for tie-breaking

    h0 = _time_heuristic(nodes, origin, destinations)

    # Heap tuple: (f_score, insertion_order, g_score, state, path)
    # Identical structure to Part A astar()
    heap = [(h0, counter[0], 0.0, origin, [origin])]
    counter[0] += 1
    
    # Closed set to prevent exponential explosion
    best_g = {}

    while heap:
        _, _, g, state, path = heapq.heappop(heap)

        if state in goal_set:
            return state, created, path
            
        if state in best_g and g > best_g[state]:
            continue
        best_g[state] = g

        # Neighbours sorted ascending
        for nbr, cost in _get_neighbors_sorted(edges, state):
            if nbr not in path:        # ancestor check — same as Part A
                ng = g + cost
                # Only explore if we found a strictly better path to nbr
                if nbr not in best_g or ng < best_g[nbr]:
                    nh = _time_heuristic(nodes, nbr, destinations)
                    created += 1
                    heapq.heappush(heap, (ng + nh, counter[0], ng, nbr, path + [nbr]))
                    counter[0] += 1

    return None, created, []


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Traffic Flow to Travel Time Conversion (Fundamental Diagram)
# Reference: "Traffic Flow to Travel Time Conversion v1.0.pdf"
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_travel_time(traffic_flow_15min, distance_km):
    """
    Convert predicted 15-min traffic flow and segment distance into travel time.

    Uses the fundamental diagram quadratic:
        flow = -1.4648375 * speed^2 + 93.75 * speed
    Solves for speed (free-flow / green curve branch), then time = distance / speed.
    Adds a 30-second fixed delay per controlled intersection.

    Returns
    -------
    float  Total travel time in seconds.
    """
    flow_hr = min(traffic_flow_15min * 4, 1500.0)

    if flow_hr <= 351:
        speed_kmh = 60.0
    else:
        a, b, c = -1.4648375, 93.75, -flow_hr
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            speed_kmh = 32.0
        else:
            speed_kmh = (-b - math.sqrt(discriminant)) / (2 * a)

    speed_kmh = max(32.0, min(60.0, speed_kmh))
    return (distance_km / speed_kmh) * 3600.0 + 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# Part A + Part B Bridge: build time-weighted edge graph
# ═══════════════════════════════════════════════════════════════════════════════

def build_time_weighted_edges(edges_km, predicted_flow):
    """
    Replace distance-based edge costs with ML-predicted travel-time costs.

    The resulting dict has the SAME adjacency-list structure as the `edges`
    dict produced by search.py's parse_file(), so _astar_timed() can consume
    it directly.

    Parameters
    ----------
    edges_km       : dict  {node: [(neighbour, km), ...]}
    predicted_flow : float traffic flow (vehicles / 15-min) from ML model

    Returns
    -------
    dict  {node: [(neighbour, seconds), ...]}
    """
    return {
        u: [(v, calculate_travel_time(predicted_flow, dist_km))
            for v, dist_km in nbrs]
        for u, nbrs in edges_km.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-path search entry point
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_ALGORITHMS = {
    'AS':   _astar_timed,   # Modified A* (time heuristic) — primary algorithm
}


def run_parta_search(nodes, edges, origin, destination, algorithm='AS'):
    """
    Run the modified A* algorithm from origin to destination.
    (algorithm parameter kept for GUI compatibility; 'AS' is the supported mode)

    Returns
    -------
    (path, cost_seconds, nodes_created)
    """
    goal, nodes_created, path = _astar_timed(nodes, edges, origin, [destination])

    if not path:
        return [], math.inf, nodes_created

    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for neighbour, c in edges.get(u, []):
            if neighbour == v:
                cost += c
                break

    return path, cost, nodes_created


# ═══════════════════════════════════════════════════════════════════════════════
# Yen's K-Shortest Paths (inner engine: modified A*)
# ═══════════════════════════════════════════════════════════════════════════════

def _astar_timed_with_exclusions(nodes, edges, origin, destination,
                                  removed_nodes, removed_edges):
    """
    Run _astar_timed on a subgraph that excludes specific nodes and edges.
    This is Yen's algorithm inner engine.

    Returns (cost_seconds, path).
    """
    filtered_edges = {}
    for u, nbrs in edges.items():
        if u in removed_nodes:
            continue
        filtered_edges[u] = [
            (v, c) for v, c in nbrs
            if v not in removed_nodes and (u, v) not in removed_edges
        ]

    if origin not in filtered_edges or destination not in filtered_edges:
        return math.inf, []

    goal, _, path = _astar_timed(nodes, filtered_edges, origin, [destination])

    if not path:
        return math.inf, []

    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for nbr, c in filtered_edges.get(u, []):
            if nbr == v:
                cost += c
                break

    return cost, path


def yens_k_shortest_paths(nodes, edges, origin, destination, K=5):
    """
    Yen's K-Shortest Paths algorithm.
    Inner engine: modified A* (_astar_timed_with_exclusions).

    Parameters
    ----------
    nodes, edges : real SCATS graph with time-weighted edges (seconds)
    K            : number of shortest paths to find (default 5)

    Returns
    -------
    list of {'path': [...], 'cost': float}  sorted by ascending cost
    """
    cost, path = _astar_timed_with_exclusions(
        nodes, edges, origin, destination, set(), set())
    if not path:
        return []

    A = [{'cost': cost, 'path': path}]
    B = []

    for k in range(1, K):
        prev_path = A[k - 1]['path']

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            removed_edges = set()
            for p in A:
                if len(p['path']) > i and p['path'][:i + 1] == root_path:
                    removed_edges.add((p['path'][i], p['path'][i + 1]))

            removed_nodes = set(root_path[:-1])

            spur_cost, spur_path = _astar_timed_with_exclusions(
                nodes, edges, spur_node, destination,
                removed_nodes, removed_edges
            )

            if spur_path:
                total_path = root_path[:-1] + spur_path

                total_cost = 0.0
                for j in range(len(total_path) - 1):
                    u, v = total_path[j], total_path[j + 1]
                    for nbr, c in edges.get(u, []):
                        if nbr == v:
                            total_cost += c
                            break

                candidate = {'cost': total_cost, 'path': total_path}
                if candidate not in B and not any(
                        a['path'] == total_path for a in A):
                    heapq.heappush(B, (total_cost, k, candidate))

        if not B:
            break

        _, _, next_best = heapq.heappop(B)
        A.append(next_best)

    return A


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline Demo
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import warnings
    warnings.filterwarnings("ignore")

    from tensorflow.keras.models import load_model
    from build_real_network import build_real_boroondara_graph
    from data_preprocessing import prepare_dataloaders

    print("1. Building real Boroondara SCATS network...")
    nodes, edges_km, site_names = build_real_boroondara_graph("data/Scats Data October 2006.xls", K=3)
    scats_list = list(nodes.keys())
    origin, destination = scats_list[0], scats_list[-1]
    print(f"   => {len(nodes)} nodes | Testing: {origin} -> {destination}")

    print("\n2. Loading GRU model...")
    model = load_model("models/gru_model.keras")
    _, _, _, scaler, arrays = prepare_dataloaders(
        "data/Scats Data October 2006.xls", window_size=4)
    sample_seq = arrays[4][0:1]

    pred_scaled = model.predict(sample_seq, verbose=0)[0][0]
    predicted_flow = max(0.0, scaler.inverse_transform([[pred_scaled]])[0][0])
    print(f"   => Predicted flow: {predicted_flow:.1f} vehicles / 15-min")

    print("\n3. Converting flow to travel-time edge costs...")
    time_edges = build_time_weighted_edges(edges_km, predicted_flow)

    print("\n4. Running modified A* (time heuristic, ported from Part A):")
    path, cost, nc = run_parta_search(nodes, time_edges, origin, destination)
    print(f"   [A*] {cost/60:.2f} min | {nc} nodes created")
    print(f"   Path: {' -> '.join(path)}")

    print("\n5. Yen's Top-5 (inner engine = modified A*):")
    top_k = yens_k_shortest_paths(nodes, time_edges, origin, destination, K=5)
    print("=" * 60)
    print("         TOP-K RECOMMENDED ROUTES (TBRGS)")
    print("=" * 60)
    for i, p in enumerate(top_k):
        print(f"  Rank {i+1}: {p['cost']/60:.2f} min | {' -> '.join(p['path'])}")


if __name__ == '__main__':
    main()
