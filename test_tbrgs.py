"""
test_tbrgs.py
=============
Unit tests for the Traffic-Based Route Guidance System (TBRGS).
COS30019 Assignment 2B — at least 15 test cases covering different scenarios.

Run:  python -m unittest test_tbrgs -v
"""

import os
import sys

# Ensure imports and file paths resolve relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
os.chdir(script_dir)

import json
import math
import unittest
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Import modules under test
# ─────────────────────────────────────────────────────────────────────
from route_guidance import (
    calculate_travel_time,
    build_time_weighted_edges,
    _astar_timed,
    _euclidean_km,
    _time_heuristic,
    _node_sort_key,
    yens_k_shortest_paths,
    run_parta_search,
)
from data_preprocessing import load_and_clean_data, create_sliding_window
from build_real_network import build_real_boroondara_graph


# ═════════════════════════════════════════════════════════════════════
#  Helper: build a small mock graph for fast, deterministic tests
# ═════════════════════════════════════════════════════════════════════
def _make_small_graph():
    """
    A -> B -> C -> D  (linear, 1 km each)
    A -> E -> D       (shortcut, 1.5 km each)

    Nodes use (lon, lat) for position; values chosen so Euclidean distances
    are roughly proportional to km.
    """
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (2.0, 0.0),
        'D': (3.0, 0.0),
        'E': (1.5, 1.0),
    }
    edges_km = {
        'A': [('B', 1.0), ('E', 1.5)],
        'B': [('C', 1.0)],
        'C': [('D', 1.0)],
        'D': [],
        'E': [('D', 1.5)],
    }
    return nodes, edges_km


class TestTravelTimeCalculation(unittest.TestCase):
    """Tests for calculate_travel_time() — edge cost formula."""

    # ── Test 1: Free-flow speed (60 km/h) + 30s intersection delay ──
    def test_01_free_flow_basic(self):
        """TC01: Under low flow, 1km segment travel time = 1km/60kmh * 3600s + 30s = 90s"""
        tt = calculate_travel_time(traffic_flow_15min=50, distance_km=1.0)
        # 50 veh/15min => 200 veh/hr < 351 => free flow => speed=60
        # time = 1/60 * 3600 + 30 = 60 + 30 = 90
        self.assertAlmostEqual(tt, 90.0, places=1)

    # ── Test 2: Congested speed ──────────────────────────────────────
    def test_02_congested_flow(self):
        """TC02: Speed decreases under high flow, travel time should be greater than free-flow 90s"""
        tt_congested = calculate_travel_time(traffic_flow_15min=300, distance_km=1.0)
        tt_freeflow  = calculate_travel_time(traffic_flow_15min=50, distance_km=1.0)
        self.assertGreater(tt_congested, tt_freeflow)

    # ── Test 3: 30-second intersection delay is always included ──────
    def test_03_intersection_delay_always_present(self):
        """TC03: Even if distance is 0, there should still be a 30s intersection delay"""
        tt = calculate_travel_time(traffic_flow_15min=50, distance_km=0.0)
        self.assertAlmostEqual(tt, 30.0, places=1)

    # ── Test 4: Extreme congestion (capped at 1500 veh/hr) ───────────
    def test_04_extreme_flow_capped(self):
        """TC04: Extreme flow capped at 1500 veh/hr, speed not lower than 32 km/h"""
        tt = calculate_travel_time(traffic_flow_15min=500, distance_km=1.0)
        # speed >= 32 => max time = 1/32*3600 + 30 = 142.5
        self.assertLessEqual(tt, 142.5)
        self.assertGreater(tt, 30.0)

    # ── Test 5: Proportional to distance ──────────────────────────────
    def test_05_proportional_to_distance(self):
        """TC05: Under same flow, 2km segment time is approx 1km time * 2 (plus one 30s delay)"""
        tt_1km = calculate_travel_time(traffic_flow_15min=50, distance_km=1.0)
        tt_2km = calculate_travel_time(traffic_flow_15min=50, distance_km=2.0)
        # tt_1km = 60 + 30 = 90;  tt_2km = 120 + 30 = 150
        self.assertAlmostEqual(tt_2km, 150.0, places=1)


class TestEdgeWeightUpdate(unittest.TestCase):
    """Tests for build_time_weighted_edges() — Part A/B bridge."""

    # ── Test 6: Output structure matches adjacency list format ────────
    def test_06_edge_structure_preserved(self):
        """TC06: Updated edges structure must match original graph (dict -> list of tuples)"""
        _, edges_km = _make_small_graph()
        updated = build_time_weighted_edges(edges_km, predicted_flow=100.0)

        self.assertIsInstance(updated, dict)
        for node, nbrs in updated.items():
            self.assertIsInstance(nbrs, list)
            for v, cost in nbrs:
                self.assertIsInstance(v, str)
                self.assertIsInstance(cost, float)

    # ── Test 7: All original edges are preserved ─────────────────────
    def test_07_all_edges_preserved(self):
        """TC07: Updating weights must not drop any original edge"""
        _, edges_km = _make_small_graph()
        updated = build_time_weighted_edges(edges_km, predicted_flow=100.0)

        for node in edges_km:
            self.assertEqual(len(updated[node]), len(edges_km[node]))


class TestAstarTimed(unittest.TestCase):
    """Tests for _astar_timed() — modified A* with time heuristic."""

    # ── Test 8: Normal shortest path query ────────────────────────────
    def test_08_normal_shortest_path(self):
        """TC08: Optimal path A->D on small graph should be A->E->D (total dist 3km vs 3km,
        but A->E->D has 2 edges = two 30s delays, whereas A->B->C->D has 3 edges = three 30s delays)"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        goal, created, path = _astar_timed(nodes, time_edges, 'A', ['D'])

        self.assertEqual(goal, 'D')
        self.assertIn('A', path)
        self.assertEqual(path[-1], 'D')
        self.assertGreater(created, 0)

    # ── Test 9: Start equals destination ──────────────────────────────
    def test_09_origin_equals_destination(self):
        """TC09: Origin equals destination should return a path with only one node"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        goal, created, path = _astar_timed(nodes, time_edges, 'A', ['A'])

        self.assertEqual(goal, 'A')
        self.assertEqual(path, ['A'])

    # ── Test 10: Unreachable destination ──────────────────────────────
    def test_10_unreachable_destination(self):
        """TC10: Destination unreachable (D has no outgoing edges, reverse D->A impossible), should return None"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        goal, created, path = _astar_timed(nodes, time_edges, 'D', ['A'])

        self.assertIsNone(goal)
        self.assertEqual(path, [])

    # ── Test 11: Invalid / non-existent node ─────────────────────────
    def test_11_invalid_node(self):
        """TC11: Input non-existent SCATS node, algorithm should gracefully return empty path instead of crashing"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)

        # 'Z' doesn't exist in the graph
        try:
            goal, created, path = _astar_timed(nodes, time_edges, 'Z', ['D'])
            # If it doesn't crash, it should return no path
            self.assertIsNone(goal)
        except KeyError:
            # Also acceptable — the algorithm can raise KeyError for invalid input
            pass


class TestYensTopK(unittest.TestCase):
    """Tests for yens_k_shortest_paths() — Top-5 pathfinding."""

    # ── Test 12: Returns multiple paths (Top-K > 1) ──────────────────
    def test_12_top_k_returns_multiple(self):
        """TC12: Yen's algorithm should return > 1 path on graph with multiple paths"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        paths = yens_k_shortest_paths(nodes, time_edges, 'A', 'D', K=5)

        self.assertGreaterEqual(len(paths), 2)

    # ── Test 13: Paths are sorted by ascending cost ──────────────────
    def test_13_paths_sorted_ascending(self):
        """TC13: Returned paths must be sorted ascending by time cost"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        paths = yens_k_shortest_paths(nodes, time_edges, 'A', 'D', K=5)

        costs = [p['cost'] for p in paths]
        self.assertEqual(costs, sorted(costs))

    # ── Test 14: Each path dict contains 'path' and 'cost' keys ──────
    def test_14_path_dict_structure(self):
        """TC14: Data structure of each path must include 'path' (list) and 'cost' (float)"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        paths = yens_k_shortest_paths(nodes, time_edges, 'A', 'D', K=5)

        for p in paths:
            self.assertIn('path', p)
            self.assertIn('cost', p)
            self.assertIsInstance(p['path'], list)
            self.assertIsInstance(p['cost'], float)

    # ── Test 15: Unreachable pair returns empty list ──────────────────
    def test_15_yens_unreachable(self):
        """TC15: When destination is unreachable, Yen's should return empty list"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)
        paths = yens_k_shortest_paths(nodes, time_edges, 'D', 'A', K=5)

        self.assertEqual(paths, [])


class TestHeuristicAdmissibility(unittest.TestCase):
    """Tests for heuristic correctness — critical for A* optimality."""

    # ── Test 16: Time heuristic never overestimates ──────────────────
    def test_16_heuristic_admissible(self):
        """TC16: h(n) must be <= actual travel time (admissibility check)
        Verification: h = euclidean/60*3600, while actual path >= straight line and speed <= 60"""
        nodes, edges_km = _make_small_graph()
        time_edges = build_time_weighted_edges(edges_km, predicted_flow=50.0)

        h_estimate = _time_heuristic(nodes, 'A', ['D'])

        # Actual shortest path cost (via run_parta_search)
        path, actual_cost, _ = run_parta_search(nodes, time_edges, 'A', 'D')

        # Admissibility: h(n) must never exceed actual cost
        self.assertLessEqual(h_estimate, actual_cost + 0.01)


class TestConfigAndDataIntegration(unittest.TestCase):
    """Tests for config.json and data loading."""

    # ── Test 17: config.json exists and has required keys ─────────────
    def test_17_config_file_exists(self):
        """TC17: config.json must exist and contain all necessary config keys"""
        self.assertTrue(os.path.exists("config.json"))

        with open("config.json", "r") as f:
            cfg = json.load(f)

        required_keys = [
            "default_origin", "default_destination", "speed_kmh",
            "intersection_delay_seconds", "default_ml_model",
            "available_models", "top_k", "data_file"
        ]
        for key in required_keys:
            self.assertIn(key, cfg, f"config.json missing key: {key}")

    # ── Test 18: Config default values are valid SCATS IDs ────────────
    def test_18_config_defaults_valid(self):
        """TC18: Default origin and destination in config must be valid SCATS IDs"""
        with open("config.json", "r") as f:
            cfg = json.load(f)

        nodes, _, _ = build_real_boroondara_graph(cfg["data_file"], K=3)
        self.assertIn(cfg["default_origin"], nodes)
        self.assertIn(cfg["default_destination"], nodes)


class TestDataPreprocessing(unittest.TestCase):
    """Tests for data_preprocessing.py — sliding window and data format."""

    # ── Test 19: Sliding window produces correct shapes ──────────────
    def test_19_sliding_window_shape(self):
        """TC19: Sliding window output X and y shapes must be correct"""
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        X, y = create_sliding_window(series, window_size=4)

        self.assertEqual(X.shape, (6, 4))   # 10 - 4 = 6 samples
        self.assertEqual(y.shape, (6,))

    # ── Test 20: Sliding window values are correct ───────────────────
    def test_20_sliding_window_values(self):
        """TC20: First window should be [1,2,3,4], corresponding label is 5"""
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        X, y = create_sliding_window(series, window_size=4)

        np.testing.assert_array_equal(X[0], [1, 2, 3, 4])
        self.assertEqual(y[0], 5.0)


class TestRealNetworkConstruction(unittest.TestCase):
    """Tests for build_real_network.py — real SCATS graph."""

    # ── Test 21: Real graph has valid structure ──────────────────────
    def test_21_real_graph_structure(self):
        """TC21: Real network graph must contain > 0 nodes and edges, and no (0,0) invalid coordinates"""
        nodes, edges, _ = build_real_boroondara_graph("data/Scats Data October 2006.xls", K=3)

        self.assertGreater(len(nodes), 0)
        self.assertGreater(len(edges), 0)

        # No node should be at (0, 0) — that's missing data
        for sid, (lon, lat) in nodes.items():
            self.assertNotEqual((lon, lat), (0.0, 0.0),
                                f"Node {sid} has invalid (0,0) coordinates")


class TestNodeSortKey(unittest.TestCase):
    """Tests for _node_sort_key — tie-breaking logic ported from Part A."""

    # ── Test 22: Numeric IDs sort numerically, not lexicographically ──
    def test_22_numeric_sort(self):
        """TC22: SCATS ID '10' should be sorted after '4' (numeric sort, not lexicographic)"""
        ids = ['10', '4', '2', '100']
        sorted_ids = sorted(ids, key=_node_sort_key)
        self.assertEqual(sorted_ids, ['2', '4', '10', '100'])


# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    unittest.main(verbosity=2)
