import json
import os
import sys

# Ensure imports and file paths resolve relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import joblib

from tensorflow.keras.models import load_model


from data_preprocessing import prepare_dataloaders
from route_guidance import yens_k_shortest_paths, calculate_travel_time
from build_real_network import build_real_boroondara_graph

import warnings
warnings.filterwarnings("ignore")

class RouteGuidanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Route Guidance System")
        self.root.geometry("1000x700")
        
        self.load_config()
        self.create_widgets()
        # Use 'after' to let the UI render first before heavy ML loading
        self.root.after(100, self.init_data_and_models)

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config.json: {e}")
            self.config = {
                "default_origin": "SCATS_Origin",
                "default_destination": "SCATS_Dest",
                "default_ml_model": "GRU",
                "available_models": ["LSTM", "GRU", "Random Forest"]
            }

    def init_data_and_models(self):
        self.status_var.set("Loading real road network... Please wait.")
        self.root.update()
        
        # Build graph from VicRoads SCATS data
        self.nodes, self.edges_km, self.site_names = build_real_boroondara_graph("data/Scats Data October 2006.xls", K=3)
        scats_list = list(self.nodes.keys())
        

        self.origin_cb['values'] = scats_list
        self.dest_cb['values'] = scats_list
        
        if scats_list:
            if self.origin_var.get() not in scats_list:
                self.origin_var.set(scats_list[0])
            if self.dest_var.get() not in scats_list:
                self.dest_var.set(scats_list[-1])
        
        self.status_var.set("Loading dataset and ML models... Please wait.")
        self.root.update()
        
        try:
            # Get scaler and sample test sequence from dataloader
            _, _, _, self.scaler, arrays = prepare_dataloaders("data/Scats Data October 2006.xls", window_size=4)
            X_test = arrays[4]
            self.sample_sequence = X_test[0:1]  # first window for demo
            
            self.models = {}
            if os.path.exists("models/lstm_model.keras"):
                self.models["LSTM"] = load_model("models/lstm_model.keras")
            if os.path.exists("models/gru_model.keras"):
                self.models["GRU"] = load_model("models/gru_model.keras")
            if os.path.exists("models/random_forest_model.pkl"):
                self.models["Random Forest"] = joblib.load("models/random_forest_model.pkl")
                
            self.status_var.set("Models loaded. Ready to compute routes.")
            self.calc_btn.config(state=tk.NORMAL)
            
            self.visualize_graph(self.edges_km, [])
        except Exception as e:
            self.status_var.set(f"Error loading models: {e}")
            messagebox.showerror("Initialization Error", str(e))
            
    def create_widgets(self):

        input_frame = ttk.LabelFrame(self.root, text="Configuration & Search Parameters", padding=10)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Origin (O):").grid(row=0, column=0, padx=5, pady=5)
        self.origin_var = tk.StringVar(value=self.config.get("default_origin", ""))
        self.origin_cb = ttk.Combobox(input_frame, textvariable=self.origin_var, values=[])
        self.origin_cb.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Destination (D):").grid(row=0, column=2, padx=5, pady=5)
        self.dest_var = tk.StringVar(value=self.config.get("default_destination", ""))
        self.dest_cb = ttk.Combobox(input_frame, textvariable=self.dest_var, values=[])
        self.dest_cb.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="ML Model:").grid(row=0, column=4, padx=5, pady=5)
        self.model_var = tk.StringVar(value=self.config.get("default_ml_model", "GRU"))
        self.model_cb = ttk.Combobox(input_frame, textvariable=self.model_var, values=self.config.get("available_models", []))
        self.model_cb.grid(row=0, column=5, padx=5, pady=5)
        
        # Disabled initially until models load
        self.calc_btn = ttk.Button(input_frame, text="Find Top-5 Routes", command=self.calculate_routes, state=tk.DISABLED)
        self.calc_btn.grid(row=0, column=6, padx=15, pady=5)


        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        

        self.result_text = tk.Text(content_frame, width=45, state=tk.DISABLED, font=("Consolas", 10))
        self.result_text.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=content_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        

        self.status_var = tk.StringVar(value="Initializing UI...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def calculate_routes(self):
        origin = self.origin_var.get()
        dest = self.dest_var.get()
        model_name = self.model_var.get()
        
        if origin not in self.nodes or dest not in self.nodes:
            messagebox.showwarning("Warning", "Invalid Origin or Destination node selected.")
            return
            
        if model_name not in self.models:
            messagebox.showwarning("Warning", f"Selected model '{model_name}' is not loaded.")
            return
            
        self.status_var.set(f"Using {model_name} to predict traffic...")
        self.root.update()
        
        model = self.models[model_name]
        

        if model_name == "Random Forest":
            X_rf = self.sample_sequence.reshape(self.sample_sequence.shape[0], -1)
            pred_scaled = model.predict(X_rf)[0]
        else:
            pred_scaled = model.predict(self.sample_sequence, verbose=0)[0][0]
            
        predicted_flow = self.scaler.inverse_transform([[pred_scaled]])[0][0]
        predicted_flow = max(0, predicted_flow)
        
        # Dynamically update edge costs based on predicted flow
        updated_edges = {}
        for u, neighbors in self.edges_km.items():
            updated_edges[u] = []
            for v, dist in neighbors:
                tt = calculate_travel_time(predicted_flow, dist)
                updated_edges[u].append((v, tt))
                
        self.status_var.set("Running Yen's Top-K Algorithm...")
        self.root.update()
        
        top_k_paths = yens_k_shortest_paths(self.nodes, updated_edges, origin, dest, K=5)
        
        self.display_results(top_k_paths, predicted_flow)
        
        if top_k_paths:
            self.visualize_graph(updated_edges, top_k_paths[0]['path'])
        else:
            self.visualize_graph(updated_edges, [])
            
        self.status_var.set("Search complete.")

    def display_results(self, paths, flow):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Predicted Traffic: {flow:.1f} veh/15m\n")
        self.result_text.insert(tk.END, "="*40 + "\n\n")
        
        if not paths:
            self.result_text.insert(tk.END, "No valid routes found.\n")
        else:
            for i, p in enumerate(paths):
                cost_min = p['cost'] / 60.0
                route_str = " ->\n  ".join(p['path'])
                self.result_text.insert(tk.END, f"[Rank {i+1}]\n")
                self.result_text.insert(tk.END, f"Est. Time : {cost_min:.2f} mins\n")
                self.result_text.insert(tk.END, f"Route     : {route_str}\n")
                self.result_text.insert(tk.END, "-"*40 + "\n")
                
        self.result_text.config(state=tk.DISABLED)

    def visualize_graph(self, edges, best_path):
        self.ax.clear()
        
        G = nx.DiGraph()
        for u, pos in self.nodes.items():
            G.add_node(u, pos=pos)
            
        for u, nbrs in edges.items():
            for v, cost in nbrs:
                # 'cost' is total time in seconds. Formatting as minutes.
                weight_label = f"{cost/60:.1f} min"
                G.add_edge(u, v, weight=weight_label)
                
        pos = nx.get_node_attributes(G, 'pos')
        
        # Base network (light gray)
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='#E0E0E0', node_size=150, edgecolors='#C0C0C0')
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_size=6, font_color='gray')
        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color='#D3D3D3', arrows=False, width=1.0)
        
        # Highlight best path
        if best_path:
            path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
            
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=path_edges, edge_color='red', width=3.0, arrows=True, arrowstyle='-|>', arrowsize=15, node_size=300)
            nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=best_path, node_color='red', node_size=300)
            path_labels = {node: node for node in best_path}
            nx.draw_networkx_labels(G, pos, ax=self.ax, labels=path_labels, font_size=9, font_color='black', font_weight='bold')
            
            # Extract weights just for the path to prevent clutter
            edge_labels = nx.get_edge_attributes(G, 'weight')
            path_edge_labels = {edge: edge_labels.get(edge) or edge_labels.get((edge[1], edge[0])) for edge in path_edges}
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edge_labels, ax=self.ax, font_size=8, font_color='darkred')
            
            self.ax.set_title("Boroondara Network (Optimum Path Highlighted in Red)", pad=10)
        else:
            self.ax.set_title("Boroondara Network", pad=10)
            
        self.ax.axis('off')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RouteGuidanceApp(root)
    root.mainloop()
