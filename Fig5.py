# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:13:58 2025

@author: Richita
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

edges_file = "414.edges"
G = nx.read_edgelist(edges_file, nodetype=int)


ego_node = int(edges_file.split(".")[0]) 
G.add_node(ego_node)
for n in list(G.nodes()):
    if n != ego_node:
        G.add_edge(ego_node, n)


N = G.number_of_nodes()
k = 0.8 / np.sqrt(N) 
pos = nx.spring_layout(G, k=k, iterations=600)


neighbor_nodes = [n for n in G.nodes() if n != ego_node]

node_size_neighbors = 120     
node_size_ego       = 120    
node_color_neighbors = "#225ea8" 
node_color_ego       = "#b22222" 
edge_color           = "#3a3a3a" 


fig = plt.figure(figsize=(6.5, 6.0), dpi=300)  

nx.draw_networkx_edges(
    G, pos,
    width=0.9,
    edge_color=edge_color,
    alpha=0.55
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=neighbor_nodes,
    node_size=node_size_neighbors,
    node_color=node_color_neighbors,
    edgecolors="white",     
    linewidths=0.3,
    alpha=0.95
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=[ego_node],
    node_size=node_size_ego,
    node_color=node_color_ego,
    edgecolors="white",
    linewidths=0.6,
    alpha=1.0
)


plt.axis("off")
plt.tight_layout()


nodes = sorted(G.nodes())
node_to_idx = {n: i+1 for i, n in enumerate(nodes)}

edges_idx = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()], dtype=np.int32)
X = np.array([pos[n][0] for n in nodes], dtype=float)
Y = np.array([pos[n][1] for n in nodes], dtype=float)
ego_idx = np.int32(node_to_idx[ego_node])

savemat("graph_export.mat", {
    "edges": edges_idx,     # Mx2 int32 (1-based)
    "X": X,                 # Nx1 double
    "Y": Y,                 # Nx1 double
    "ego_idx": ego_idx      # scalar
})

# import networkx as nx


# nx.write_gexf(G, "ego_network.gexf")

# import csv

# with open("ego_network.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Source", "Target"])   # Gephi expects headers
#     for u, v in G.edges():
#         writer.writerow([u, v])
