import math, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import yaml

sys.path.insert(0, '.')
from src.graph import build_adjacency_matrix

# Load stations
with open('configs/stations.yaml') as f:
    cfg = yaml.safe_load(f)
stations = cfg['stations']

# Build adjacency matrix exactly as in your pipeline
k = 5
sigma = 120.0
adj = build_adjacency_matrix(
    stations,
    threshold_km=200,
    method='distance_weighted_knn',
    k_neighbors=k,
    distance_sigma_km=sigma
)

names = [s['name'] for s in stations]
pos = {s['name']: (s['longitude'], s['latitude']) for s in stations}

# Build networkx graph
G = nx.Graph()
for n in names:
    G.add_node(n)

n_st = len(stations)
for i in range(n_st):
    for j in range(i+1, n_st):
        w = float(adj[i, j])
        if w > 0.01 and i != j:
            G.add_edge(names[i], names[j], weight=w)

all_weights = [d['weight'] for _, _, d in G.edges(data=True)]
max_w = max(all_weights) if all_weights else 1.0

# Plot
fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
ax.set_facecolor('#f8f8f8')

for (u, v, d) in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    lw = 0.5 + 3.0 * (d['weight'] / max_w)
    alpha = 0.3 + 0.55 * (d['weight'] / max_w)
    ax.plot([x0, x1], [y0, y1], color='#4a90a4',
            linewidth=lw, alpha=alpha, zorder=1)

for name, (x, y) in pos.items():
    ax.scatter(x, y, s=420, color='#5ba3b8',
               edgecolors='#2c6e8a', linewidth=1.8, zorder=3)
    clean = name.replace('_', '\n')
    ax.annotate(clean, (x, y),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=8, fontweight='bold',
                color='#1a1a2e')

ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.set_title(
    r'Spatial Graph of Weather Stations in Atlanta Region'
    '\n'
    r'($k$-NN, $k=5$, $\sigma=120\,$km, exponential distance weights)',
    fontsize=10, pad=10
)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('fig/spatial_graph.png', dpi=180, bbox_inches='tight')
print("Done! Saved to fig/spatial_graph.png")