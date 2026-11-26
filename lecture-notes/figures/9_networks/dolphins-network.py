"""Dolphin social network with communities.

Data: dolphins.gml, the network of frequent associations among 62 bottlenose
dolphins in Doubtful Sound, New Zealand, compiled by Lusseau et al.,
Behav. Ecol. Sociobiol. 54, 396-405 (2003); file distributed by M. E. J. Newman
(http://www-personal.umich.edu/~mejn/netdata/).
Communities: modularity maximization by the Louvain algorithm (Blondel et al. 2008).
Layout: spring (force-directed) layout.
Output: dolphins-network.png
"""
import os
import networkx as nx
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
G = nx.read_gml(os.path.join(here, "dolphins.gml"))
comms = sorted(nx.community.louvain_communities(G, seed=1), key=len, reverse=True)
print("nodes", G.number_of_nodes(), "edges", G.number_of_edges(),
      "communities", [len(c) for c in comms],
      "modularity %.3f" % nx.community.modularity(G, comms))

colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
node_color = {v: colors[i] for i, c in enumerate(comms) for v in c}
pos = nx.spring_layout(G, seed=1, iterations=500)

fig, ax = plt.subplots(figsize=(7, 6))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.6", width=0.8)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=90,
                       node_color=[node_color[v] for v in G],
                       edgecolors="black", linewidths=0.6)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(here, "dolphins-network.png"), dpi=200)
