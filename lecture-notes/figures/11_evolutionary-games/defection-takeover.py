"""Schematic: in a well-mixed population, a defector mutant takes over.

Own drawing, inspired by Fig. 1 of Nowak, Science 314, 1560 (2006).
Output: defection-takeover.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

here = os.path.dirname(os.path.abspath(__file__))
BLUE, RED = "#2a5caa", "#d62f27"

# six positions inside a unit box
POS = [(0.25, 0.72), (0.55, 0.78), (0.80, 0.55), (0.22, 0.33), (0.50, 0.45), (0.72, 0.22)]
STATES = ["CCCCCC", "CCCCCD", "DCCDCD", "DDDDDD"]
LABELS = ["mutation", "selection", "selection"]

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(-0.1, 10.4); ax.set_ylim(-0.95, 1.1); ax.set_aspect("equal"); ax.axis("off")

x0s = [0, 2.9, 5.8, 8.7]
for x0, s in zip(x0s, STATES):
    ax.add_patch(FancyBboxPatch((x0, 0), 1.6, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f4f4f4", ec="#888888", lw=1.2))
    for (px, py), ch in zip(POS, s):
        col = BLUE if ch == "C" else RED
        ax.add_patch(Circle((x0 + 1.6 * px, py), 0.15, fc=col, ec="none"))
        ax.text(x0 + 1.6 * px, py, ch, color="white", ha="center", va="center",
                fontsize=11, fontweight="bold")

for i, lab in enumerate(LABELS):
    xa, xb = x0s[i] + 1.7, x0s[i + 1] - 0.1
    ls = "--" if lab == "mutation" else "-"
    ax.add_patch(FancyArrowPatch((xa, 0.5), (xb, 0.5), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, ls=ls, color="#333333"))
    ax.text((xa + xb) / 2, 0.62, lab, ha="center", va="bottom", fontsize=12)

# fitness arrow: color gradient from blue to red
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("br", [BLUE, RED])
xs = np.linspace(0.3, 9.6, 301)
for xa, xb, u in zip(xs[:-1], xs[1:], np.linspace(0, 1, 300)):
    ax.add_patch(Rectangle((xa, -0.38), xb - xa + 0.005, 0.10, fc=cm(u), ec="none"))
ax.add_patch(Polygon([(9.6, -0.47), (10.1, -0.33), (9.6, -0.19)], fc=RED, ec="none"))
ax.text(5.0, -0.55, "average payoff (fitness) of the population decreases",
        ha="center", va="top", fontsize=12)

fig.savefig(os.path.join(here, "defection-takeover.png"), dpi=200, bbox_inches="tight")
print("done")
