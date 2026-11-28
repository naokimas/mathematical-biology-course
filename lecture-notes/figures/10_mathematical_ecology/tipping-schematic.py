"""Schematic of an ecosystem tipping point: tree cover collapses abruptly as
the climate gets drier (forest -> savanna/grassland). Own drawing.
Output: tipping-schematic.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap

here = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)

# background: lush green (wet) to sandy (dry)
bg = LinearSegmentedColormap.from_list("bg", ["#d9efd9", "#eef3d2", "#f6ebc8", "#f3e2bd"])
ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(0, 10, 0, 6), cmap=bg, aspect="auto", zorder=0)
ax.add_patch(Rectangle((0, 0), 10, 0.35, color="#c8b38a", zorder=1))  # ground


def tree(x, h, crown, col="#3f7f3f"):
    ax.add_patch(Rectangle((x - 0.06, 0.35), 0.12, h, color="#7a5230", zorder=2))
    for dx, dy, r in [(0, h + 0.15, crown), (-0.6 * crown, h - 0.1, 0.8 * crown),
                      (0.6 * crown, h - 0.1, 0.8 * crown), (0, h + 0.55 * crown, 0.75 * crown)]:
        ax.add_patch(Circle((x + dx, 0.35 + dy), r, color=col, zorder=3))


def dead_tree(x, h):
    ax.plot([x, x], [0.35, 0.35 + h], color="#7a5230", lw=3, zorder=2, solid_capstyle="round")
    for s, y0 in [(1, 0.5), (-1, 0.7), (1, 0.85)]:
        ax.plot([x, x + 0.25 * s], [0.35 + y0 * h, 0.35 + y0 * h + 0.25], color="#7a5230",
                lw=2, zorder=2, solid_capstyle="round")


def grass(x):
    for a in (-0.12, 0, 0.12):
        ax.plot([x, x + a], [0.35, 0.6], color="#a39a4a", lw=1.5, zorder=2)


for x, h, c in [(0.6, 1.0, 0.45), (1.5, 1.2, 0.5), (2.4, 0.9, 0.42), (3.2, 1.0, 0.4)]:
    tree(x, h, c)
tree(4.3, 0.7, 0.3, "#6d9a48")
dead_tree(5.4, 1.1)
tree(6.2, 0.5, 0.22, "#8fa553")
dead_tree(7.0, 0.9)
for x in np.arange(7.6, 9.9, 0.45):
    grass(x)

# state curve: tree cover vs dryness
x = np.linspace(0.3, 9.7, 400)
y = 1.9 + 3.0 / (1 + np.exp(4.0 * (x - 5.6)))
ax.plot(x, y, color="#1f6b2f", lw=4, zorder=4)
# tipping point marker on the steep part
xt = 5.0; yt = 1.9 + 3.0 / (1 + np.exp(4.0 * (xt - 5.6)))
ax.plot(xt, yt + 0.22, "o", ms=16, mfc="#d62f27", mec="black", zorder=5)
ax.annotate("tipping point", xy=(5.2, 4.95), xytext=(6.4, 5.35), fontsize=14,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=6)

ax.annotate("", xy=(9.9, -0.45), xytext=(0.1, -0.45), annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
ax.text(5, -0.95, "drier climate (less rainfall)", ha="center", fontsize=15)
ax.annotate("", xy=(-0.35, 5.9), xytext=(-0.35, 0.1), annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
ax.text(-0.75, 3, "tree cover", rotation=90, va="center", ha="center", fontsize=15)
ax.text(0.4, 5.55, "forest", fontsize=14, color="#1f6b2f", fontweight="bold")
ax.text(9.6, 1.25, "savanna /\ngrassland", fontsize=14, color="#8a6d2f", fontweight="bold", ha="right")
ax.axis("off")
fig.savefig(os.path.join(here, "tipping-schematic.png"), dpi=200, bbox_inches="tight")
print("done")
