"""Frequencies of the three male throat-color morphs of the side-blotched lizard
Uta stansburiana, 1990-1995: trajectory on the simplex and time series.

Data: read off the de Finetti triangle in Fig. 2 of Sinervo & Lively,
Nature 380, 240 (1996), by locating each year's point and converting its
position to barycentric coordinates (accuracy about +-0.02).
Output: lizard-morph-cycle.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ORANGE, BLUE, YELLOW = "#e8731a", "#2a5caa", "#e3b505"

years = np.array([1990, 1991, 1992, 1993, 1994, 1995])
#              orange blue  yellow
freq = np.array([[0.11, 0.57, 0.32],
                 [0.14, 0.74, 0.12],
                 [0.24, 0.40, 0.36],
                 [0.17, 0.39, 0.44],
                 [0.16, 0.41, 0.44],
                 [0.10, 0.55, 0.35]])
freq = freq / freq.sum(axis=1, keepdims=True)

# simplex corners as in the original: O bottom-left, B top, Y bottom-right
CO, CB, CY = np.array([0.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), np.array([1.0, 0.0])
xy = freq[:, [0]] * CO + freq[:, [1]] * CB + freq[:, [2]] * CY

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))

# (a) trajectory on the simplex
tri = np.array([CO, CB, CY, CO])
ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
ax.plot(*np.mean([CO, CB, CY], axis=0), "+", color="0.6", ms=10)
for i in range(len(years) - 1):
    ax.annotate("", xy=xy[i + 1], xytext=xy[i],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=14,
                                shrinkA=5, shrinkB=5))
ax.plot(xy[:, 0], xy[:, 1], "o", ms=7, mfc="white", mec="black", mew=1.2, zorder=5)
offs = {1990: (0.03, 0.02), 1991: (0.03, 0.01), 1992: (-0.14, -0.015),
        1993: (-0.03, -0.075), 1994: (0.035, -0.01), 1995: (0.03, -0.01)}
for (x, y), yr in zip(xy, years):
    dx, dy = offs[yr]
    ax.text(x + dx, y + dy, str(yr), fontsize=11)
kw = dict(fontsize=16, fontweight="bold")
ax.text(*(CB + [0, 0.04]), "blue", color=BLUE, ha="center", va="bottom", **kw)
ax.text(*(CO + [-0.02, -0.05]), "orange", color=ORANGE, ha="center", va="top", **kw)
ax.text(*(CY + [0.02, -0.05]), "yellow", color=YELLOW, ha="center", va="top", **kw)
ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.0)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a)", loc="left", fontsize=15)

# (b) time series
for k, (col, lab) in enumerate([(ORANGE, "orange"), (BLUE, "blue"), (YELLOW, "yellow")]):
    bx.plot(years, freq[:, k], "-o", color=col, lw=2.2, ms=6, label=lab)
bx.set_xlabel("year", fontsize=13)
bx.set_ylabel("fraction of males", fontsize=13)
bx.set_ylim(0, 0.8); bx.set_xticks(years)
bx.tick_params(labelsize=11)
for side in ("top", "right"):
    bx.spines[side].set_visible(False)
bx.legend(fontsize=12, frameon=False, loc="upper right", ncol=3)
bx.set_title("(b)", loc="left", fontsize=15)

fig.tight_layout()
fig.savefig(os.path.join(here, "lizard-morph-cycle.png"), dpi=200, bbox_inches="tight")
print("done")
