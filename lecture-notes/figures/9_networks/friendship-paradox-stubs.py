"""Six nodes with known degrees but unknown edges (friendship paradox exercise).
Each node v_i is a circle with as many edge stubs as its degree, pointing to the
center ("?": we pretend not to know which nodes are connected).
Output: friendship-paradox-stubs.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
degrees = [1, 2, 3, 1, 4, 1]          # v_1, ..., v_6 (clockwise from the top)
R, r, stub = 1.0, 0.13, 0.22           # ring radius, node radius, stub length

fig, ax = plt.subplots(figsize=(5, 5))
for i, k in enumerate(degrees):
    th = np.pi / 2 - 2 * np.pi * i / len(degrees)
    c = np.array([np.cos(th), np.sin(th)]) * R
    inward = -c / R
    ax.add_patch(plt.Circle(c, r, fill=False, lw=1.5))
    ax.text(*c, f"$v_{i + 1}$", ha="center", va="center", fontsize=16)
    spread = np.linspace(-0.45, 0.45, k) if k > 1 else [0.0]
    for a in spread:
        d = np.array([inward[0] * np.cos(a) - inward[1] * np.sin(a),
                      inward[0] * np.sin(a) + inward[1] * np.cos(a)])
        p0 = c + r * d
        ax.plot(*zip(p0, p0 + stub * d), color="black", lw=1.5)
    ax.text(*(c * (R + 0.30) / R), str(k), ha="center", va="center", fontsize=20)
ax.text(0, 0, "?", ha="center", va="center", fontsize=26)
ax.set_xlim(-1.45, 1.45)
ax.set_ylim(-1.45, 1.45)
ax.set_aspect("equal")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(here, "friendship-paradox-stubs.png"), dpi=200)
