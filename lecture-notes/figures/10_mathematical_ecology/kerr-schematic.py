"""Schematic (not quantitative) summary of the outcome of the experiment of
Kerr, Riley, Feldman & Bohannan, Nature 418, 171 (2002): three E. coli strains
(colicin producer C, sensitive S, resistant R) coexist on a static plate but
not in a well-mixed flask. The curves are hand-made shapes, not data.
Output: kerr-schematic.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
RED, BLUE, GREEN = "#d62f27", "#2a5caa", "#2e9e4a"
t = np.linspace(0, 1, 400)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)

# (a) static plate: all three persist
ax = axes[0]
ax.plot(t, 0.75 + 0.03 * np.sin(9 * t), color=GREEN, lw=2.5)
ax.plot(t, 0.62 + 0.04 * np.sin(9 * t + 2.1), color=BLUE, lw=2.5)
ax.plot(t, 0.50 + 0.04 * np.sin(9 * t + 4.2), color=RED, lw=2.5)
for y, lab, col in [(0.80, "R", GREEN), (0.67, "S", BLUE), (0.55, "C", RED)]:
    ax.text(1.02, y - 0.04, lab, color=col, fontsize=14, fontweight="bold", va="center")
ax.set_title("(a) static plate (local interactions)", fontsize=13, loc="left")

# (b) flask: S dies out first, then C; R remains
ax = axes[1]
ax.plot(t, 0.75 * (1 - np.exp(-t / 0.05)) * 0.2 + 0.6, color=GREEN, lw=2.5)
s = np.clip(0.6 - 4.0 * t, 0, None)
ax.plot(t[s > 0], s[s > 0], color=BLUE, lw=2.5)
c = np.clip(0.6 + 0.3 * t - 1.6 * t ** 2.2 - 0.5 * t, 0, None)
c = np.where(t < 0.62, np.maximum(c, 0), 0)
ax.plot(t[c > 0], c[c > 0], color=RED, lw=2.5)
ax.plot(t[np.argmin(s > 0)], 0, "x", color=BLUE, ms=10, mew=2.5, clip_on=False, zorder=5)
ax.plot(t[np.argmin(c > 0)], 0, "x", color=RED, ms=10, mew=2.5, clip_on=False, zorder=5)
ax.text(1.02, 0.75, "R", color=GREEN, fontsize=14, fontweight="bold", va="center")
ax.text(0.05, 0.25, "S", color=BLUE, fontsize=14, fontweight="bold")
ax.text(0.40, 0.42, "C", color=RED, fontsize=14, fontweight="bold")
ax.set_title("(b) flask (well-mixed)", fontsize=13, loc="left")

for ax in axes:
    ax.set_xlim(0, 1.08); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("time", fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("abundance", fontsize=13)

fig.tight_layout()
fig.savefig(os.path.join(here, "kerr-schematic.png"), dpi=200, bbox_inches="tight")
print("done")
