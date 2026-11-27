"""Deterministic spatial prisoner's dilemma of Nowak & May.

Own simulations, inspired by
  Nowak & May, Nature 359, 826 (1992), Fig. 1, and
  Nowak & May, Int. J. Bifurcation Chaos 3, 35 (1993), Figs. 4 and 11.

Rule: each player plays the game with every neighbor and with itself
(payoffs R = 1, T = b, S = P = 0) and sums the payoffs. Then, all players
simultaneously adopt the strategy of the highest-scoring player among
themselves and their neighbors (a player keeps its strategy in a tie).

Colors: blue C->C, red D->D, yellow C->D, green D->C.

Output:
  spatial-pd-random.png   200x200, fixed boundaries, 10% defectors at random,
                          Moore neighborhood (8 neighbors), after 200 generations
  spatial-pd-kaleido.png  49x49, fixed boundaries, single defector at the center,
                          Moore neighborhood, b = 1.85
  spatial-pd-fractal.png  single defector on an unbounded square lattice,
                          von Neumann neighborhood (4 neighbors), b = 1.8
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))

MOORE = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]  # includes self
VN = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]                # includes self

# 0: C->C (blue), 1: D->D (red), 2: C->D (yellow), 3: D->C (green)
CMAP = ListedColormap(["#2a5caa", "#d62f27", "#f5c518", "#37a24b"])


def shifted(a, di, dj, fill):
    """a shifted by (di, dj) with fixed (non-periodic) boundaries."""
    n, m = a.shape
    out = np.full_like(a, fill)
    out[max(di, 0):n + min(di, 0), max(dj, 0):m + min(dj, 0)] = \
        a[max(-di, 0):n + min(-di, 0), max(-dj, 0):m + min(-dj, 0)]
    return out


def step(c, b, nbhd):
    """c: boolean array, True = cooperator. Returns the next generation."""
    ci = c.astype(float)
    nc = sum(shifted(ci, di, dj, 0.0) for di, dj in nbhd)  # cooperators met
    pay = np.where(c, nc, b * nc)
    best_pay = pay.copy()
    best_c = c.copy()
    for di, dj in nbhd:
        if (di, dj) == (0, 0):
            continue
        p = shifted(pay, di, dj, -np.inf)
        s = shifted(c, di, dj, False)
        better = p > best_pay + 1e-12
        best_pay = np.where(better, p, best_pay)
        best_c = np.where(better, s, best_c)
    return best_c


def color_code(prev, cur):
    return np.select([prev & cur, ~prev & ~cur, prev & ~cur], [0, 1, 2], 3)


def run(c, b, nbhd, T):
    """Returns the list of color-coded states for t = 0, ..., T."""
    frames = [color_code(c, c)]
    for _ in range(T):
        new = step(c, b, nbhd)
        frames.append(color_code(c, new))
        c = new
    return frames


def show(ax, img, title=None):
    ax.imshow(img, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)


# 1. Random initial condition (Nature 1992, Fig. 1)
rng = np.random.default_rng(1)
c0 = rng.random((200, 200)) > 0.1
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
for ax, b in zip(axes, [1.77, 1.9]):
    fr = run(c0.copy(), b, MOORE, 200)
    show(ax, fr[-1], f"$b = {b}$")
fig.tight_layout()
fig.savefig(os.path.join(here, "spatial-pd-random.png"), dpi=150)
plt.close(fig)

# 2. Kaleidoscope from a single defector (IJBC 1993, Fig. 4)
n = 49
c0 = np.ones((n, n), bool); c0[n // 2, n // 2] = False
fr = run(c0, 1.85, MOORE, 160)
ts = list(range(0, 160, 4))  # 40 snapshots
fig, axes = plt.subplots(5, 8, figsize=(12, 8))
for ax, t in zip(axes.flat, ts):
    show(ax, fr[t]); ax.set_title(f"$t={t}$", fontsize=9, pad=2)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(here, "spatial-pd-kaleido.png"), dpi=150)
plt.close(fig)

# 3. Dynamic fractal with the von Neumann neighborhood (IJBC 1993, Fig. 11)
n = 2 * 100 + 1  # large enough that the pattern never touches the boundary
c0 = np.ones((n, n), bool); c0[n // 2, n // 2] = False
T = 96
fr = run(c0, 1.8, VN, T)
ts = [24, 48, 72, 96]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
for ax, t in zip(axes, ts):
    w = T + 3
    sub = fr[t][n // 2 - w:n // 2 + w + 1, n // 2 - w:n // 2 + w + 1]
    show(ax, sub, f"$t = {t}$")
fig.tight_layout()
fig.savefig(os.path.join(here, "spatial-pd-fractal.png"), dpi=150)
plt.close(fig)
print("done")
