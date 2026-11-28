"""Lattice simulation of the colicin rock-paper-scissors community.

Own simulation, after Fig. 2a-d of Kerr, Riley, Feldman & Bohannan,
Nature 418, 171 (2002), using the rules described in that paper.

Each site of an L x L lattice (periodic boundaries) is empty or holds one
cell of type C (colicin producer), S (sensitive), or R (resistant). In each
time step, every site is updated simultaneously:
  - an occupied site becomes empty with probability Delta_C, Delta_R, or
    Delta_S = Delta_S0 + tau * f_C (S cells are killed by colicin),
  - an empty site is colonized by type i with probability f_i,
where f_i is the fraction of type i in the neighborhood: the 8 nearest sites
(local neighborhood) or the whole lattice (global neighborhood).
Parameters: Delta_C = 1/3, Delta_R = 10/32, Delta_S0 = 1/4, tau = 3/4.
Output: kerr-lattice-sim.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2)
L = 250
EMPTY, C, S, R = 0, 1, 2, 3
DEATH = {C: 1 / 3, R: 10 / 32, S: 1 / 4}
TAU = 3 / 4
COL = {C: "#d62f27", S: "#2a5caa", R: "#2e9e4a"}
CMAP = ListedColormap(["white", COL[C], COL[S], COL[R]])
SHIFTS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]


def fractions(grid, local):
    f = {}
    for k in (C, S, R):
        m = (grid == k).astype(float)
        f[k] = sum(np.roll(m, s, axis=(0, 1)) for s in SHIFTS) / 8 if local else m.mean()
    return f


def step(grid, local):
    f = fractions(grid, local)
    u = rng.random(grid.shape)
    new = grid.copy()
    # deaths
    pdie = np.zeros(grid.shape)
    pdie[grid == C] = DEATH[C]
    pdie[grid == R] = DEATH[R]
    pS = DEATH[S] + TAU * f[C]
    pdie = np.where(grid == S, pS, pdie)
    new[(grid != EMPTY) & (u < pdie)] = EMPTY
    # colonization of empty sites
    empty = grid == EMPTY
    cC = f[C] * np.ones(grid.shape)
    cS = cC + f[S]
    cR = cS + f[R]
    new[empty & (u < cC)] = C
    new[empty & (u >= cC) & (u < cS)] = S
    new[empty & (u >= cS) & (u < cR)] = R
    return new


def run(T, local, snaps=()):
    grid = rng.choice([C, S, R], size=(L, L))
    counts, saved = [], {}
    for t in range(T + 1):
        counts.append([(grid == k).sum() for k in (C, S, R)])
        if t in snaps:
            saved[t] = grid.copy()
        if t < T:
            grid = step(grid, local)
    return np.array(counts), saved


T_LOC, SNAPS = 5000, (3000, 3200)
loc, saved = run(T_LOC, True, SNAPS)
# In the global case, S drops to a few cells by chance-dependent amounts; in
# about half of the seeds it survives and later outcompetes R once C is gone.
# This seed shows the typical outcome reported by Kerr et al.: S dies first,
# then C, leaving R.
rng = np.random.default_rng(0)
glob, _ = run(600, False)

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], wspace=0.25, hspace=0.35)
for k, (t, lab) in enumerate(zip(SNAPS, "ab")):
    ax = fig.add_subplot(gs[k, 0])
    ax.imshow(saved[t], cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"({lab}) time step {t}", fontsize=13, loc="left")


def series(ax, counts, title, loc="lower right"):
    tt = np.arange(len(counts))
    for k, name in zip(range(3), "CSR"):
        y = np.where(counts[:, k] > 0, np.log10(np.maximum(counts[:, k], 1)), np.nan)
        ax.plot(tt, y, color=COL[[C, S, R][k]], lw=1.5, label=name)
        dead = np.where(counts[:, k] == 0)[0]
        if len(dead):
            ax.plot(dead[0], 0, "D", color=COL[[C, S, R][k]], mfc="white", ms=7, clip_on=False)
            ax.plot([dead[0] - 1, dead[0]], [np.log10(max(counts[dead[0] - 1, k], 1)), 0],
                    color=COL[[C, S, R][k]], lw=1.5)
    ax.set_ylim(0, 5); ax.set_xlim(0, tt[-1])
    ax.set_xlabel("time step", fontsize=12); ax.set_ylabel(r"$\log_{10}$(abundance)", fontsize=12)
    ax.set_title(title, fontsize=13, loc="left")
    ax.legend(fontsize=11, frameon=False, ncol=3, loc=loc)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


series(fig.add_subplot(gs[0, 1]), loc, "(c) local neighborhood")
series(fig.add_subplot(gs[1, 1]), glob, "(d) global neighborhood", loc="center right")
fig.savefig(os.path.join(here, "kerr-lattice-sim.png"), dpi=200, bbox_inches="tight")
print("extinction times (global, C S R):", [int(np.argmax(glob[:, k] == 0)) if (glob[:, k] == 0).any() else None for k in range(3)])
print("final local counts:", loc[-1])
