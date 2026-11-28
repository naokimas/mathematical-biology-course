"""Stability of random community matrices for three interaction types.

Own simulation, after Fig. 1 of Allesina & Tang, Nature 483, 205 (2012).

Community matrix M (S x S): diagonal -d. Each pair (i, j) interacts with
probability C. For an interacting pair, with X, Y independent N(0, sigma^2):
  random:        M_ij = X, M_ji = Y (each entry nonzero independently, prob C)
  predator-prey: (M_ij, M_ji) = (|X|, -|Y|) or (-|X|, |Y|) with prob 1/2 each
  mixture:       (M_ij, M_ji) = (|X|, |Y|) (mutualism) or (-|X|, -|Y|)
                 (competition) with prob 1/2 each
Elliptic law: the eigenvalues fill an ellipse centered at -d with horizontal
and vertical semi-axes sigma sqrt(SC) (1 + rho) and sigma sqrt(SC) (1 - rho),
where rho = corr(M_ij, M_ji) = 0, -2/pi, +2/pi for the three types.
Stability (all eigenvalues have negative real part) requires
  sigma sqrt(SC) (1 + rho) < d.
Output: random-community-stability.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
D = 1.0
TYPES = ["random", "predator-prey", "mixture"]
TITLES = ["Random", "Predator–prey", "Competition + mutualism"]
RHO = {"random": 0.0, "predator-prey": -2 / np.pi, "mixture": 2 / np.pi}
BLUE, RED = "#2a5caa", "#d62f27"


def community(S, C, sigma, kind):
    M = np.zeros((S, S))
    iu = np.triu_indices(S, 1)
    n = len(iu[0])
    if kind == "random":
        M[iu] = rng.normal(0, sigma, n) * (rng.random(n) < C)
        M[iu[1], iu[0]] = rng.normal(0, sigma, n) * (rng.random(n) < C)
    else:
        link = rng.random(n) < C
        x, y = np.abs(rng.normal(0, sigma, n)), np.abs(rng.normal(0, sigma, n))
        s = rng.choice([-1, 1], n)
        if kind == "predator-prey":
            M[iu], M[iu[1], iu[0]] = s * x * link, -s * y * link
        else:
            M[iu], M[iu[1], iu[0]] = s * x * link, s * y * link
    M[np.diag_indices(S)] = -D
    return M


fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), gridspec_kw=dict(height_ratios=[1.15, 1]))

# (a) eigenvalue clouds, S = 250, C = 0.25, sigma sqrt(SC) = 10
S, C = 250, 0.25
sig = 10 / np.sqrt(S * C)
for ax, kind, title in zip(axes[0], TYPES, TITLES):
    for _ in range(4):
        ev = np.linalg.eigvals(community(S, C, sig, kind))
        ax.plot(ev.real, ev.imag, ".", color=BLUE, ms=2.5, alpha=0.5)
    r = sig * np.sqrt(S * C)
    ax.add_patch(Ellipse((-D, 0), 2 * r * (1 + RHO[kind]), 2 * r * (1 - RHO[kind]),
                         fill=False, ec="black", lw=1.8))
    ax.axvline(0, color=RED, lw=1.5, ls="--")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20); ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("real part", fontsize=12)
    ax.tick_params(labelsize=10)
axes[0, 0].set_ylabel("imaginary part", fontsize=12)

# (b) probability of stability
NREP = 200
for ax, kind, title in zip(axes[1], TYPES, TITLES):
    thr = D / (1 + RHO[kind])
    xs = np.linspace(0.6 * thr, 1.4 * thr, 17)
    for (S, C), mk in zip([(50, 0.5), (250, 0.1)], ["o", "s"]):
        ps = []
        for x in xs:
            sig = x / np.sqrt(S * C)
            ps.append(np.mean([np.linalg.eigvals(community(S, C, sig, kind)).real.max() < 0
                               for _ in range(NREP)]))
        ax.plot(xs, ps, "-" + mk, color=BLUE if S == 250 else "0.45", ms=5, lw=1.2,
                mfc="white" if S == 50 else BLUE, label=f"$S={S}$, $C={C}$")
    ax.axvline(thr, color=RED, lw=1.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\sigma\sqrt{SC}$", fontsize=13)
    ax.tick_params(labelsize=10)
axes[1, 0].set_ylabel("probability of stability", fontsize=12)
axes[1, 0].legend(fontsize=10, frameon=False, loc="lower left")

fig.text(0.01, 0.98, "(a)", fontsize=16, va="top")
fig.text(0.01, 0.47, "(b)", fontsize=16, va="top")
fig.tight_layout(rect=(0.02, 0, 1, 1))
fig.savefig(os.path.join(here, "random-community-stability.png"), dpi=200, bbox_inches="tight")
print("done")
