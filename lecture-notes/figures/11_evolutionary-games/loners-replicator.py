"""Replicator dynamics of the optional public goods game (cooperators,
defectors, and loners) on the simplex.

Own simulation, after Fig. 1(A, B) of Hauert, De Monte, Hofbauer & Sigmund,
Science 296, 1129 (2002).

Model: groups of N players are sampled from an infinite well-mixed population
with fractions x (cooperators, C), y (defectors, D), and z (loners, L).
Loners do not participate and get sigma. Each participant cooperator pays
c = 1; the contributions are multiplied by r and shared among the S
participants. If S = 1, the single participant gets sigma like a loner.
The expected payoffs are computed exactly by enumerating the strategies of
the N - 1 co-players. Replicator equation: dx_i/dt = x_i (P_i - mean P).

Parameters (as in the original): N = 5, c = 1; (a) r = 1.8, sigma = 0.5;
(b) r = 3, sigma = 1.
Output: loners-replicator.png
"""
import os
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

here = os.path.dirname(os.path.abspath(__file__))
N, C = 5, 1.0
BLUE, RED, GRAY = "#2a5caa", "#d62f27", "#555555"

# all compositions (nc, nd, nl) of the N - 1 co-players
COMPS = [(a, b, N - 1 - a - b) for a in range(N) for b in range(N - a)]
MULT = [comb(N - 1, a) * comb(N - 1 - a, b) for a, b, _ in COMPS]


def payoffs(p, r, SIGMA):
    x, y, z = p
    Pc = Pd = 0.0
    for (a, b, l), m in zip(COMPS, MULT):
        w = m * x**a * y**b * z**l
        S = a + b + 1  # participants including the focal player
        if S == 1:
            Pc += w * SIGMA
            Pd += w * SIGMA
        else:
            Pc += w * (r * C * (a + 1) / S - C)
            Pd += w * (r * C * a / S)
    return np.array([Pc, Pd, SIGMA])


def rhs_log(t, u, r, sig):
    """Replicator equation in log coordinates u_i = log x_i (keeps x_i > 0)."""
    p = np.exp(u - u.max()); p /= p.sum()
    P = payoffs(p, r, sig)
    return P - p @ P


def orbit(p0, r, sig, T, back=True):
    u0 = np.log(p0)
    ts = np.linspace(0, T, 4000)
    fw = solve_ivp(rhs_log, (0, T), u0, args=(r, sig), t_eval=ts, rtol=1e-9, atol=1e-11).y.T
    parts = [fw]
    if back:
        bw = solve_ivp(rhs_log, (0, -T), u0, args=(r, sig), t_eval=-ts, rtol=1e-9, atol=1e-11).y.T
        parts = [bw[::-1], fw]
    u = np.vstack(parts)
    p = np.exp(u - u.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


# simplex -> plane: L bottom-left, D bottom-right, C top
def xy(p):
    p = np.atleast_2d(p)
    return p[:, 1] + 0.5 * p[:, 0], (np.sqrt(3) / 2) * p[:, 0]


def arrow(ax, X, Y, i, k=6):
    ax.annotate("", xy=(X[i + k], Y[i + k]), xytext=(X[i], Y[i]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14))


def frame(ax, title):
    E = {"C": (1, 0, 0), "D": (0, 1, 0), "L": (0, 0, 1)}
    P = {k: np.array([c[0] for c in xy(v)]) for k, v in E.items()}
    tri = np.array([P["L"], P["D"], P["C"], P["L"]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
    # flow on the edges: C -> D, D -> L, L -> C
    for a, b in [("C", "D"), ("D", "L"), ("L", "C")]:
        m = 0.5 * (P[a] + P[b]); d = 0.04 * (P[b] - P[a])
        ax.annotate("", xy=m + d, xytext=m - d,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14))
    for k in E:
        ax.plot(*P[k], "o", ms=8, mfc="white", mec="black", mew=1.2, zorder=5)
    kw = dict(fontsize=24, fontweight="bold")
    ax.text(P["C"][0], P["C"][1] + 0.05, "C", color=BLUE, ha="center", va="bottom", **kw)
    ax.text(P["L"][0] - 0.04, P["L"][1] - 0.03, "L", color=GRAY, ha="right", va="top", **kw)
    ax.text(P["D"][0] + 0.04, P["D"][1] - 0.03, "D", color=RED, ha="left", va="top", **kw)
    ax.set_title(title, fontsize=22, loc="left")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.0)
    ax.set_aspect("equal"); ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# (a) r = 1.8 < 2: every interior orbit is homoclinic to the loner state
ax = axes[0]
r, sig = 1.8, 0.5
frame(ax, r"(a)  $r = 1.8$, $\sigma = 0.5$")
for s in [0.45, 0.7, 0.9]:
    p0 = np.array([0.6 * s, 0.4 * s, 1 - s])
    p = orbit(p0, r, sig, 400)
    p = p[p[:, 2] < 0.9999]
    X, Y = xy(p)
    ax.plot(X, Y, color="black", lw=1.8)
    i = int(np.argmax(Y)); arrow(ax, X, Y, i)                      # top of the loop
    j = int(np.argmax(X)); arrow(ax, X, Y, j)                      # rightmost point

# (b) r = 3 > 2: closed orbits around the interior fixed point Q
ax = axes[1]
r, sig = 3.0, 1.0
frame(ax, r"(b)  $r = 3$, $\sigma = 1$")

def interior(v):
    x, y = v
    P = payoffs((x, y, 1 - x - y), r, sig)
    return [P[0] - P[2], P[1] - P[2]]

xq, yq = fsolve(interior, [0.3, 0.3])
Q = np.array([xq, yq, 1 - xq - yq])
print("Q =", Q)
for s in [0.15, 0.35, 0.55]:
    p0 = Q + s * (np.array([0.0, 1.0, 0.0]) - Q)
    p = orbit(p0, r, sig, 400, back=False)
    X, Y = xy(p)
    ax.plot(X, Y, color="black", lw=1.8)
    i = int(np.argmax(Y)); arrow(ax, X, Y, i)
Xq, Yq = xy(Q)
ax.plot(Xq, Yq, "o", ms=7, mfc="white", mec="black", mew=1.2, zorder=5)
ax.text(Xq[0] + 0.03, Yq[0] + 0.01, "Q", fontsize=20, va="bottom")

fig.tight_layout()
fig.savefig(os.path.join(here, "loners-replicator.png"), dpi=200, bbox_inches="tight")
print("done")
