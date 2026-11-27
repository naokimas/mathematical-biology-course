"""Evolution of reactive strategies in the iterated prisoner's dilemma:
defectors, then TFT, then generous TFT (GTFT).

Own simulation, after Fig. 1 of Nowak & Sigmund, Nature 355, 250 (1992).

Strategy (p, q): cooperate with probability p after the opponent cooperated
and with probability q after the opponent defected; p, q in [0.001, 0.999].
Payoffs R = 3, S = 0, T = 5, P = 1. For the infinitely repeated game, player
1 with (p, q) cooperates in the long run with probability
    c = (q + r q') / (1 - r r'),  r = p - q,  r' = p' - q',
against player 2 with (p', q'), and similarly c' for player 2. The expected
payoff per round is R c c' + S c (1 - c') + T (1 - c) c' + P (1 - c)(1 - c').

Dynamics: 100 strategies drawn uniformly at random, equal initial frequencies,
discrete-time replicator dynamics x_i <- x_i f_i / mean f. Every 100
generations, strategies with frequency below 0.001 are removed and one new
random strategy is added with frequency 0.01 (mutation).
Output: reactive-gtft.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
R, S, T, P = 3, 0, 5, 1
EPS, N0, MUT, SEED, TMAX = 0.001, 100, 100, 65, 4000
SNAP = [0, 20, 500, 1100, 1600, 4000]
BLUE, RED, GRAY = "#2a5caa", "#d62f27", "#333333"


def payoff_matrix(p, q):
    r = p - q
    c = (q[:, None] + r[:, None] * q[None, :]) / (1 - r[:, None] * r[None, :])
    c2 = c.T
    return R * c * c2 + S * c * (1 - c2) + T * (1 - c) * c2 + P * (1 - c) * (1 - c2)


rng = np.random.default_rng(SEED)
p = rng.uniform(EPS, 1 - EPS, N0)
q = rng.uniform(EPS, 1 - EPS, N0)
x = np.ones(N0) / N0
snaps, ts = {0: (p.copy(), q.copy(), x.copy())}, []
for t in range(1, TMAX + 1):
    f = payoff_matrix(p, q) @ x
    x = x * f / (x @ f)
    if t % MUT == 0:
        keep = x > 1e-3
        p, q, x = p[keep], q[keep], x[keep] / x[keep].sum()
        p = np.append(p, rng.uniform(EPS, 1 - EPS))
        q = np.append(q, rng.uniform(EPS, 1 - EPS))
        x = np.append(0.99 * x, 0.01)
    ts.append((x @ p, x @ q, x @ (payoff_matrix(p, q) @ x)))
    if t in SNAP:
        snaps[t] = (p.copy(), q.copy(), x.copy())
ts = np.array(ts)

fig = plt.figure(figsize=(13, 6.6))
gs = fig.add_gridspec(2, len(SNAP), height_ratios=[1, 1.15], hspace=0.45, wspace=0.25)

# (a) snapshots of the population in the (p, q) plane
for k, t in enumerate(SNAP):
    ax = fig.add_subplot(gs[0, k])
    pp, qq, xx = snaps[t]
    ax.scatter(pp, qq, s=1200 * xx, color=BLUE, alpha=0.75, edgecolors="none", clip_on=False)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("$p$", fontsize=13, labelpad=-8)
    if k == 0:
        ax.set_ylabel("$q$", fontsize=13, labelpad=-6)
    else:
        ax.set_yticklabels([])
    ax.set_title(f"$t = {t}$", fontsize=13)
fig.text(0.075, 0.90, "(a)", fontsize=16)

# (b) time series
ax = fig.add_subplot(gs[1, :])
t_ax = np.arange(1, TMAX + 1)
ax.plot(t_ax, ts[:, 2], color=GRAY, lw=1.8, label="average payoff")
ax.plot(t_ax, ts[:, 0], color=BLUE, lw=1.8, label=r"average $p$")
ax.plot(t_ax, ts[:, 1], color=RED, lw=1.8, label=r"average $q$")
for t in SNAP[1:]:
    ax.axvline(t, color="0.7", lw=0.8, ls=":")
ax.set_xlim(0, TMAX); ax.set_ylim(0, 3.1)
ax.set_xlabel("generation $t$", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(loc="center right", fontsize=12, frameon=False)
for u, y, lab in [(450, 1.2, "defectors"), (1100, 2.2, "TFT"), (2800, 2.55, "GTFT")]:
    ax.text(u, y, lab, fontsize=13, ha="center")
fig.text(0.075, 0.495, "(b)", fontsize=16)

fig.savefig(os.path.join(here, "reactive-gtft.png"), dpi=200, bbox_inches="tight")
print("done")
