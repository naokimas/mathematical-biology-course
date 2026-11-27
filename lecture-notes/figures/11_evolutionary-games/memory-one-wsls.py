"""Evolution of memory-one strategies in the iterated prisoner's dilemma:
win-stay lose-shift (WSLS) eventually takes over.

Own simulation, after Fig. 1 of Nowak & Sigmund, Nature 364, 56 (1993).

Strategy (p1, p2, p3, p4): probability of cooperating after the previous
round gave the player R, T, S, P, respectively; each p_i in [0.001, 0.999].
Payoffs R = 3, T = 5, S = 0, P = 1. The payoff of the infinitely repeated game
is the payoff averaged over the stationary distribution of the four-state
Markov chain (CC, DC, CD, DD) defined by the two strategies.
WSLS = (1, 0, 0, 1).

Dynamics: start from one random strategy; discrete-time replicator dynamics
x_i <- x_i f_i / mean f. Every 100 generations, strategies with frequency
below 0.001 are removed and one new random strategy is added with frequency
0.01 (mutation). The p_i of a new strategy are drawn from the U-shaped
Beta(1/2, 1/2) distribution, which favors nearly deterministic strategies.
Output: memory-one-wsls.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
PAY = np.array([3.0, 5.0, 0.0, 1.0])  # states from own view: R (CC), T (DC), S (CD), P (DD)
SWAP = [0, 2, 1, 3]                    # the same state seen by the opponent
EPS, MUT, SEED, TMAX, REC = 0.001, 100, 31, 500000, 100
BLUE, GRAY = "#2a5caa", "#333333"


def state(own, opp):  # 1 = cooperate
    return {(1, 1): 0, (0, 1): 1, (1, 0): 2, (0, 0): 3}[(own, opp)]


def payoff(s1, s2):
    """Long-run payoff per round to s1 against s2."""
    M = np.zeros((4, 4))
    for k in range(4):
        c1, c2 = s1[k], s2[SWAP[k]]
        for a in (0, 1):
            for b in (0, 1):
                M[k, state(a, b)] += (c1 if a else 1 - c1) * (c2 if b else 1 - c2)
    w, v = np.linalg.eig(M.T)
    v = np.real(v[:, np.argmin(abs(w - 1))])
    return (v / v.sum()) @ PAY


rng = np.random.default_rng(SEED)
new_strategy = lambda: np.clip(rng.beta(0.5, 0.5, 4), EPS, 1 - EPS)
S = [new_strategy()]
x = np.array([1.0])
A = np.array([[payoff(S[0], S[0])]])
rec = []
for t in range(1, TMAX + 1):
    f = A @ x
    x = x * f / (x @ f)
    if t % MUT == 0:
        keep = x > 1e-3
        S = [s for s, k in zip(S, keep) if k]
        A, x = A[np.ix_(keep, keep)], x[keep] / x[keep].sum()
        s, n = new_strategy(), len(S)
        A2 = np.zeros((n + 1, n + 1)); A2[:n, :n] = A
        A2[n, :n] = [payoff(s, u) for u in S]
        A2[:n, n] = [payoff(u, s) for u in S]
        A2[n, n] = payoff(s, s)
        A, x = A2, np.append(0.99 * x, 0.01)
        S.append(s)
    if t % REC == 0:
        rec.append([x @ (A @ x), len(S), *(x @ np.array(S))])
rec = np.array(rec)
t_ax = np.arange(1, len(rec) + 1) * REC / 1e5

rec = np.delete(rec, 1, axis=1)  # drop the number of strategies
labels = ["average\npayoff", "$p_1$ (R)", "$p_2$ (T)", "$p_3$ (S)", "$p_4$ (P)"]
fig, axes = plt.subplots(5, 1, figsize=(12, 7.5), sharex=True)
for k, ax in enumerate(axes):
    ax.plot(t_ax, rec[:, k], color=GRAY if k == 0 else BLUE, lw=1.1)
    ax.set_ylabel(labels[k], fontsize=12, rotation=0, ha="right", va="center")
    ax.tick_params(labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if k >= 1:
        ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 1])
axes[0].set_ylim(0.9, 3.1); axes[0].set_yticks([1, 2, 3])
for k, v in zip(range(1, 5), [1, 0, 0, 1]):
    axes[k].text(t_ax[-1] + 0.05, v, str(v), color=BLUE, fontsize=12, va="center")
axes[-1].set_xlim(0, t_ax[-1])
axes[-1].set_xlabel("generation (in units of 100,000)", fontsize=12)
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig(os.path.join(here, "memory-one-wsls.png"), dpi=200, bbox_inches="tight")
print("done")
