"""Figures for HW4 of MATH 463 (Fall 2026)."""
import numpy as np
import matplotlib.pyplot as plt

# ---- Problem 6(e): stimulus S as a function of the steady state y* ----
# Two-enzyme (kinase/phosphatase) switch with positive feedback.
# Steady states satisfy  S = (Vmax2/Vmax1) * y(J1+1-y) / [(1-y)(J2+y)] - y.
J1 = J2 = 0.05
ratio = 1.0                      # Vmax2 / Vmax1

y = np.linspace(1e-4, 0.9965, 4000)
S = ratio * y * (J1 + 1 - y) / ((1 - y) * (J2 + y)) - y

fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(y, S, color="C0", lw=2)
ax.axhline(0.5, color="0.35", ls="--", lw=1.2)
ax.text(0.63, 0.515, r"$S=0.5$", color="0.25", fontsize=11, va="bottom")
ax.set_xlabel(r"$y^*$", fontsize=13)
ax.set_ylabel(r"$S$", fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=11)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("hw4-gk-feedback.png", dpi=200, bbox_inches="tight")
print("wrote hw4-gk-feedback.png")
