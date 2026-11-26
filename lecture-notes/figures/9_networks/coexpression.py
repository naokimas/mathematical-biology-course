"""
Figure for section 9 (network biology): what a co-expression edge means.

Two scatter plots of the expression of two genes across samples:
left, strongly correlated (an edge in a co-expression network);
right, essentially uncorrelated (no edge).

Schematic data (not from a real experiment).
Produces: coexpression.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})
rng = np.random.default_rng(7)

n = 120
mean, cov_hi, cov_lo = [3.0, 2.6], 0.90, 0.04


def sample(rho):
    C = np.array([[0.8**2, rho * 0.8 * 0.7], [rho * 0.8 * 0.7, 0.7**2]])
    xy = rng.multivariate_normal(mean, C, size=n)
    return np.clip(xy[:, 0], 0, None), np.clip(xy[:, 1], 0, None)


fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True)
for ax, rho, title, color in [
    (axes[0], cov_hi, "high co-expression: an edge", "tab:red"),
    (axes[1], cov_lo, "low co-expression: no edge", "tab:blue"),
]:
    x, y = sample(rho)
    r = np.corrcoef(x, y)[0, 1]
    ax.plot(x, y, "o", ms=6, color=color, alpha=0.65, mec="none")
    ax.set_xlabel(r"$\log(\mathrm{TPM}+1)$ of gene $g_1$")
    ax.set_title(f"{title}\n(correlation coefficient $r={r:.2f}$)", fontsize=11)
    ax.set_xlim(0.3, 5.5)
    ax.set_ylim(0.3, 5.2)
axes[0].set_ylabel(r"$\log(\mathrm{TPM}+1)$ of gene $g_2$")
fig.tight_layout()
fig.savefig("coexpression.png", dpi=200)
print("saved")
