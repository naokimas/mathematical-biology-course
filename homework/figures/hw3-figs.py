"""
Figures for hw3_MATH463-Fall2026.tex (MATH 463, sections 4 and 5).

  hw3-order-parameter.png : order parameter R for four phase configurations
  hw3-networks.png        : four 4-node contact networks and their lambda_max
  hw3-metapop.png         : a 4-subpopulation metapopulation network
  hw3-bottleneck.png      : phase drift of phi' = mu - sin(phi) for mu slightly > 1
  hw3-triangle.png        : triangle-wave response function and its drift period
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})

# ------------------------------------------------------------ order parameter
configs = [
    ("(a) full synchrony", np.full(6, 0.6)),
    ("(b) equidistant", np.arange(6) * 2 * np.pi / 6),
    ("(c) two equal clusters", np.array([0, 0, 0, np.pi, np.pi, np.pi])),
    ("(d) partial synchrony", np.array([0.1, -0.35, 0.55, 0.0, 1.2, -0.9])),
]
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
th = np.linspace(0, 2 * np.pi, 400)
for ax, (title, ph) in zip(axes, configs):
    ax.plot(np.cos(th), np.sin(th), "-", color="0.75", lw=1.2)
    z = np.mean(np.exp(1j * ph))
    for p in ph:
        ax.annotate("", xy=(np.cos(p), np.sin(p)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.6))
        ax.plot(np.cos(p), np.sin(p), "o", color="tab:blue", ms=7)
    ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.6))
    ax.plot(0, 0, "k+", ms=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{title}\n$R={abs(z):.2f}$", fontsize=11)
fig.tight_layout()
fig.savefig("hw3-order-parameter.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------- networks
def draw(ax, pos, edges, title, lam):
    for i, j in edges:
        ax.plot(*zip(pos[i], pos[j]), "-", color="0.35", lw=1.8, zorder=1)
    for i, (x, y) in pos.items():
        ax.plot(x, y, "o", ms=20, mfc="tab:orange", mec="k", zorder=2)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=10, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    ax.set_title(f"{title}\n" + rf"$\lambda_{{\max}}={lam}$", fontsize=17)

sq = {0: (-0.8, 0.8), 1: (0.8, 0.8), 2: (0.8, -0.8), 3: (-0.8, -0.8)}
line = {0: (-1.2, 0), 1: (-0.4, 0), 2: (0.4, 0), 3: (1.2, 0)}
star = {0: (0, 0), 1: (0, 1.1), 2: (-0.95, -0.55), 3: (0.95, -0.55)}
fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.3))
draw(axes[0], line, [(0, 1), (1, 2), (2, 3)], "(a) chain", r"(1+\sqrt{5})/2\approx1.62")
draw(axes[1], star, [(0, 1), (0, 2), (0, 3)], "(b) star", r"\sqrt{3}\approx1.73")
draw(axes[2], sq, [(0, 1), (1, 2), (2, 3), (3, 0)], "(c) ring", "2")
draw(axes[3], sq, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
     "(d) complete", "3")
fig.subplots_adjust(top=0.74, bottom=0.02, left=0.01, right=0.99)
fig.savefig("hw3-networks.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- metapopulation
fig, ax = plt.subplots(figsize=(5.0, 3.6))
pos = {0: (-1.0, 0.75), 1: (-1.0, -0.75), 2: (0.3, 0.0), 3: (1.6, 0.0)}
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
rad = 0.26
for i, j in edges:                      # draw each edge only outside the two disks
    (x1, y1), (x2, y2) = pos[i], pos[j]
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    ax.plot([x1 + rad * ux, x2 - rad * ux], [y1 + rad * uy, y2 - rad * uy],
            "-", color="black", lw=2.0, zorder=1)
for i, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), rad, facecolor="#bfe3bf", edgecolor="black",
                            lw=1.6, zorder=2))
    ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=19, zorder=3)
ax.set_aspect("equal")
ax.set_xlim(-1.6, 2.1)
ax.set_ylim(-1.25, 1.25)
ax.axis("off")
fig.tight_layout()
fig.savefig("hw3-metapop.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ bottleneck
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
mu = 1.1
ax = axes[0]
p = np.linspace(-np.pi, 2 * np.pi, 600)
ax.plot(p, mu - np.sin(p), "k-", lw=2)
ax.axhline(0, color="0.5", lw=0.9)
ax.fill_between(p, 0, mu - np.sin(p), where=(mu - np.sin(p) < 0.3),
                color="tab:red", alpha=0.25)
ax.set_xticks([-np.pi, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels([r"$-\pi$", "0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\dot\phi=\mu-\sin\phi$")
ax.set_title(rf"$\mu={mu}$: $\dot\phi>0$ everywhere, but tiny near $\phi=\pi/2$")
ax = axes[1]
sol = solve_ivp(lambda t, y: mu - np.sin(y), [0, 60], [0.0], max_step=0.02,
                rtol=1e-10, atol=1e-12)
ax.plot(sol.t, sol.y[0], "-", color="tab:blue", lw=2)
T = 2 * np.pi / np.sqrt(mu**2 - 1)
for k in range(1, 5):
    ax.axhline(np.pi / 2 + 2 * np.pi * (k - 1), color="0.8", lw=0.8)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$\phi(\tau)$")
ax.set_title(rf"drift: one lap takes $T=2\pi/\sqrt{{\mu^2-1}}\approx{T:.1f}$")
fig.tight_layout()
fig.savefig("hw3-bottleneck.png", dpi=200)
plt.close(fig)
print("drift period", T)

# ------------------------------------------------- triangle-wave response (P8)
def tri(p):
    p = np.mod(p + np.pi / 2, 2 * np.pi) - np.pi / 2
    return np.where(p <= np.pi / 2, p, np.pi - p)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axes[0]
pp = np.linspace(-np.pi / 2, 3 * np.pi / 2, 800)
ax.plot(pp, tri(pp), "-", color="tab:blue", lw=2.2, label=r"triangle wave $f(\phi)$")
ax.plot(pp, np.sin(pp), "--", color="tab:red", lw=2, label=r"$\sin\phi$")
ax.axhline(0, color="0.6", lw=0.8)
ax.axhline(np.pi / 2, color="tab:blue", lw=0.8, ls=":")
ax.axhline(1, color="tab:red", lw=0.8, ls=":")
ax.set_xticks([-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
ax.set_yticks([-np.pi / 2, -1, 0, 1, np.pi / 2])
ax.set_yticklabels([r"$-\pi/2$", "$-1$", "0", "1", r"$\pi/2$"])
ax.set_xlabel(r"$\phi$")
ax.set_title("(a) two response functions")
ax.legend(fontsize=9, loc="lower left")
ax = axes[1]
m = np.linspace(np.pi / 2 + 1e-3, 4, 500)
ax.plot(m, 2 * np.log((2 * m + np.pi) / (2 * m - np.pi)), "-", color="tab:blue",
        lw=2.2, label=r"triangle: $2\ln\frac{2\mu+\pi}{2\mu-\pi}$")
m2 = np.linspace(1 + 1e-4, 4, 500)
ax.plot(m2, 2 * np.pi / np.sqrt(m2**2 - 1), "--", color="tab:red", lw=2,
        label=r"sine: $2\pi/\sqrt{\mu^2-1}$")
ax.axvline(1, color="tab:red", lw=0.8, ls=":")
ax.axvline(np.pi / 2, color="tab:blue", lw=0.8, ls=":")
ax.set_ylim(0, 25)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$T_{\rm drift}$")
ax.set_title("(b) drift period above the entrainment range")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("hw3-triangle.png", dpi=200)
plt.close(fig)
