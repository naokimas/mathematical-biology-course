"""
Figures for hw2_MATH463-Fall2026.tex (MATH 463, section 3 self-study problems).

  hw2-competition.png : competition problem, nullclines and flow of two competition models
  hw2-brusselator.png : Brusselator problem, below and above the Hopf bifurcation
  hw2-euler.png       : Euler problem, vs exact solution of x'=y, y'=-x
  hw2-sir.png         : phase plane of x'=-kxy, y'=kxy-ly
  hw2-node-spiral.png : schematic of a stable node and a stable spiral
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})


def mark(ax, pts, kinds):
    for (x, y), k in zip(pts, kinds):
        if k == "s":
            ax.plot(x, y, "o", ms=8, mfc="k", mec="k", zorder=6, clip_on=False)
        else:
            ax.plot(x, y, "o", ms=8, mfc="w", mec="k", mew=1.6, zorder=6, clip_on=False)


# ---------------------------------------------------------------- competition
def comp(c12):
    return lambda t, z: [z[0] * (3 - (2 if c12 == 1 else 1) * z[0] - c12 * z[1]),
                         z[1] * (2 - z[0] - z[1])]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
cases = [
    (1, "(a) $x'=x(3-2x-y)$, $y'=y(2-x-y)$",
     [(0, 0), (1.5, 0), (0, 2), (1, 1)], ["u", "u", "u", "s"], 1.5),
    (2, "(b) $x'=x(3-x-2y)$, $y'=y(2-x-y)$",
     [(0, 0), (3, 0), (0, 2), (1, 1)], ["u", "s", "s", "u"], 3.0),
]
for ax, (c12, title, pts, kinds, xint) in zip(axes, cases):
    f = comp(c12)
    X, Y = np.meshgrid(np.linspace(0.05, 3.3, 19), np.linspace(0.05, 2.6, 15))
    U, V = f(0, [X, Y])
    S = np.hypot(U, V) + 1e-9
    ax.quiver(X, Y, U / S, V / S, color="0.7", scale=32, width=0.003)
    xs = np.linspace(0, 3.4, 100)
    xnull = 3 - 2 * xs if c12 == 1 else (3 - xs) / 2
    ax.plot(xs, xnull, color="tab:blue", lw=2, label=r"$x$-nullcline")
    ax.plot(xs, 2 - xs, color="tab:red", lw=2, label=r"$y$-nullcline")
    ax.axvline(0, color="tab:blue", lw=3)
    ax.axhline(0, color="tab:red", lw=3)
    for x0, y0 in [(0.1, 2.5), (0.3, 0.1), (3.2, 0.3), (3.2, 2.4), (0.05, 0.6),
                   (1.6, 2.5), (2.5, 0.05)]:
        sol = solve_ivp(f, [0, 30], [x0, y0], max_step=0.02)
        ax.plot(sol.y[0], sol.y[1], "k-", lw=1)
    mark(ax, pts, kinds)
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 2.7)
    ax.set_xlabel("$x$ (rabbits)")
    ax.set_ylabel("$y$ (sheep)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig("hw2-competition.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- brusselator
a = 1.0
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
for j, b in enumerate([1.5, 2.5]):
    f = lambda t, z, b=b: [1 - (b + 1) * z[0] + a * z[0] ** 2 * z[1],
                           b * z[0] - a * z[0] ** 2 * z[1]]
    sol = solve_ivp(f, [0, 60], [0.5, 1.0], max_step=0.01, rtol=1e-9, atol=1e-9)
    ax = axes[0, j]
    ax.plot(sol.y[0], sol.y[1], "k-", lw=1)
    mark(ax, [(1, b / a)], ["s" if b < 1 + a else "u"])
    ax.plot(0.5, 1.0, "s", color="tab:red", ms=6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(rf"$a=1$, $b={b}$ ($b_c=2$): phase plane")
    ax = axes[1, j]
    ax.plot(sol.t, sol.y[0], lw=1.4, label="$x(t)$")
    ax.plot(sol.t, sol.y[1], lw=1.4, label="$y(t)$")
    ax.set_xlabel("$t$")
    ax.set_title(rf"$a=1$, $b={b}$: time series")
    ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("hw2-brusselator.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------- euler
dt, T = 0.1, 2 * np.pi
n = int(round(T / dt))
x, y = np.empty(n + 1), np.empty(n + 1)
x[0], y[0] = 1.0, 0.0
for k in range(n):
    x[k + 1] = x[k] + dt * y[k]
    y[k + 1] = y[k] - dt * x[k]
th = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot(np.cos(th), -np.sin(th), "k-", lw=2, label="exact solution (circle)")
ax.plot(x, y, "o-", color="tab:red", ms=2.5, lw=1,
        label=rf"Euler, $\Delta t={dt}$, {n} steps")
ax.plot(1, 0, "ks", ms=6)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(r"$x'=y,\ y'=-x$, from $(1,0)$ over $0\leq t\leq 2\pi$")
ax.legend(fontsize=9, loc="center")
fig.tight_layout()
fig.savefig("hw2-euler.png", dpi=200)
plt.close(fig)
print("Euler final radius:", np.hypot(x[-1], y[-1]))

# period of Brusselator just above b_c
f = lambda t, z: [1 - (2.05 + 1) * z[0] + z[0] ** 2 * z[1], 2.05 * z[0] - z[0] ** 2 * z[1]]
sol = solve_ivp(f, [0, 400], [1.1, 2.0], max_step=0.01, rtol=1e-9, atol=1e-9)
t, xx = sol.t, sol.y[0]
m = t > 300
up = np.where((xx[m][:-1] < 1) & (xx[m][1:] >= 1))[0]
print("period near b_c:", np.diff(t[m][up]).mean(), "2pi =", 2 * np.pi)

# ------------------------------------------------------------------- SIR
k, l = 2.0, 1.0
fig, ax = plt.subplots(figsize=(5.8, 5.0))
f = lambda t, z: [-k * z[0] * z[1], k * z[0] * z[1] - l * z[1]]
for x0 in [0.3, 0.45, 0.6, 0.75, 0.9, 0.99]:
    sol = solve_ivp(f, [0, 80], [x0, 1 - x0 if x0 < 0.99 else 0.01],
                    max_step=0.02, rtol=1e-9, atol=1e-11)
    ax.plot(sol.y[0], sol.y[1], "k-", lw=1.3)
    ax.plot(sol.y[0][0], sol.y[1][0], "s", color="tab:red", ms=4)
    ax.annotate("", xy=(sol.y[0][60], sol.y[1][60]), xytext=(sol.y[0][40], sol.y[1][40]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
ax.plot([0, 1], [1, 0], ":", color="0.5", lw=1)
ax.axvline(l / k, color="tab:blue", ls="--", lw=1.5, label=r"$x=\ell/k$ (threshold)")
xs = np.linspace(0, 1, 200)
ax.plot(xs[xs < l / k], 0 * xs[xs < l / k], color="tab:green", lw=4, alpha=0.6,
        label=r"stable part of the line of equilibria")
ax.plot(xs[xs > l / k], 0 * xs[xs > l / k], color="tab:orange", lw=4, alpha=0.6,
        label=r"unstable part of the line of equilibria")
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.02, 0.75)
ax.set_xlabel("$x$ (susceptible)")
ax.set_ylabel("$y$ (infected)")
ax.set_title(r"$x'=-kxy$, $y'=kxy-\ell y$ with $k=2$, $\ell=1$")
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
fig.tight_layout()
fig.savefig("hw2-sir.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------ node vs spiral
from scipy.linalg import expm
fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4),
                         gridspec_kw={"height_ratios": [1.6, 1]})
mats = [(np.array([[-2.0, 1.0], [1.0, -2.0]]),
         "stable node\n(real eigenvalues $-1$, $-3$)"),
        (np.array([[-0.3, -1.0], [1.0, -0.3]]),
         "stable spiral\n(complex eigenvalues $-0.3\\pm i$)")]
ts = np.linspace(0, 14, 1400)
for j, (A, title) in enumerate(mats):
    ax = axes[0, j]
    for ang in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.2:
        z0 = np.array([np.cos(ang), np.sin(ang)])
        Z = np.array([expm(A * t) @ z0 for t in ts])
        ax.plot(Z[:, 0], Z[:, 1], "k-", lw=1)
        i0, i1 = 25, 45
        ax.annotate("", xy=Z[i1], xytext=Z[i0],
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))
    ax.plot(0, 0, "o", ms=9, mfc="tab:red", mec="k", zorder=5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title, fontsize=11)
    ax = axes[1, j]
    z0 = np.array([np.cos(0.2), np.sin(0.2)])
    Z = np.array([expm(A * t) @ z0 for t in ts])
    ax.plot(ts, Z[:, 0], lw=1.6)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    ax.set_yticks([0])
    ax.set_title("no oscillation" if j == 0 else "damped oscillation", fontsize=11)
fig.tight_layout()
fig.savefig("hw2-node-spiral.png", dpi=200)
plt.close(fig)
