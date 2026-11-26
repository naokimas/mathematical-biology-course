"""
Figures for qa-section2-revised.tex (MATH 463, section 2 self-study Q&A).

  fig-gene-switch.png   : autoactivating gene, x' = x^2/(1+x^2) - k x
  fig-language.png      : Abrams-Strogatz language competition with a = 2,
                          x' = x(1-x)(x-(1-s))
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})


def arrows(ax, xs, f, y=0.0, d0=0.13):
    """Draw phase-line arrows along y = const, direction given by sign of f."""
    for x in xs:
        v = f(x)
        if abs(v) < 1e-12:
            continue
        d = d0 * np.sign(v)
        ax.annotate("", xy=(x + d, y), xytext=(x - d, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.25"))


def dots(ax, pts, stable, y=0.0):
    for p, s in zip(pts, stable):
        if s == "s":
            ax.plot(p, y, "o", ms=8, mfc="k", mec="k", zorder=5)
        elif s == "u":
            ax.plot(p, y, "o", ms=8, mfc="w", mec="k", mew=1.6, zorder=5)
        else:  # half-stable
            ax.plot(p, y, "o", ms=8, mfc="0.6", mec="k", mew=1.6, zorder=5)


# ---------------------------------------------------------------- gene switch
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

x = np.linspace(0, 3.2, 800)
prod = x**2 / (1 + x**2)

ax = axes[0]
ax.plot(x, prod, "k-", lw=2, label=r"production $\dfrac{x^2}{1+x^2}$")
for k, c in zip([0.35, 0.5, 0.75], ["tab:blue", "tab:orange", "tab:red"]):
    ax.plot(x, k * x, "--", lw=1.8, color=c, label=rf"$kx$, $k={k}$")
ax.set_xlim(0, 3.2)
ax.set_ylim(0, 1.15)
ax.set_xlabel(r"$x$")
ax.set_title("(a) production vs. degradation")
ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

ax = axes[1]
for k, c in zip([0.35, 0.5, 0.75], ["tab:blue", "tab:orange", "tab:red"]):
    ax.plot(x, prod - k * x, "-", lw=1.8, color=c, label=rf"$k={k}$")
ax.axhline(0, color="0.4", lw=0.9)
k = 0.35
f = lambda z: z**2 / (1 + z**2) - k * z
xm = (1 - np.sqrt(1 - 4 * k**2)) / (2 * k)
xp = (1 + np.sqrt(1 - 4 * k**2)) / (2 * k)
arrows(ax, [0.26, 0.5 * (xm + xp), 0.5 * (xp + 3.2)], f, y=0.0)
dots(ax, [0.0, xm, xp], ["s", "u", "s"])
ax.set_xlim(0, 3.2)
ax.set_ylim(-0.75, 0.35)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$x'$")
ax.set_title(r"(b) $x'=\frac{x^2}{1+x^2}-kx$; phase line for $k=0.35$")
ax.legend(fontsize=9, loc="lower left", framealpha=0.95)

fig.tight_layout()
fig.savefig("fig-gene-switch.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------- language
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.linspace(0, 1, 600)
for s, c in zip([0.3, 0.5, 0.7], ["tab:blue", "tab:orange", "tab:red"]):
    ax.plot(x, x * (1 - x) * (x - (1 - s)), "-", lw=1.9, color=c,
            label=rf"$s={s}$  ($x^*=1-s={1-s:.1f}$)")
ax.axhline(0, color="0.4", lw=0.9)
s = 0.3
f = lambda z: z * (1 - z) * (z - (1 - s))
arrows(ax, [0.32, 0.85], f, y=0.0, d0=0.05)
dots(ax, [0.0, 1 - s, 1.0], ["s", "u", "s"])
ax.set_xlim(0, 1)
ax.set_ylim(-0.09, 0.09)
ax.set_xlabel(r"$x$ (fraction speaking language X)")
ax.set_ylabel(r"$x'$")
ax.set_title(r"$x'=x(1-x)\,[x-(1-s)]$; phase line for $s=0.3$")
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig("fig-language.png", dpi=200)
plt.close(fig)

print("done")
