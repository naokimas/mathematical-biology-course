"""
Draws the two simplex panels of the finite-population rock-paper-scissors
figure from the trajectories produced by rps-finite-sims-simulate.py.

    rps-finite-sims.npz  ->  rps-finite-sims.png

The resulting image is placed in the lecture notes immediately to the right of
rps-meanfield.png (the deterministic mean-field orbits), so that the three
triangles form one row.  The panel size, the simplex embedding and the font
sizes are therefore kept identical to those of rps-meanfield.py: R is the
bottom-left corner, S the bottom-right corner, P the apex, and the flow is
clockwise (R -> P -> S -> R).

Left panel (red): the finite well-mixed population.  Its trajectory winds
around the center with a growing amplitude and ends at a corner of the
simplex, that is, in extinction.  The marker at the corner is the end of the
run.  (If N is raised, one turn still takes about six generations while the
number of turns before extinction grows roughly in proportion to N, so for
N much larger than 10^3 the whole run becomes an unreadable tangle and only
an initial part of it should be drawn.)

Right panel (blue): the square lattice with the same N and the same initial
fractions.  It keeps circulating around (1/3,1/3,1/3) without reaching any
corner, that is, the three types coexist.

Nothing is simulated here; rerun rps-finite-sims-simulate.py to change the
data, and this script to change the look of the figure.
"""

import numpy as np
import matplotlib.pyplot as plt

DATA = "rps-finite-sims.npz"
OUT = "rps-finite-sims.png"

COLOR_MF = "#d62728"      # finite well-mixed population
COLOR_LAT = "#1f77b4"     # square lattice


def to_xy(point):
    """Map (r,s,p) with r+s+p=1 to the plane: R bottom-left, S bottom-right, P apex."""
    r, s, p = np.asarray(point, dtype=float).T
    return np.atleast_1d(s + 0.5 * p), np.atleast_1d((np.sqrt(3) / 2) * p)


def draw_simplex(ax):
    """Triangle outline, corner labels and the interior equilibrium."""
    corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    cx, cy = to_xy(corners)
    ax.plot(cx, cy, color="black", lw=1.5)

    ex, ey = to_xy([1 / 3, 1 / 3, 1 / 3])
    ax.plot(ex, ey, "ko", ms=5)

    ax.text(cx[0] - 0.04, cy[0] - 0.04, "R", fontsize=18, ha="right")
    ax.text(cx[1] + 0.04, cy[1] - 0.04, "S", fontsize=18, ha="left")
    ax.text(cx[2], cy[2] + 0.03, "P", fontsize=18, ha="center")

    ax.set_aspect("equal")
    ax.axis("off")


def main():
    data = np.load(DATA)
    n, side = int(data["N"]), int(data["L"])
    traj_mf, traj_lat = data["traj_mf"], data["traj_lat"]
    t_mf, t_lat = data["t_mf"], data["t_lat"]
    print(f"N = {n} ({side} x {side}); "
          f"well-mixed run {t_mf[-1]:.0f} generations, "
          f"lattice run {t_lat[-1]:.0f} generations")

    # Two panels of the same size as the single panel of rps-meanfield.py,
    # so that all three triangles come out equally large in the notes.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # --- left: finite well-mixed population -------------------------------
    ax = axes[0]
    draw_simplex(ax)
    x, y = to_xy(traj_mf)
    ax.plot(x, y, color=COLOR_MF, lw=0.9, alpha=0.9, zorder=2)
    ax.plot(x[-1], y[-1], "o", color=COLOR_MF, ms=8, zorder=4)        # extinction
    ax.plot(x[0], y[0], "o", color="black", ms=6, zorder=5)           # start

    # --- right: square lattice --------------------------------------------
    ax = axes[1]
    draw_simplex(ax)
    x, y = to_xy(traj_lat)
    ax.plot(x, y, color=COLOR_LAT, lw=0.5, alpha=0.6, zorder=2)
    ax.plot(x[0], y[0], "o", color="black", ms=6, zorder=5)           # start

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
