"""
Mean-field (well-mixed) dynamics of a rock-paper-scissors interacting particle
system (cyclic Lotka-Volterra).

Each individual is of type R, S, or P. Pairs meet at random and the dominant
type converts the other (invasion / replacement reactions):

    R + S -> R + R,    S + P -> S + S,    P + R -> P + P.

Mass-action mean-field ODEs for the fractions r, s, p (r+s+p=1):

    dr/dt = r (s - p),
    ds/dt = s (p - r),
    dp/dt = p (r - s).

The interior fixed point (1/3,1/3,1/3) is a *neutrally stable center*:
H = r s p is conserved, so trajectories are closed orbits. (This is the same
ODE as the replicator dynamics of the symmetric zero-sum RPS game.)
"""
import numpy as np
import matplotlib.pyplot as plt

def rhs(x):
    r, s, p = x
    return np.array([r * (s - p), s * (p - r), p * (r - s)])

def rk4(x, dt):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    x = np.clip(x, 0, None)
    return x / x.sum()

# Simplex -> 2D (R bottom-left, S bottom-right, P top).
def to_xy(p):
    R, S, P = p
    return S + 0.5 * P, (np.sqrt(3) / 2) * P

fig, ax = plt.subplots(figsize=(6, 5.4))

corners = np.array([to_xy([1, 0, 0]), to_xy([0, 1, 0]), to_xy([0, 0, 1])])
tri = np.vstack([corners, corners[0]])
ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.5)

center = np.array([1/3, 1/3, 1/3])
orbits = {}
for s in [0.12, 0.30, 0.55, 0.85]:
    x = center + s * (np.array([1.0, 0.0, 0.0]) - center)
    x = x / x.sum()
    traj = [to_xy(x)]
    for _ in range(60000):
        x = rk4(x, 0.005)
        traj.append(to_xy(x))
    traj = np.array(traj)
    orbits[s] = traj
    ax.plot(traj[:, 0], traj[:, 1], lw=1.3)

cx, cy = to_xy(center)
ax.plot(cx, cy, "ko", ms=6)
ax.text(cx + 0.02, cy, r"$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$", fontsize=13, va="center")
ax.text(corners[0, 0] - 0.04, corners[0, 1] - 0.04, "R", fontsize=18, ha="right")
ax.text(corners[1, 0] + 0.04, corners[1, 1] - 0.04, "S", fontsize=18, ha="left")
ax.text(corners[2, 0], corners[2, 1] + 0.03, "P", fontsize=18, ha="center")

# Direction-of-flow arrow placed ON one of the closed orbits, following the
# actual (forward-time) integration so the direction is guaranteed correct.
# The flow is clockwise in this embedding (R bottom-left, S bottom-right, P top):
# the abundance peak cycles R -> P -> S -> R.
traj = orbits[0.55]
# choose the point of maximum x (rightmost point of the orbit) and draw a short
# arrow to the next integrated point.
i0 = int(np.argmax(traj[:, 0]))
i1 = i0 + 300
ax.annotate("", xy=tuple(traj[i1]), xytext=tuple(traj[i0]),
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=2.0,
                            mutation_scale=22))

ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.savefig("rps-meanfield.png", dpi=200, bbox_inches="tight")
print("wrote rps-meanfield.png")
