"""
Early-warning signals (critical slowing down) before a fold (saddle-node)
bifurcation, illustrated with the classic grazing / harvesting model

    dx/dt = x (1 - x/K) - c(t) * x^2 / (x^2 + 1) + sigma * noise.

The term c(t) (grazing pressure) is increased slowly. The upper ("vegetated")
stable equilibrium loses stability at a fold and the system collapses to a
low state. As the fold is approached, the recovery rate -> 0 (critical slowing
down), so the fluctuations of x show rising variance and rising lag-1
autocorrelation (-> 1). We plot the state and the two rolling indicators.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

K = 10.0
sigma = 0.18          # noise amplitude
dt = 0.01
T = 2000.0
n = int(T / dt)

c0, c1 = 1.0, 2.6     # grazing increases slowly through the fold
c = c0 + (c1 - c0) * np.arange(n) / n

x = 8.0               # start on the upper (vegetated) branch
xs = np.empty(n)
for i in range(n):
    drift = x * (1 - x / K) - c[i] * x**2 / (x**2 + 1)
    x = x + drift * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    x = max(x, 0.0)
    xs[i] = x

# Sample the series (e.g. once per unit time) for the "monitoring" time series.
step = int(1.0 / dt)
t_obs = np.arange(n)[::step] * dt
x_obs = xs[::step]
c_obs = c[::step]

# Detect collapse (state drops below 2) to restrict indicator computation to
# the pre-transition window (as one must do with real monitoring data).
collapse = np.argmax(x_obs < 2.0) if np.any(x_obs < 2.0) else len(x_obs)
pre = slice(0, max(collapse - 2, 10))

# Rolling-window variance and lag-1 autocorrelation on the (detrended) series.
win = 60
xv = x_obs[pre]
tv = t_obs[pre]
cv = c_obs[pre]
var = np.full(len(xv), np.nan)
ac1 = np.full(len(xv), np.nan)
for i in range(win, len(xv)):
    w = xv[i - win:i]
    w = w - np.polyval(np.polyfit(np.arange(win), w, 1), np.arange(win))  # detrend
    var[i] = np.var(w)
    ac1[i] = np.corrcoef(w[:-1], w[1:])[0, 1]

fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

ax[0].plot(t_obs, x_obs, color="tab:green", lw=0.8)
ax[0].axvline(t_obs[collapse - 1] if collapse < len(x_obs) else t_obs[-1],
              color="gray", ls=":", lw=1)
ax[0].set_ylabel("state $x$")
ax[0].set_title("Approaching a fold bifurcation (grazing $c$ increases slowly)")

ax[1].plot(tv, var, color="tab:blue", lw=1.5)
ax[1].set_ylabel("variance")

ax[2].plot(tv, ac1, color="tab:red", lw=1.5)
ax[2].axhline(1.0, color="gray", ls=":", lw=1)
ax[2].set_ylabel("lag-1 autocorr.")
ax[2].set_xlabel("time $t$  (grazing pressure $c$ increases with $t$)")
ax[2].set_ylim(-0.2, 1.1)

plt.tight_layout()
plt.savefig("early-warning-signals.png", dpi=200, bbox_inches="tight")
print("wrote early-warning-signals.png")
