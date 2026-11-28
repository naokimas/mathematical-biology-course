"""
Stochastic rock-paper-scissors (cyclic Lotka-Volterra) simulations:
finite well-mixed population vs. square lattice, same N, same initial state.

This script ONLY generates the trajectories and writes them to
    rps-finite-sims.npz
The plotting is done separately by rps-finite-sims-plot.py, so that the figure
can be restyled without re-running the simulations (which take a few minutes).

Model (identical in both cases, only the interaction neighborhood differs)
-------------------------------------------------------------------------
Every individual is of type R, S, or P, and R beats S, S beats P, P beats R.
One elementary update ("attempt") picks a pair of individuals that are allowed
to interact and, if one of the two types dominates the other, converts the
dominated individual into the dominant type:

    R + S -> R + R,    S + P -> S + S,    P + R -> P + P.

There are no empty sites, no mutation and no mobility, so the population size
is constant and the three absorbing states (all R, all S, all P) are the only
fixed points of the stochastic process.

  * well-mixed ("finite mean-field"): the pair is two distinct individuals
    drawn uniformly at random from the whole population.  As N -> infinity
    this converges to the mean-field ODE of the lecture notes,
    dr/dt = r(s-p), ds/dt = s(p-r), dp/dt = p(r-s), whose orbits are closed.
    At finite N the orbits are not closed: H = rsp performs a random walk with
    a systematic outward drift, so the trajectory spirals out to the boundary
    of the simplex and one type, and soon after a second one, goes extinct.

  * square lattice: the pair is a uniformly random site and one of its four
    nearest neighbors (von Neumann neighborhood, periodic boundary).  The
    local interactions build spatial structure (rotating spirals), which keeps
    the global fractions close to (1/3,1/3,1/3) and maintains coexistence for
    very long times.

Time unit
---------
One "generation" = N attempts in both models, so the two time axes are
directly comparable.  (For the well-mixed model one generation of N attempts
corresponds to two time units of the ODE above: per attempt the expected
change of n_R is 2 r (s-p), hence E[dr] = 2 r (s-p) per generation.)

Output of rps-finite-sims.npz
-----------------------------
    N, L, seed, init_fractions   simulation parameters
    t_mf,  traj_mf               generations and (r,s,p) of the well-mixed run
    t_lat, traj_lat              generations and (r,s,p) of the lattice run
    lattice_final                final L x L configuration (0=R, 1=S, 2=P)

Runtime: a few seconds (pure-Python event loops; N = 900).
"""

import math
import random
import time

import numpy as np

# ----------------------------------------------------------------------
# Parameters.  The two models must share N and the initial fractions.
# ----------------------------------------------------------------------
L = 30                        # lattice side; the well-mixed run uses the same
N = L * L                     # number of individuals ( = number of nodes)
INIT = (0.35, 0.35, 0.30)     # initial fractions of (R, S, P)
SEED = 20261104

GEN_MF = 40000                # cap on the well-mixed run (it stops at extinction)
GEN_LAT = 1000                # length of the lattice run (it does not go extinct)

R, S, P = 0, 1, 2             # type codes; "a beats b" iff (a + 1) % 3 == b

OUTFILE = "rps-finite-sims.npz"


def initial_counts(n, init):
    """Counts (n_R, n_S, n_P) closest to the requested fractions, summing to n."""
    counts = [int(round(f * n)) for f in init]
    counts[0] += n - sum(counts)       # absorb the rounding error into R
    return counts


# ----------------------------------------------------------------------
# 1. Finite well-mixed population.
# ----------------------------------------------------------------------
def simulate_well_mixed(n, init, max_generations, rng):
    """Exact simulation of the well-mixed process; returns (times, fractions).

    An attempt draws an unordered pair of distinct individuals uniformly at
    random, so with C = n(n-1)/2 possible pairs the probability that a given
    attempt converts an individual is

        q = (n_R n_S + n_S n_P + n_P n_R) / C ,

    and the three conversions occur with probabilities proportional to the
    three products.  Attempts that change nothing (equal types) are skipped by
    drawing the number of attempts up to the next conversion from a geometric
    distribution, which is exact and much faster than looping over attempts.

    The state is recorded once per generation (= n attempts).
    """
    n_r, n_s, n_p = initial_counts(n, init)
    pairs = n * (n - 1) / 2.0

    times, traj = [0.0], [(n_r / n, n_s / n, n_p / n)]
    attempts = 0                       # attempts performed so far
    next_record = n                    # record at the end of each generation

    log1p = math.log1p
    log = math.log
    rand = rng.random

    while attempts < max_generations * n:
        w_rs, w_sp, w_pr = n_r * n_s, n_s * n_p, n_p * n_r
        w_tot = w_rs + w_sp + w_pr
        if w_tot == 0:                 # only one type left: absorbing state
            break

        # Attempts until (and including) the next conversion ~ Geometric(q).
        q = w_tot / pairs
        if q >= 1.0:
            skip = 1
        else:
            skip = int(log(rand()) / log1p(-q)) + 1
        attempts += skip

        # Which conversion?
        u = rand() * w_tot
        if u < w_rs:                   # R + S -> R + R
            n_r += 1
            n_s -= 1
        elif u < w_rs + w_sp:          # S + P -> S + S
            n_s += 1
            n_p -= 1
        else:                          # P + R -> P + P
            n_p += 1
            n_r -= 1

        while attempts >= next_record:
            times.append(next_record / n)
            traj.append((n_r / n, n_s / n, n_p / n))
            next_record += n

    # Record the final state (the moment of extinction, if it was reached).
    times.append(attempts / n)
    traj.append((n_r / n, n_s / n, n_p / n))
    return np.array(times), np.array(traj)


# ----------------------------------------------------------------------
# 2. Square lattice, periodic boundary, von Neumann neighborhood.
# ----------------------------------------------------------------------
def neighbor_table(side):
    """Flat list nb[4*i + d] = index of the d-th neighbor of site i."""
    nb = [0] * (4 * side * side)
    for y in range(side):
        for x in range(side):
            i = y * side + x
            nb[4 * i + 0] = y * side + (x + 1) % side          # right
            nb[4 * i + 1] = y * side + (x - 1) % side          # left
            nb[4 * i + 2] = ((y + 1) % side) * side + x        # down
            nb[4 * i + 3] = ((y - 1) % side) * side + x        # up
    return nb


def simulate_lattice(side, init, generations, rng):
    """Random-sequential-update simulation; returns (times, fractions, final state).

    One attempt picks a uniformly random site and one of its four neighbors.
    The same conversion rule as in the well-mixed model is applied, so the two
    models differ only in who is allowed to interact with whom.
    """
    n = side * side
    counts = initial_counts(n, init)

    state = [R] * counts[0] + [S] * counts[1] + [P] * counts[2]
    rng.shuffle(state)                 # random spatial arrangement
    nb = neighbor_table(side)

    n_r, n_s, n_p = counts
    times, traj = [0.0], [(n_r / n, n_s / n, n_p / n)]
    delta = {R: (1, 0, 0), S: (0, 1, 0), P: (0, 0, 1)}

    randrange = rng.randrange
    four_n = 4 * n
    for g in range(generations):
        for _ in range(n):
            k = randrange(four_n)      # site i = k >> 2, direction d = k & 3
            i = k >> 2
            j = nb[k]
            a = state[i]
            b = state[j]
            if a == b:
                continue
            if (a + 1) % 3 == b:       # a dominates b: b is converted into a
                state[j] = a
                winner, loser = a, b
            else:                      # b dominates a: a is converted into b
                state[i] = b
                winner, loser = b, a
            dw, dl = delta[winner], delta[loser]
            n_r += dw[0] - dl[0]
            n_s += dw[1] - dl[1]
            n_p += dw[2] - dl[2]
        times.append(g + 1.0)
        traj.append((n_r / n, n_s / n, n_p / n))

    final = np.array(state, dtype=np.int8).reshape(side, side)
    return np.array(times), np.array(traj), final


# ----------------------------------------------------------------------
def main():
    print(f"N = {N} (lattice {L} x {L}), initial fractions {INIT}, seed {SEED}")

    t0 = time.time()
    rng = random.Random(SEED)
    t_mf, traj_mf = simulate_well_mixed(N, INIT, GEN_MF, rng)
    print(f"  well-mixed: {t_mf[-1]:.0f} generations, "
          f"final (r,s,p) = {tuple(np.round(traj_mf[-1], 4))} "
          f"[{time.time() - t0:.0f} s]")

    t0 = time.time()
    rng = random.Random(SEED + 1)
    t_lat, traj_lat, final = simulate_lattice(L, INIT, GEN_LAT, rng)
    print(f"  lattice:    {t_lat[-1]:.0f} generations, "
          f"final (r,s,p) = {tuple(np.round(traj_lat[-1], 4))} "
          f"[{time.time() - t0:.0f} s]")

    np.savez_compressed(
        OUTFILE,
        N=N, L=L, seed=SEED, init_fractions=np.array(INIT),
        t_mf=t_mf, traj_mf=traj_mf,
        t_lat=t_lat, traj_lat=traj_lat,
        lattice_final=final,
    )
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
