"""Data-generating processes for Extension 2 (synthetic only).

Every random draw comes from an explicit np.random.Generator built from (path seed, stream id), so a
path is reproducible on its own and independent of every other path and stream:

    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))

Streams:
    1  step-level draws: Z_t ~ N(0, 1) and V_t ~ chi2(3), t = 0..T-1 (used by outcomes a and c)
    2  fine-grid draws for outcome b: e_{t,k} ~ N(0, 1), k = 0..K-1, and V_t ~ chi2(3)
    3  outcome b, jump-within-step DGP: index of the substep that carries the step's shock
    4  Thompson Sampling posterior draws (outcome c)

Return model. "Bounded" (primary): x is drawn as a LOG return and the simple return is r = exp(x) - 1,
so r > -1 always. "Unbounded" (thesis, comparison only): the same draw x is used directly as the simple
return r = x.

Regimes (log-return parameters, per step):
    bull: x = 0.08 + 0.15 Z                           (Gaussian)
    shock(mu, s): x = mu + s * Z * sqrt(3 / V)        (Student-t, 3 degrees of freedom, location mu, scale s)
The thesis bear is shock(-0.10, 0.30).
"""
import numpy as np

T_STEPS = 250
BREAK_T = 125
DF = 3.0
BULL_MU, BULL_SIG = 0.08, 0.15
THESIS_BEAR = (-0.10, 0.30)

# Outcome a shock grid: name -> (location, scale) of the Student-t(3) log return during the shock
SHOCKS = {'small': (0.00, 0.20), 'thesis': (-0.10, 0.30), 'large': (-0.20, 0.45)}
DURATIONS = (5, 20, 125)            # shock lasts steps [125, 125 + D); D = 125 runs to the end

EXPLORATORY_SEEDS = np.arange(0, 1000)
CONFIRMATORY_SEEDS = np.arange(100000, 102000)


def path_rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def step_draws(seeds, T=T_STEPS):
    """Z (n, T) standard normals and V (n, T) chi2(3) from stream 1."""
    n = len(seeds)
    Z = np.empty((n, T))
    V = np.empty((n, T))
    for i, s in enumerate(seeds):
        g = path_rng(s, 1)
        Z[i] = g.standard_normal(T)
        V[i] = g.chisquare(DF, T)
    return Z, V


def shock_log_returns(Z, V, shock=THESIS_BEAR, duration=T_STEPS - BREAK_T, break_t=BREAK_T):
    """Log returns x (n, T): bull everywhere except the shock window [break_t, break_t + duration)."""
    x = BULL_MU + BULL_SIG * Z
    mu, s = shock
    end = min(Z.shape[1], break_t + duration)
    x[:, break_t:end] = mu + s * Z[:, break_t:end] * np.sqrt(DF / V[:, break_t:end])
    return x


def stationary_log_returns(Z):
    return BULL_MU + BULL_SIG * Z


def drift_log_returns(Z):
    """Thesis Drift scenario: mean falls linearly from 0.08 to -0.05 over T steps, sd 0.15 (Gaussian)."""
    T = Z.shape[1]
    mu_path = np.linspace(0.08, -0.05, T)
    return mu_path[None, :] + 0.15 * Z


def to_simple(x, model):
    if model == 'bounded':
        return np.expm1(x)
    if model == 'unbounded':
        return x.copy()
    raise ValueError(model)


# ----------------------------------------------------------------------------------------------
# Outcome b: fine-grid paths (K substeps per step) with the per-step log return distribution fixed
# ----------------------------------------------------------------------------------------------
K_FINE = 50


def fine_log_increments(seeds, T=T_STEPS, K=K_FINE, break_t=BREAK_T, shock=THESIS_BEAR,
                        within='diffusive'):
    """Fine log increments y (n, T, K) whose per-step sums have exactly the step distribution.

    Step t's standard normal is Z_t = sum_k e_{t,k} / sqrt(K) with e iid N(0, 1), so Z_t ~ N(0, 1).
    within = 'diffusive' (primary): y_{t,k} = mu_t / K + s_t * c_t * e_{t,k} / sqrt(K), where c_t = 1
        for bull steps and c_t = sqrt(3 / V_t) for shock steps (Student-t as a normal variance mixture:
        given V_t, the step is a Gaussian random walk; sum_k y_{t,k} = mu_t + s_t c_t Z_t exactly).
    within = 'jump' (sensitivity): bull steps as above; in a shock step the drift mu_t / K is spread
        evenly and the whole shock s_t c_t Z_t arrives at one substep chosen uniformly (stream 3).
    """
    n = len(seeds)
    y = np.empty((n, T, K))
    mu_b, s_b = shock
    for i, s in enumerate(seeds):
        g = path_rng(s, 2)
        e = g.standard_normal((T, K))
        V = g.chisquare(DF, T)
        Zt = e.sum(axis=1) / np.sqrt(K)
        mu = np.full(T, BULL_MU)
        sc = np.full(T, BULL_SIG)
        mu[break_t:] = mu_b
        sc[break_t:] = s_b * np.sqrt(DF / V[break_t:])
        y[i] = mu[:, None] / K + sc[:, None] * e / np.sqrt(K)
        if within == 'jump':
            j = path_rng(s, 3).integers(0, K, size=T)
            for t in range(break_t, T):
                y[i, t, :] = mu[t] / K
                y[i, t, j[t]] += sc[t] * Zt[t]
        elif within != 'diffusive':
            raise ValueError(within)
    return y


def fine_gaussian_increments(seeds, T, K, mu, sig, stream=2):
    """Pure Gaussian fine path (tests): per-step log return N(mu, sig^2), K substeps."""
    n = len(seeds)
    y = np.empty((n, T, K))
    for i, s in enumerate(seeds):
        e = path_rng(s, stream).standard_normal((T, K))
        y[i] = mu / K + sig * e / np.sqrt(K)
    return y
