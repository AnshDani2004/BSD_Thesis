"""Online regime / changepoint detectors for Extension 2 (outcome a).

Every detector reads the per-step log returns x_0, ..., x_{T-1} of one path and, after observing x_t,
outputs a real score s_t (larger = more evidence of the bad regime / a change). It never reads x_{t+1}
or later (tested in tests/test_detectors.py). An alarm is s_t > h. The thesis-native threshold is h = 0.

Detectors (names used everywhere):
  hmm_ret      2-state forward filter, Gaussian emissions N(0.08, 0.15) and N(-0.10, 0.30) on x.
  volhmm       the thesis Vol-HMM (simulation/hmm_refined.py) without CUSUM: emission = Gaussian(x) x
               Gamma(v; 2, 0.075) for bull and Gamma(v; 4, 0.15) for bear, v = population std of the
               last 5 x (0.05 if fewer than 2 observations) + 1e-6.
  volhmm_cusum the thesis Vol-HMM with its CUSUM: S = max(0, S - (x - 0.08)/0.15 - 0.5); once S > 3 the
               trigger latches for the rest of the path and adds +2 to the bear log-emission.
               For the three HMMs: A_ii on the diagonal of the 2x2 transition matrix, uniform start, and
               s_t = log P(bear | x_0..t) - log P(bull | x_0..t).
  bocpd_rl     Adams-MacKay (2007) BOCPD, constant hazard H = 1 - A_ii, Normal-Gamma prior
               (mu0 = 0, kappa0 = 1, alpha0 = 2, beta0 = 0.04) giving a Student-t predictive; no run-length
               truncation. s_t = log P(run length <= 10) - log P(run length > 10), where run length counts
               the observations of the current segment including x_t ("a change in the last 10 steps").
  bocpd_mean   same filter; s_t = - E[segment mean | x_0..t] = - sum_l P(l) mu_l (state-like score).
  jm           online statistical jump model (Bemporad et al. 2018; Nystrup et al. 2020), refitted at each t
               on the last W = 60 observations (fewer at the start) of the features (x_t, rolling 5-step
               std of x), each z-scored within the window. K = 2 states, squared Euclidean loss, jump
               penalty lambda = 2 log(A_ii / (1 - A_ii)) (the Viterbi penalty of an HMM with unit-variance
               emissions). Fit: median split of the return feature, then alternate (dynamic programme,
               centroid update) up to 10 times or until the states stop changing. The bear state is the
               state whose return-feature centroid is lower. s_t = V_t(bull) - V_t(bear), the difference of
               the optimal costs of state sequences ending in each state.
"""
import math
import numpy as np
from numba import njit, prange

LOG2PI = math.log(2.0 * math.pi)
DETECTORS = ('hmm_ret', 'volhmm', 'volhmm_cusum', 'bocpd_rl', 'bocpd_mean', 'jm')

BULL = (0.08, 0.15, 2.0, 0.15 / 2)     # mu, sigma, vol_shape, vol_scale (create_vol_augmented_hmm_kelly)
BEAR = (-0.10, 0.30, 4.0, 0.30 / 2)
BOCPD_PRIOR = (0.0, 1.0, 2.0, 0.04)    # mu0, kappa0, alpha0, beta0
BOCPD_L = 10
JM_WINDOW = 60
JM_MAXIT = 10
BIG = 1e9


@njit(cache=False)
def _norm_lpdf(x, mu, sig):
    z = (x - mu) / sig
    return -0.5 * z * z - math.log(sig) - 0.5 * LOG2PI


@njit(cache=False)
def _gamma_lpdf(v, a, scale):
    if v <= 0.0:
        return -np.inf
    y = v / scale
    return (a - 1.0) * math.log(y) - y - math.lgamma(a) - math.log(scale)


@njit(cache=False)
def _lse2(a, b):
    m = max(a, b)
    if m == -np.inf:
        return -np.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=False)
def _hmm_path(x, aii, use_vol, use_cusum, out):
    T = x.shape[0]
    lstay = math.log(aii)
    lmove = math.log(1.0 - aii)
    lb0 = math.log(0.5)
    lb1 = math.log(0.5)
    S = 0.0
    trig = False
    for t in range(T):
        r = x[t]
        e0 = _norm_lpdf(r, BULL[0], BULL[1])
        e1 = _norm_lpdf(r, BEAR[0], BEAR[1])
        if use_vol:
            if t + 1 < 2:
                v = 0.05
            else:
                lo = max(0, t + 1 - 5)
                m = 0.0
                for k in range(lo, t + 1):
                    m += x[k]
                m /= (t + 1 - lo)
                ss = 0.0
                for k in range(lo, t + 1):
                    ss += (x[k] - m) ** 2
                v = math.sqrt(ss / (t + 1 - lo)) + 1e-6
            e0 += _gamma_lpdf(v, BULL[2], BULL[3])
            e1 += _gamma_lpdf(v, BEAR[2], BEAR[3])
        if use_cusum:
            z = (r - BULL[0]) / BULL[1]
            S = max(0.0, S - z - 0.5)
            if S > 3.0:
                trig = True
            if trig:
                e1 += 2.0
        n0 = e0 + _lse2(lb0 + lstay, lb1 + lmove)
        n1 = e1 + _lse2(lb0 + lmove, lb1 + lstay)
        c = _lse2(n0, n1)
        lb0 = n0 - c
        lb1 = n1 - c
        out[t] = lb1 - lb0


@njit(parallel=True, cache=False)
def hmm_scores(X, aii, use_vol, use_cusum):
    n, T = X.shape
    out = np.empty((n, T))
    for i in prange(n):
        _hmm_path(X[i], aii, use_vol, use_cusum, out[i])
    return out


@njit(cache=False)
def _t_lpdf(x, nu, mu, s2):
    return (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi * s2)
            - 0.5 * (nu + 1.0) * math.log(1.0 + (x - mu) ** 2 / (nu * s2)))


@njit(cache=False)
def _bocpd_path(x, hazard, L, out_rl, out_mean, keep_last_rl):
    T = x.shape[0]
    mu = np.empty(T + 2)
    ka = np.empty(T + 2)
    al = np.empty(T + 2)
    be = np.empty(T + 2)
    logR = np.full(T + 2, -np.inf)
    newR = np.full(T + 2, -np.inf)
    mu[0], ka[0], al[0], be[0] = BOCPD_PRIOR[0], BOCPD_PRIOR[1], BOCPD_PRIOR[2], BOCPD_PRIOR[3]
    logR[0] = 0.0
    lH = math.log(hazard)
    l1H = math.log(1.0 - hazard)
    for t in range(T):
        xt = x[t]
        # predictive log-likelihood for every current run length l = 0..t
        cp = -np.inf
        for l in range(t + 1):
            if logR[l] == -np.inf:
                newR[l + 1] = -np.inf
                continue
            s2 = be[l] * (ka[l] + 1.0) / (al[l] * ka[l])
            lp = _t_lpdf(xt, 2.0 * al[l], mu[l], s2)
            newR[l + 1] = logR[l] + lp + l1H
            cp = _lse2(cp, logR[l] + lH)
        lp0 = _t_lpdf(xt, 2.0 * al[0], mu[0], be[0] * (ka[0] + 1.0) / (al[0] * ka[0]))
        newR[1] = _lse2(newR[1], cp + lp0)
        newR[0] = -np.inf
        # normalise
        c = -np.inf
        for l in range(1, t + 2):
            c = _lse2(c, newR[l])
        for l in range(1, t + 2):
            newR[l] -= c
        # update sufficient statistics (shift: stats[l+1] = update(stats[l], x_t)); stats[0] stays prior
        for l in range(t, -1, -1):
            k1 = ka[l] + 1.0
            mu_new = (ka[l] * mu[l] + xt) / k1
            be[l + 1] = be[l] + ka[l] * (xt - mu[l]) ** 2 / (2.0 * k1)
            al[l + 1] = al[l] + 0.5
            mu[l + 1] = mu_new
            ka[l + 1] = k1
        # scores
        a = -np.inf
        b = -np.inf
        em = 0.0
        for l in range(1, t + 2):
            if l <= L:
                a = _lse2(a, newR[l])
            else:
                b = _lse2(b, newR[l])
            em += math.exp(newR[l]) * mu[l]
        if b == -np.inf:
            out_rl[t] = BIG
        elif a == -np.inf:
            out_rl[t] = -BIG
        else:
            out_rl[t] = a - b
        out_mean[t] = -em
        for l in range(t + 2):
            logR[l] = newR[l]
    if keep_last_rl.shape[0] > 0:
        for l in range(min(keep_last_rl.shape[0], T + 2)):
            keep_last_rl[l] = logR[l]


@njit(parallel=True, cache=False)
def bocpd_scores(X, hazard, L):
    n, T = X.shape
    rl = np.empty((n, T))
    mn = np.empty((n, T))
    dummy = np.empty(0)
    for i in prange(n):
        _bocpd_path(X[i], hazard, L, rl[i], mn[i], dummy)
    return rl, mn


def bocpd_runlength_posterior(x, hazard):
    """Run-length posterior (log, index = run length) after the last observation (tests)."""
    T = len(x)
    keep = np.full(T + 2, -np.inf)
    rl = np.empty(T)
    mn = np.empty(T)
    _bocpd_path(np.asarray(x, float), hazard, BOCPD_L, rl, mn, keep)
    return keep


@njit(cache=False)
def _jm_fit(Z, lam, maxit, states, V, cent):
    """Fit a 2-state jump model to Z (m, 2). Returns final costs V_end (2,) via V[m-1]."""
    m = Z.shape[0]
    # median split on feature 0
    tmp = np.sort(Z[:, 0])
    med = tmp[m // 2] if m % 2 == 1 else 0.5 * (tmp[m // 2 - 1] + tmp[m // 2])
    for k in range(2):
        for d in range(2):
            cent[k, d] = 0.0
    cnt0 = 0
    cnt1 = 0
    for j in range(m):
        if Z[j, 0] <= med:
            states[j] = 0
            cnt0 += 1
            cent[0, 0] += Z[j, 0]
            cent[0, 1] += Z[j, 1]
        else:
            states[j] = 1
            cnt1 += 1
            cent[1, 0] += Z[j, 0]
            cent[1, 1] += Z[j, 1]
    if cnt0 > 0:
        cent[0, 0] /= cnt0
        cent[0, 1] /= cnt0
    if cnt1 > 0:
        cent[1, 0] /= cnt1
        cent[1, 1] /= cnt1
    back = np.empty((m, 2), dtype=np.int64)
    newst = np.empty(m, dtype=np.int64)
    for it in range(maxit):
        for k in range(2):
            V[0, k] = (Z[0, 0] - cent[k, 0]) ** 2 + (Z[0, 1] - cent[k, 1]) ** 2
        for j in range(1, m):
            for k in range(2):
                c = (Z[j, 0] - cent[k, 0]) ** 2 + (Z[j, 1] - cent[k, 1]) ** 2
                stay = V[j - 1, k]
                move = V[j - 1, 1 - k] + lam
                if stay <= move:
                    V[j, k] = c + stay
                    back[j, k] = k
                else:
                    V[j, k] = c + move
                    back[j, k] = 1 - k
        k = 0 if V[m - 1, 0] <= V[m - 1, 1] else 1
        newst[m - 1] = k
        for j in range(m - 1, 0, -1):
            k = back[j, k]
            newst[j - 1] = k
        changed = False
        for j in range(m):
            if newst[j] != states[j]:
                changed = True
            states[j] = newst[j]
        if not changed and it > 0:
            break
        # centroid update (keep old centroid if a state is empty)
        s00 = 0.0
        s01 = 0.0
        s10 = 0.0
        s11 = 0.0
        c0 = 0
        c1 = 0
        for j in range(m):
            if states[j] == 0:
                s00 += Z[j, 0]
                s01 += Z[j, 1]
                c0 += 1
            else:
                s10 += Z[j, 0]
                s11 += Z[j, 1]
                c1 += 1
        if c0 > 0:
            cent[0, 0] = s00 / c0
            cent[0, 1] = s01 / c0
        if c1 > 0:
            cent[1, 0] = s10 / c1
            cent[1, 1] = s11 / c1
        if it == maxit - 1:
            # recompute costs with the final centroids so V matches cent
            for k2 in range(2):
                V[0, k2] = (Z[0, 0] - cent[k2, 0]) ** 2 + (Z[0, 1] - cent[k2, 1]) ** 2
            for j in range(1, m):
                for k2 in range(2):
                    c = (Z[j, 0] - cent[k2, 0]) ** 2 + (Z[j, 1] - cent[k2, 1]) ** 2
                    V[j, k2] = c + min(V[j - 1, k2], V[j - 1, 1 - k2] + lam)


@njit(cache=False)
def _jm_path(x, lam, W, maxit, out):
    T = x.shape[0]
    vol = np.empty(T)
    for t in range(T):
        lo = max(0, t - 4)
        m = 0.0
        for k in range(lo, t + 1):
            m += x[k]
        m /= (t + 1 - lo)
        ss = 0.0
        for k in range(lo, t + 1):
            ss += (x[k] - m) ** 2
        vol[t] = math.sqrt(ss / (t + 1 - lo))
    Zw = np.empty((W, 2))
    states = np.empty(W, dtype=np.int64)
    V = np.empty((W, 2))
    cent = np.empty((2, 2))
    for t in range(T):
        lo = max(0, t - W + 1)
        m = t + 1 - lo
        if m < 3:
            out[t] = 0.0
            continue
        for d in range(2):
            s = 0.0
            for j in range(m):
                s += x[lo + j] if d == 0 else vol[lo + j]
            mean = s / m
            ss = 0.0
            for j in range(m):
                v = (x[lo + j] if d == 0 else vol[lo + j]) - mean
                ss += v * v
            sd = math.sqrt(ss / m)
            if sd <= 0.0:
                sd = 1.0
            for j in range(m):
                Zw[j, d] = ((x[lo + j] if d == 0 else vol[lo + j]) - mean) / sd
        _jm_fit(Zw[:m], lam, maxit, states[:m], V[:m], cent)
        bear = 0 if cent[0, 0] <= cent[1, 0] else 1
        out[t] = V[m - 1, 1 - bear] - V[m - 1, bear]


@njit(parallel=True, cache=False)
def jm_scores(X, lam, W, maxit):
    n, T = X.shape
    out = np.empty((n, T))
    for i in prange(n):
        _jm_path(X[i], lam, W, maxit, out[i])
    return out


def jm_offline(Z, lam, maxit=JM_MAXIT):
    """Offline fit on a feature matrix (tests). Returns (states, centroids, V_end)."""
    Z = np.ascontiguousarray(Z, float)
    m = Z.shape[0]
    st = np.empty(m, dtype=np.int64)
    V = np.empty((m, 2))
    cent = np.empty((2, 2))
    _jm_fit(Z, lam, maxit, st, V, cent)
    return st, cent, V[m - 1].copy()


def jump_penalty(aii):
    return 2.0 * np.log(aii / (1.0 - aii))


def all_scores(X, aii):
    """Dict detector -> score matrix (n, T) for log returns X at persistence aii."""
    X = np.ascontiguousarray(X, float)
    rl, mn = bocpd_scores(X, 1.0 - aii, BOCPD_L)
    return {
        'hmm_ret': hmm_scores(X, aii, False, False),
        'volhmm': hmm_scores(X, aii, True, False),
        'volhmm_cusum': hmm_scores(X, aii, True, True),
        'bocpd_rl': rl,
        'bocpd_mean': mn,
        'jm': jm_scores(X, jump_penalty(aii), JM_WINDOW, JM_MAXIT),
    }
