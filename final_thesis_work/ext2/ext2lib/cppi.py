"""CPPI-Kelly simulator (CPPI_SPEC.md), numba, one path per thread.

cppi_paths(Y, n_reb, m, l_min, l_max, d_max, fdet, use_cppi)
    Y      (n, T, K) fine LOG increments; K must be divisible by n_reb. With K = 1 and n_reb = 1 this is the
           thesis agent on step returns r_t = expm1(Y[:, t, 0]).
    fdet   (n, T) base fraction to scale, or an empty (0, 0) array to use the thesis Beta-posterior
           binary Kelly fraction clip(2 a / (a + b) - 1, 0, 1) (CPPI_SPEC.md section 2).
    use_cppi  False -> lambda = l_max always (no floor), used for the "no floor" comparison.
    peak_every_rebalance  True (CPPI_SPEC.md section 11): P := max(P, W) after every rebalance interval.
           False (diagnostic, amendment 2): P is updated only at step ends, so the floor is tracked on the
           same grid whatever n is and only the exposure is rebalanced n times.
Returns a dict of per-path arrays:
    terminal  terminal wealth W_T
    min_w     minimum wealth on the rebalancing grid
    ruined    1 + f r <= 0 happened (absorbing, wealth 0)
    breach_step  W_t < 0.8 * max_{s<=t} W_s on the step-end grid t = 0..T (CPPI_SPEC.md section 8)
    breach_fine  the same on every fine point (K per step), against the fine-grid running peak
    mdd_step  maximum drawdown on the step-end grid
    first_breach_lambda_at_min  for the first step that ends in a step-grid breach: 1 if lambda sat at l_min
           (or the cushion was <= 0) at the start of the first rebalance interval of that step whose end
           wealth was below 0.8 x the step-grid peak, 0 if not, -1 if no breach (diagnostic)
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=False)
def _cppi_path(Y, n_reb, m, l_min, l_max, d_max, fdet, has_fdet, use_cppi, peak_every, res):
    T = Y.shape[0]
    K = Y.shape[1]
    step = K // n_reb
    a = 1.0
    b = 1.0
    logW = 0.0
    W = 1.0
    P = 1.0
    peak_step = 1.0
    peak_fine = 1.0
    min_w = 1.0
    ruined = False
    br_step = False
    br_fine = False
    mdd = 0.0
    lam_min_hit = False      # lambda clipped at l_min in the current step
    first_at_min = -1.0      # 1 if lambda was at l_min in some interval of the first breaching step, 0 if not
    for t in range(T):
        cross_flag = -1.0    # lambda-at-min flag of the first interval this step ending below 0.8 * peak_step
        if has_fdet:
            fb = fdet[t]
        else:
            p = a / (a + b)
            fb = p - (1.0 - p) / 1.0
            if fb < 0.0:
                fb = 0.0
            if fb > 1.0:
                fb = 1.0
        ysum_step = 0.0
        for j in range(n_reb):
            if use_cppi:
                F = P * (1.0 - d_max)
                C = W - F
                at_min = False
                if C <= 0.0:
                    lam = l_min
                    at_min = True
                else:
                    lam = m * C / W
                    if lam <= l_min:
                        lam = l_min
                        at_min = True
                    if lam > l_max:
                        lam = l_max
            else:
                lam = l_max
                at_min = False
            f = lam * fb
            if f > 1.0:
                f = 1.0
            cum = 0.0
            for k in range(j * step, (j + 1) * step):
                cum += Y[t, k]
                if not ruined:
                    wk = W * (1.0 + f * math.expm1(cum))
                    if wk > peak_fine:
                        peak_fine = wk
                    if wk < (1.0 - d_max) * peak_fine:
                        br_fine = True
            ysum_step += cum
            if not ruined:
                mult = 1.0 + f * math.expm1(cum)
                if mult <= 0.0:
                    ruined = True
                    W = 0.0
                    min_w = 0.0
                    br_fine = True
                else:
                    logW += math.log(mult)
                    W = math.exp(logW)
                    if W < min_w:
                        min_w = W
                    if cross_flag < 0.0 and W < (1.0 - d_max) * peak_step:
                        cross_flag = 1.0 if at_min else 0.0
                    if peak_every and W > P:
                        P = W
        # step end
        if (not peak_every) and W > P:
            P = W
        # Beta update on the sign of the step's simple return expm1(ysum_step) (same sign as ysum_step)
        if ysum_step > 0.0:
            a += 1.0
        else:
            b += 1.0
        if W > peak_step:
            peak_step = W
        dd = 1.0 - W / peak_step
        if dd > mdd:
            mdd = dd
        if W < (1.0 - d_max) * peak_step:
            if not br_step:
                first_at_min = cross_flag
            br_step = True
    res[0] = W
    res[1] = min_w
    res[2] = 1.0 if ruined else 0.0
    res[3] = 1.0 if br_step else 0.0
    res[4] = 1.0 if br_fine else 0.0
    res[5] = mdd
    res[6] = first_at_min


@njit(parallel=True, cache=False)
def _cppi_batch(Y, n_reb, m, l_min, l_max, d_max, fdet, has_fdet, use_cppi, peak_every):
    n = Y.shape[0]
    out = np.empty((n, 7))
    for i in prange(n):
        fd = fdet[i] if has_fdet else np.empty(0)
        _cppi_path(Y[i], n_reb, m, l_min, l_max, d_max, fd, has_fdet, use_cppi, peak_every, out[i])
    return out


def cppi_paths(Y, n_reb=1, m=3.0, l_min=0.01, l_max=0.5, d_max=0.20, fdet=None, use_cppi=True,
               peak_every_rebalance=True):
    Y = np.ascontiguousarray(Y, float)
    if Y.ndim == 2:
        Y = Y[:, :, None]
    assert Y.shape[2] % n_reb == 0
    has = fdet is not None
    fd = np.ascontiguousarray(fdet, float) if has else np.empty((Y.shape[0], 0))
    o = _cppi_batch(Y, int(n_reb), float(m), float(l_min), float(l_max), float(d_max), fd, has, bool(use_cppi),
                    bool(peak_every_rebalance))
    return {'terminal': o[:, 0], 'min_w': o[:, 1], 'ruined': o[:, 2] > 0.5, 'breach_step': o[:, 3] > 0.5,
            'breach_fine': o[:, 4] > 0.5, 'mdd_step': o[:, 5], 'first_breach_lambda_at_min': o[:, 6]}


def simple_wealth(R, F):
    """Generic wealth for bets F (n, T) on simple returns R (n, T) with the WealthTracker rules."""
    n, T = R.shape
    W = np.ones(n)
    logW = np.zeros(n)
    ruined = np.zeros(n, bool)
    min_w = np.ones(n)
    for t in range(T):
        mult = 1.0 + F[:, t] * R[:, t]
        newly = (~ruined) & (mult <= 0)
        ruined |= newly
        ok = ~ruined
        logW[ok] += np.log(mult[ok])
        W = np.where(ruined, 0.0, np.exp(logW))
        min_w = np.minimum(min_w, W)
    return W, min_w, ruined


def cppi_simple_returns(R, m=3.0, l_min=0.01, l_max=0.5, d_max=0.20, fdet=None, use_cppi=True):
    """Thesis CPPI agent (one rebalance per step) on SIMPLE returns R (n, T), vectorised over paths.
    Works for the unbounded model (r <= -1 possible). Same outputs as cppi_paths (no fine grid).
    fdet / use_cppi as in cppi_paths."""
    n, T = R.shape
    a = np.ones(n)
    b = np.ones(n)
    logW = np.zeros(n)
    W = np.ones(n)
    P = np.ones(n)
    peak = np.ones(n)
    ruined = np.zeros(n, bool)
    breach = np.zeros(n, bool)
    min_w = np.ones(n)
    mdd = np.zeros(n)
    for t in range(T):
        p = a / (a + b)
        fb = np.clip(p - (1.0 - p), 0.0, 1.0) if fdet is None else fdet[:, t]
        C = W - P * (1.0 - d_max)
        lam = np.where(C <= 0.0, l_min, np.clip(m * C / np.where(W > 0, W, 1.0), l_min, l_max))
        if not use_cppi:
            lam = np.full(n, l_max)
        f = np.minimum(lam * fb, 1.0)
        mult = 1.0 + f * R[:, t]
        ruined |= (mult <= 0.0)
        ok = ~ruined
        logW[ok] += np.log(mult[ok])
        W = np.where(ruined, 0.0, np.exp(logW))
        win = R[:, t] > 0
        a += win
        b += ~win
        P = np.maximum(P, W)
        peak = np.maximum(peak, W)
        min_w = np.minimum(min_w, W)
        mdd = np.maximum(mdd, 1.0 - W / peak)
        breach |= W < (1.0 - d_max) * peak
    return {'terminal': W, 'min_w': min_w, 'ruined': ruined, 'breach_step': breach, 'mdd_step': mdd}
