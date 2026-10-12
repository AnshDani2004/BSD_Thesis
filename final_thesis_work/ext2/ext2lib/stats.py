"""Statistics for Extension 2 (PREREG.md section 6)."""
import numpy as np

BURN_IN = 10          # steps 0..9 excluded from false-alarm counting and detection
BREAK_T = 125
STAT_SEED = 20261012  # seed of every resampling Generator


# ---------------------------------------------------------------- detection evaluation
def prebreak_rate(scores, h, break_t=BREAK_T, burn=BURN_IN):
    """False alarms per 1,000 pre-break steps: share of steps t in [burn, break_t) with score > h."""
    s = scores[:, burn:break_t]
    return 1000.0 * float((s > h).mean())


def prebreak_onsets(scores, h, break_t=BREAK_T, burn=BURN_IN):
    """Alarm onsets (upcrossings of h) per 1,000 pre-break steps (secondary)."""
    a = scores[:, burn - 1:break_t] > h
    on = a[:, 1:] & ~a[:, :-1]
    return 1000.0 * float(on.mean())


def tune_threshold(scores, target_per_1000, break_t=BREAK_T, burn=BURN_IN):
    """Smallest h, taken from the pooled pre-break scores, with rate(score > h) <= target.

    Pooled pre-break scores sorted descending; allowed count k = floor(target / 1000 * N). h is the
    (k+1)-th largest score, so at most k pooled scores exceed it (ties can only lower the count)."""
    s = np.sort(scores[:, burn:break_t].ravel())[::-1]
    k = int(np.floor(target_per_1000 / 1000.0 * s.size))
    return float(s[min(k, s.size - 1)])


def detection_delays(scores, h, horizon, break_t=BREAK_T):
    """Delay = first t >= break_t with score > h, minus break_t; censored at `horizon` if none in
    [break_t, break_t + horizon). Returns (delay_censored (float), detected (bool))."""
    w = scores[:, break_t:break_t + horizon] > h
    det = w.any(axis=1)
    first = np.where(det, w.argmax(axis=1), horizon).astype(float)
    return first, det


def delay_summary(d, det, horizon):
    return {'n': int(d.size), 'mean_censored': float(d.mean()), 'median_censored': float(np.median(d)),
            'p90_censored': float(np.percentile(d, 90)), 'share_never_detected': float(1 - det.mean()),
            'horizon': int(horizon), 'mean_if_detected': float(d[det].mean()) if det.any() else None}


# ---------------------------------------------------------------- intervals and tests
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - hw)), float(min(1.0, c + hw)))


def bootstrap_ci(x, stat=np.median, B=2000, seed=STAT_SEED, alpha=0.05):
    g = np.random.default_rng(seed)
    x = np.asarray(x)
    n = x.size
    idx = g.integers(0, n, size=(B, n))
    vals = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def paired_signflip(d, n_perm=10000, seed=STAT_SEED):
    """Two-sided sign-flip randomisation test of mean(d) = 0 for paired differences d."""
    g = np.random.default_rng(seed)
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = g.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (n_perm + 1))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = p.size
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(pvals):
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        prev = min(prev, p[i] * m / (r + 1))
        adj[i] = prev
    return adj


# ---------------------------------------------------------------- grouped logistic regression
def logit_fit(X, k, n, ridge=1e-8, iters=100):
    """Binomial logistic regression on grouped data by Newton-Raphson (IRLS). X (g, p) includes the
    intercept column; k successes out of n per group. A tiny ridge (1e-8) keeps the Hessian invertible
    when some cells are all zero."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = n * mu * (1 - mu)
        grad = X.T @ (k - n * mu) - ridge * beta
        H = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def gap_risk_regression(breach, n_list, m_list, B=2000, seed=STAT_SEED):
    """breach: bool array (n_paths, len(n_list), len(m_list)) on common paths.
    Model: logit P(breach) = b0 + b1 log(n) + b2 m. Point estimate on all paths; CIs and one-sided
    p-values (H1: b1 < 0; H1: b2 > 0) from a path bootstrap (paths resampled with replacement, the
    whole n x m block of a path kept together)."""
    N = breach.shape[0]
    nn, mm = np.meshgrid(np.log(np.asarray(n_list, float)), np.asarray(m_list, float), indexing='ij')
    X = np.column_stack([np.ones(nn.size), nn.ravel(), mm.ravel()])
    cnt = np.full(nn.size, float(N))

    def fit(b):
        return logit_fit(X, b.sum(axis=0).ravel().astype(float), cnt)

    est = fit(breach)
    g = np.random.default_rng(seed)
    boots = np.empty((B, 3))
    for i in range(B):
        boots[i] = fit(breach[g.integers(0, N, N)])
    ci = np.quantile(boots, [0.025, 0.975], axis=0)
    p_b1 = float((1 + (boots[:, 1] >= 0).sum()) / (B + 1))   # H1: b1 < 0
    p_b2 = float((1 + (boots[:, 2] <= 0).sum()) / (B + 1))   # H1: b2 > 0
    return {'b0': float(est[0]), 'b1_logn': float(est[1]), 'b2_m': float(est[2]),
            'b1_ci95': [float(ci[0, 1]), float(ci[1, 1])], 'b2_ci95': [float(ci[0, 2]), float(ci[1, 2])],
            'p_one_sided_b1_lt_0': p_b1, 'p_one_sided_b2_gt_0': p_b2, 'B': B}
