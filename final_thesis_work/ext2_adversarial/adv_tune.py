"""Would a better-tuned BOCPD or jump model change the ranking? Tune variants on exploratory seeds
0-999 (thresholds and variant choice), evaluate the chosen variants once on confirmatory seeds.
Primary cell only: A_ii-equivalent 0.95 where needed, target 5 per 1,000 (occupancy), thesis shock, D = 125, H = 50.

Usage (from final_thesis_work/):
  PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_adv_numba NUMBA_NUM_THREADS=2 \
      python3 ext2_adversarial/adv_tune.py > ext2_adversarial/out_tune.json
"""
import json
import math
import os
import sys

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'ext2'))
sys.path.insert(0, ROOT)
from ext2lib import dgp, detectors as DET, stats as ST  # noqa: E402


@njit(cache=False)
def _lse2(a, b):
    m = max(a, b)
    if m == -np.inf:
        return -np.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=False)
def _tl(x, nu, mu, s2):
    return (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi * s2)
            - 0.5 * (nu + 1.0) * math.log(1.0 + (x - mu) ** 2 / (nu * s2)))


@njit(cache=False)
def _bocpd(x, H, L, mu0, k0, a0, b0, rl_out, mean_out, neg_out):
    """Same recursion as ext2lib.detectors._bocpd_path with a configurable prior; adds a third score:
    log P(current segment mean < 0) approximated by sum_l P(l) * P(mu_l < 0) (Student-t marginal)."""
    T = x.shape[0]
    mu = np.empty(T + 2); ka = np.empty(T + 2); al = np.empty(T + 2); be = np.empty(T + 2)
    logR = np.full(T + 2, -np.inf); newR = np.full(T + 2, -np.inf)
    mu[0], ka[0], al[0], be[0] = mu0, k0, a0, b0
    logR[0] = 0.0
    lH = math.log(H); l1H = math.log(1.0 - H)
    for t in range(T):
        xt = x[t]
        cp = -np.inf
        for l in range(t + 1):
            if logR[l] == -np.inf:
                newR[l + 1] = -np.inf
                continue
            s2 = be[l] * (ka[l] + 1.0) / (al[l] * ka[l])
            newR[l + 1] = logR[l] + _tl(xt, 2.0 * al[l], mu[l], s2) + l1H
            cp = _lse2(cp, logR[l] + lH)
        lp0 = _tl(xt, 2.0 * al[0], mu[0], be[0] * (ka[0] + 1.0) / (al[0] * ka[0]))
        newR[1] = _lse2(newR[1], cp + lp0)
        newR[0] = -np.inf
        c = -np.inf
        for l in range(1, t + 2):
            c = _lse2(c, newR[l])
        for l in range(1, t + 2):
            newR[l] -= c
        for l in range(t, -1, -1):
            k1 = ka[l] + 1.0
            mn = (ka[l] * mu[l] + xt) / k1
            be[l + 1] = be[l] + ka[l] * (xt - mu[l]) ** 2 / (2.0 * k1)
            al[l + 1] = al[l] + 0.5
            mu[l + 1] = mn
            ka[l + 1] = k1
        a = -np.inf; b = -np.inf; em = 0.0; pneg = 0.0
        for l in range(1, t + 2):
            if l <= L:
                a = _lse2(a, newR[l])
            else:
                b = _lse2(b, newR[l])
            w = math.exp(newR[l])
            em += w * mu[l]
            # marginal of the segment mean: Student-t, 2 alpha dof, loc mu, scale^2 beta / (alpha kappa)
            sc = math.sqrt(be[l] / (al[l] * ka[l]))
            zz = (0.0 - mu[l]) / sc
            nu = 2.0 * al[l]
            # normal approximation to the t cdf (adequate for a ranking score)
            pneg += w * 0.5 * (1.0 + math.erf(zz * (1.0 - 1.0 / (4.0 * nu)) / math.sqrt(2.0 * (1.0 + zz * zz / (2.0 * nu)))))
        rl_out[t] = 1e9 if b == -np.inf else (a - b)
        mean_out[t] = -em
        pneg = min(max(pneg, 1e-300), 1 - 1e-16)
        neg_out[t] = math.log(pneg) - math.log(1 - pneg)
        for l in range(t + 2):
            logR[l] = newR[l]


@njit(parallel=True, cache=False)
def bocpd_batch(X, H, L, mu0, k0, a0, b0):
    n, T = X.shape
    rl = np.empty((n, T)); mn = np.empty((n, T)); ng = np.empty((n, T))
    for i in prange(n):
        _bocpd(X[i], H, L, mu0, k0, a0, b0, rl[i], mn[i], ng[i])
    return rl, mn, ng


def X_for(seeds):
    Z, V = dgp.step_draws(seeds)
    return dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)


def evaluate(S, h):
    d, det = ST.detection_delays(S, h, 50)
    return d, dict(h=float(h), fa=round(ST.prebreak_rate(S, h), 3), mean=round(float(d.mean()), 4),
                   median=float(np.median(d)), never=round(float(1 - det.mean()), 4))


Xe = X_for(dgp.EXPLORATORY_SEEDS)
Xc = X_for(dgp.CONFIRMATORY_SEEDS)
PRIORS = {'default': (0.0, 1.0, 2.0, 0.04), 'bull_informed': (0.08, 10.0, 5.0, 5 * 0.0225),
          'vague': (0.0, 0.1, 1.0, 0.02)}
variants = {}
for H in (1 / 250, 1 / 125, 0.01, 0.02, 0.05):
    for pname, pr in PRIORS.items():
        for L in (3, 5, 10, 20):
            rl, mn, ng = bocpd_batch(Xe, H, L, *pr)
            for sname, S in (('rl', rl), ('mean', mn), ('pneg', ng)):
                if sname != 'rl' and L != 10:
                    continue           # mean / pneg scores do not depend on L
                key = f'bocpd_{sname}|H={H:.4f}|prior={pname}|L={L}'
                h = ST.tune_threshold(S, 5)
                _, info = evaluate(S, h)
                variants[key] = dict(info, cfg=('bocpd', H, pname, L, sname))
for lam in (0.5, 1.0, 2.0, 3.0, 5.89, 10.0, 20.0):
    for W in (30, 60, 120):
        S = DET.jm_scores(np.ascontiguousarray(Xe), lam, W, 10)
        key = f'jm|lam={lam}|W={W}'
        h = ST.tune_threshold(S, 5)
        _, info = evaluate(S, h)
        variants[key] = dict(info, cfg=('jm', lam, W))
best = {}
for fam in ('bocpd_rl', 'bocpd_mean', 'bocpd_pneg', 'jm'):
    ks = [k for k in variants if k.split('|')[0] == fam]
    best[fam] = min(ks, key=lambda k: variants[k]['mean'])
# confirmatory evaluation of the chosen variants with exploratory-tuned thresholds, vs frozen HMM thresholds
THR = json.load(open(os.path.join(ROOT, 'ext2', 'results', 'exploratory', 'thresholds.json')))['0.95']
Sc = DET.all_scores(Xc, 0.95)
ref = {}
for d in ('hmm_ret', 'volhmm', 'volhmm_cusum'):
    ref[d], _ = evaluate(Sc[d], THR[d]['5'])
conf = {}
g = np.random.default_rng(12345)
boot = g.integers(0, 2000, size=(2000, 2000))
for fam, key in best.items():
    cfg = variants[key]['cfg']
    if cfg[0] == 'bocpd':
        _, H, pname, L, sname = cfg
        rl, mn, ng = bocpd_batch(Xc, H, L, *PRIORS[pname])
        S = {'rl': rl, 'mean': mn, 'pneg': ng}[sname]
    else:
        S = DET.jm_scores(np.ascontiguousarray(Xc), cfg[1], cfg[2], 10)
    d, info = evaluate(S, variants[key]['h'])
    for r in ref:
        diff = d - ref[r]
        m = diff[boot].mean(axis=1)
        info[f'minus_{r}'] = [round(float(diff.mean()), 4), round(float(np.quantile(m, .025)), 4),
                              round(float(np.quantile(m, .975)), 4)]
    conf[fam] = dict(variant=key, exploratory=variants[key], confirmatory=info)
top = sorted(variants, key=lambda k: variants[k]['mean'])[:12]
print(json.dumps({'n_variants': len(variants), 'top12_exploratory': {k: variants[k] for k in top},
                  'best_by_family': conf,
                  'frozen_default_exploratory': {k: variants[k] for k in
                                                 ('bocpd_rl|H=0.0500|prior=default|L=10', 'jm|lam=5.89|W=60')}},
                 indent=1, default=str))
