"""Is A2 ("the CUSUM makes detection slower") about CUSUM or about the thesis's latch?
Vectorised re-implementation of the Vol-HMM forward filter with (i) the thesis latching CUSUM,
(ii) a non-latching CUSUM (+2 only while S > 3), (iii) no CUSUM; checked against ext2lib on (i) and (iii).
Thresholds tuned on exploratory seeds (target 5 per 1,000, occupancy), evaluated on confirmatory seeds.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'ext2'))
sys.path.insert(0, ROOT)
from ext2lib import dgp, detectors as DET, stats as ST  # noqa: E402


def volhmm(X, aii, mode):
    n, T = X.shape
    la = np.log(np.array([[aii, 1 - aii], [1 - aii, aii]]))
    lb = np.log(np.full((n, 2), 0.5))
    S = np.zeros(n)
    trig = np.zeros(n, bool)
    out = np.empty((n, T))
    for t in range(T):
        r = X[:, t]
        e0 = sps.norm.logpdf(r, 0.08, 0.15)
        e1 = sps.norm.logpdf(r, -0.10, 0.30)
        if t + 1 < 2:
            v = np.full(n, 0.05)
        else:
            v = X[:, max(0, t - 4):t + 1].std(axis=1) + 1e-6
        e0 = e0 + sps.gamma.logpdf(v, 2.0, scale=0.075)
        e1 = e1 + sps.gamma.logpdf(v, 4.0, scale=0.15)
        if mode != 'none':
            S = np.maximum(0.0, S - (r - 0.08) / 0.15 - 0.5)
            if mode == 'latch':
                trig |= S > 3.0
                on = trig
            else:
                on = S > 3.0
            e1 = e1 + 2.0 * on
        n0 = e0 + logsumexp(np.stack([lb[:, 0] + la[0, 0], lb[:, 1] + la[1, 0]]), axis=0)
        n1 = e1 + logsumexp(np.stack([lb[:, 0] + la[0, 1], lb[:, 1] + la[1, 1]]), axis=0)
        c = np.logaddexp(n0, n1)
        lb = np.stack([n0 - c, n1 - c], axis=1)
        out[:, t] = lb[:, 1] - lb[:, 0]
    return out


def X_for(seeds):
    Z, V = dgp.step_draws(seeds)
    return dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)


Xe, Xc = X_for(dgp.EXPLORATORY_SEEDS), X_for(dgp.CONFIRMATORY_SEEDS)
res = {}
g = np.random.default_rng(12345)
boot = g.integers(0, 2000, size=(2000, 2000))
for aii in (0.95, 0.99):
    lib = DET.all_scores(Xc[:200], aii)
    chk = {'latch': float(np.max(np.abs(volhmm(Xc[:200], aii, 'latch') - lib['volhmm_cusum']))),
           'none': float(np.max(np.abs(volhmm(Xc[:200], aii, 'none') - lib['volhmm'])))}
    dl = {}
    rr = {'max_abs_diff_vs_ext2lib': chk}
    for mode in ('none', 'latch', 'reset'):
        h = ST.tune_threshold(volhmm(Xe, aii, mode), 5)
        Sc = volhmm(Xc, aii, mode)
        d, det = ST.detection_delays(Sc, h, 50)
        dl[mode] = d
        rr[mode] = dict(h=h, fa=round(ST.prebreak_rate(Sc, h), 3), mean=float(d.mean()), median=float(np.median(d)))
    for a, b in (('latch', 'none'), ('reset', 'none')):
        diff = dl[a] - dl[b]
        m = diff[boot].mean(axis=1)
        rr[f'{a}_minus_{b}'] = [round(float(diff.mean()), 4), round(float(np.quantile(m, .025)), 4),
                                round(float(np.quantile(m, .975)), 4)]
    res[str(aii)] = rr
print(json.dumps(res, indent=1))
