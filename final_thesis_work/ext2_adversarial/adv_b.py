"""Adversarial checks on outcome b (CPPI gap risk) and the bounded-model growth claim.

Usage (from final_thesis_work/):
  PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_adv_numba NUMBA_NUM_THREADS=2 \
      python3 ext2_adversarial/adv_b.py > ext2_adversarial/out_b.json
"""
import json
import os
import sys

import numpy as np
from scipy import integrate, stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'ext2'))
sys.path.insert(0, ROOT)
from ext2lib import dgp, cppi as CP  # noqa: E402

seeds = dgp.CONFIRMATORY_SEEDS
out = {}
Y = np.concatenate([dgp.fine_log_increments(seeds[i:i + 500], within='diffusive') for i in range(0, 2000, 500)])
res = {}
for n in (1, 50):
    for lmin in (0.0, 0.001, 0.005, 0.01, 0.02):
        R = CP.cppi_paths(Y, n_reb=n, m=3.0, l_min=lmin, peak_every_rebalance=False)
        br = R['breach_step']
        mdd = R['mdd_step']
        res[f'n{n}_lmin{lmin}'] = dict(
            breach=float(br.mean()),
            breach_mdd_gt_0p21=float((mdd > 0.21).mean()), breach_mdd_gt_0p25=float((mdd > 0.25).mean()),
            median_mdd_breachers=float(np.median(mdd[br])) if br.any() else None,
            p90_mdd_breachers=float(np.percentile(mdd[br], 90)) if br.any() else None,
            median_terminal=float(np.median(R['terminal'])))
        if lmin in (0.0, 0.01):
            res[f'n{n}_lmin{lmin}_breach_vec'] = br
out['b_lmin_dose'] = {k: v for k, v in res.items() if not k.endswith('_vec')}
for n in (1, 50):
    v1 = res[f'n{n}_lmin0.01_breach_vec']
    v2 = res[f'n{n}_lmin0.0_breach_vec']
    out[f'overlap_n{n}'] = dict(v1=int(v1.sum()), v2=int(v2.sum()), v2_and_v1=int((v1 & v2).sum()),
                               v2_not_v1=int((v2 & ~v1).sum()),
                               share_v1_breaches_removed_by_lmin0=float((v1 & ~v2).sum() / v1.sum()))
# peak_every with the evaluation floor on the fine grid (does the amended primary matter for the agent's own floor?)
for n in (1, 50):
    R = CP.cppi_paths(Y, n_reb=n, m=3.0, l_min=0.01, peak_every_rebalance=True)
    out[f'V1_peak_every_n{n}'] = dict(breach_step=float(R['breach_step'].mean()), breach_fine=float(R['breach_fine'].mean()))
    R = CP.cppi_paths(Y, n_reb=n, m=3.0, l_min=0.01, peak_every_rebalance=False)
    out[f'V1_peak_step_n{n}'] = dict(breach_step=float(R['breach_step'].mean()), breach_fine=float(R['breach_fine'].mean()))


# bear log growth: independent quadrature on x directly, and tail contributions
def dens(x, mu, s):
    return stats.t(3).pdf((x - mu) / s) / s


def integrand(x, f, mu, s):
    return dens(x, mu, s) * np.logaddexp(np.log1p(-f), np.log(f) + x)


def growth(f, mu, s, lo=-np.inf, hi=np.inf):
    pts = [mu - 50 * s, mu - 5 * s, mu, mu + 5 * s, mu + 50 * s]
    tot = 0.0
    edges = [lo] + [p for p in pts if lo < p < hi] + [hi]
    for a, b in zip(edges[:-1], edges[1:]):
        tot += integrate.quad(integrand, a, b, args=(f, mu, s), limit=500)[0]
    return tot


mu, s = -0.10, 0.30
g = {}
for f in (0.01, 0.05, 0.072, 0.1, 0.5):
    full = growth(f, mu, s)
    # contribution of the region x > c (simple return above exp(c) - 1)
    contrib = {str(c): growth(f, mu, s, lo=c) for c in (1.0, 2.0, 3.0)}
    # growth if the bear were Student-t truncated at |z| <= 5 (renormalised)
    zc = 5.0
    mass = stats.t(3).cdf(zc) - stats.t(3).cdf(-zc)
    trunc = growth(f, mu, s, lo=mu - zc * s, hi=mu + zc * s) / mass
    g[str(f)] = dict(full=full, contrib_x_gt=contrib, truncated_z5=trunc)
kel = optimize.minimize_scalar(lambda f: -growth(f, mu, s), bounds=(1e-6, 1 - 1e-9), method='bounded',
                               options={'xatol': 1e-7})
out['bear_growth'] = dict(per_f=g, kelly=float(kel.x), kelly_growth=float(-kel.fun),
                          P_x_gt_1=float(stats.t(3).sf((1 - mu) / s)), P_x_gt_2=float(stats.t(3).sf((2 - mu) / s)))
print(json.dumps(out, indent=1))
