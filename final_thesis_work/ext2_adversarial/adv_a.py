"""Adversarial checks on outcome a (detection frontier). Reads ext2lib, never writes into ext2/.

Usage (from final_thesis_work/):
  PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_adv_numba NUMBA_NUM_THREADS=2 \
      python3 ext2_adversarial/adv_a.py > ext2_adversarial/out_a.json
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'ext2'))
sys.path.insert(0, ROOT)
from ext2lib import dgp, detectors as DET, stats as ST  # noqa: E402

THR = json.load(open(os.path.join(ROOT, 'ext2', 'results', 'exploratory', 'thresholds.json')))
B, BURN = 125, 10
seeds = dgp.CONFIRMATORY_SEEDS
Z, V = dgp.step_draws(seeds)
X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
g = np.random.default_rng(12345)
boot = g.integers(0, len(seeds), size=(2000, len(seeds)))


def ci(d):
    m = d[boot].mean(axis=1)
    return [round(float(d.mean()), 4), round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def onset_rate(S, h):
    return ST.prebreak_onsets(S, h)


def path_fa(S, h):
    return float((S[:, BURN:B] > h).any(axis=1).mean())


def match(S, fn, target):
    """Bisection on h (monotone decreasing criterion) over pooled pre-break score quantiles."""
    pre = np.unique(S[:, BURN:B].ravel())
    lo, hi = 0, pre.size - 1          # criterion(pre[lo]) high, criterion(pre[hi]) low
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fn(S, pre[mid]) > target:
            lo = mid
        else:
            hi = mid
    return float(pre[hi])


out = {}
hasher = hashlib.sha256()
for aii in (0.95, 0.97, 0.99):
    S = DET.all_scores(X, aii)
    for k in sorted(S):
        hasher.update(S[k].tobytes())
    res = {}
    # 1. reproduce frozen-threshold delays
    dl = {}
    for d in DET.DETECTORS:
        h = THR[str(aii)][d]['5']
        dd, det = ST.detection_delays(S[d], h, 50)
        dl[d] = dd
        carry = S[d][:, B - 1] > h
        # delay counting only a fresh onset at t >= 125 (alarm off at t - 1)
        a = S[d] > h
        on = a[:, B:B + 50] & ~a[:, B - 1:B + 49]
        fresh = np.where(on.any(axis=1), on.argmax(axis=1), 50).astype(float)
        hor = {}
        for H in (10, 20, 50, 125):
            x_, _ = ST.detection_delays(S[d], h, H)
            hor[H] = round(float(x_.mean()), 4)
        res[d] = dict(threshold=h, fa=round(ST.prebreak_rate(S[d], h), 3), onsets=round(onset_rate(S[d], h), 3),
                      path_fa=round(path_fa(S[d], h), 4), mean=round(float(dd.mean()), 4),
                      median=float(np.median(dd)), never=float(1 - det.mean()),
                      share_alarm_on_at_124=float(carry.mean()), share_delay0=float((dd == 0).mean()),
                      mean_fresh_onset=round(float(fresh.mean()), 4), mean_by_horizon=hor)
    pairs = {'A1': ('volhmm', 'hmm_ret'), 'A2': ('volhmm_cusum', 'volhmm'), 'A3': ('bocpd_rl', 'volhmm_cusum'),
             'A4': ('jm', 'volhmm_cusum'), 'A0': ('volhmm_cusum', 'hmm_ret')}
    res['contrasts_frozen'] = {k: ci(dl[a] - dl[b]) for k, (a, b) in pairs.items()}
    res['contrasts_frozen_median_of_each'] = {k: [float(np.median(dl[a])), float(np.median(dl[b]))]
                                              for k, (a, b) in pairs.items()}
    # A3/A4 decomposition: part of the mean difference from never-detected paths (censored at 50)
    for k in ('A3', 'A4'):
        a, b = pairs[k]
        da, db = dl[a], dl[b]
        cens = (da == 50) | (db == 50)
        res[f'{k}_share_of_mean_diff_from_censored_paths'] = round(float((da - db)[cens].sum() / (da - db).sum()), 4)
    # 2. in-sample exact matching (confirmatory seeds, sensitivity only) under three criteria
    for crit, fn, targets in (('occupancy', ST.prebreak_rate, (5.0,)), ('onsets', onset_rate, (1.0, 2.5, 5.0)),
                              ('path_any_fa', path_fa, (0.25,))):
        for tg in targets:
            dlm, info = {}, {}
            for d in DET.DETECTORS:
                h = match(S[d], fn, tg)
                dd, det = ST.detection_delays(S[d], h, 50)
                dlm[d] = dd
                info[d] = dict(h=round(h, 4), crit=round(fn(S[d], h), 4), fa=round(ST.prebreak_rate(S[d], h), 3),
                               mean=round(float(dd.mean()), 4), median=float(np.median(dd)),
                               never=round(float(1 - det.mean()), 4))
            info['contrasts'] = {k: ci(dlm[a] - dlm[b]) for k, (a, b) in pairs.items()}
            res[f'matched_{crit}_{tg}'] = info
    out[str(aii)] = res
out['score_sha256_all_aii'] = hasher.hexdigest()
print(json.dumps(out, indent=1))
