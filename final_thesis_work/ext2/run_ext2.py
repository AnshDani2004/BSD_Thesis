"""Extension 2 runner (PREREG.md). Synthetic data only.

Usage (from final_thesis_work/):
    PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_numba python3 ext2/run_ext2.py --phase exploratory
    PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_numba python3 ext2/run_ext2.py --phase confirmatory

The confirmatory phase runs once: it needs a `FREEZE ext2` commit in the history, a clean ext2/ tree
(outside results/), the frozen exploratory thresholds, and no existing results/confirmatory/primary.json.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from ext2lib import dgp, detectors as DET, cppi as CP, stats as ST  # noqa: E402

AII = (0.95, 0.97, 0.99)
TARGETS = (1, 5, 10)
LEVELS = TARGETS + ('native',)
N_LIST = (1, 2, 5, 10, 50)
M_LIST = (1.0, 2.0, 3.0, 5.0)
VARIANTS = {'V1_code': 0.01, 'V2_pure': 0.0}
PEAK_MODES = ('peak_step', 'peak_every')   # amendment 2: peak_step is primary
PRIMARY_CELL = dict(aii=0.95, level=5, shock='thesis', D=125)
CONTRASTS = {'A1': ('volhmm', 'hmm_ret'), 'A2': ('volhmm_cusum', 'volhmm'), 'A3': ('bocpd_rl', 'volhmm_cusum'),
             'A4': ('jm', 'volhmm_cusum'), 'A5': ('bocpd_mean', 'volhmm_cusum')}


def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True, text=True).stdout.strip()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def horizon(D):
    return min(D, 50)


# ============================================================================== part a
def part_a(seeds, phase, outdir, thresholds_in=None):
    Z, V = dgp.step_draws(seeds)
    rows_f, rows_d, delays, info = [], [], {}, {}
    thresholds = {} if thresholds_in is None else thresholds_in
    signs = np.random.default_rng(ST.STAT_SEED).choice(np.array([-1.0, 1.0]), size=(10000, len(seeds)))
    boot_idx = np.random.default_rng(ST.STAT_SEED).integers(0, len(seeds), size=(2000, len(seeds)))

    def med_ci(x):
        v = np.median(x[boot_idx], axis=1)
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    def downstream(X, fdet, label):
        R = CP.cppi_paths(X, fdet=fdet)
        R0 = CP.cppi_paths(X, fdet=fdet, use_cppi=False)
        RU = CP.cppi_simple_returns(X, fdet=fdet)      # amendment 1: unbounded model r = x, same alarms
        n = X.shape[0]
        lo, hi = med_ci(R['terminal'])
        br = int(R['breach_step'].sum())
        pr = int((R['min_w'] <= 0.10).sum())
        return dict(label, n=n, ruin_rate=float(R['ruined'].mean()),
                    practical_ruin_rate=pr / n, practical_ruin_ci_lo=ST.wilson(pr, n)[0],
                    practical_ruin_ci_hi=ST.wilson(pr, n)[1],
                    median_terminal_wealth=float(np.median(R['terminal'])), mtw_ci_lo=lo, mtw_ci_hi=hi,
                    breach_rate=br / n, breach_ci_lo=ST.wilson(br, n)[0], breach_ci_hi=ST.wilson(br, n)[1],
                    nofloor_ruin_rate=float(R0['ruined'].mean()),
                    nofloor_practical_ruin_rate=float((R0['min_w'] <= 0.10).mean()),
                    nofloor_median_terminal_wealth=float(np.median(R0['terminal'])),
                    unbounded_ruin_rate=float(RU['ruined'].mean()),
                    unbounded_practical_ruin_rate=float((RU['min_w'] <= 0.10).mean()),
                    unbounded_median_terminal_wealth=float(np.median(RU['terminal'])),
                    unbounded_breach_rate=float(RU['breach_step'].mean()))

    for aii in AII:
        ka = str(aii)
        pre_ref = None
        # fake-break placebo: no shock at all
        Xn = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 0)
        Sn = DET.all_scores(Xn, aii)
        for si, (shock, (mu, s)) in enumerate(dgp.SHOCKS.items()):
            for D in dgp.DURATIONS:
                X = dgp.shock_log_returns(Z, V, (mu, s), D)
                S = DET.all_scores(X, aii)
                pre = {k: sha(v[:, :dgp.BREAK_T]) for k, v in S.items()}
                if pre_ref is None:
                    pre_ref = pre
                    if thresholds_in is None:
                        thresholds[ka] = {d: {str(tg): ST.tune_threshold(S[d], tg) for tg in TARGETS}
                                          for d in DET.DETECTORS}
                    info[ka] = {'prebreak_sha256': pre}
                assert pre == pre_ref, 'pre-break scores differ across shock cells'
                H = horizon(D)
                for d in DET.DETECTORS:
                    for lev in LEVELS:
                        h = 0.0 if lev == 'native' else thresholds[ka][d][str(lev)]
                        dl, det = ST.detection_delays(S[d], h, H)
                        delays[(aii, lev, shock, D, d)] = dl
                        row = dict(aii=aii, level=lev, shock=shock, D=D, detector=d, threshold=h,
                                   fa_per_1000=ST.prebreak_rate(S[d], h),
                                   onsets_per_1000=ST.prebreak_onsets(S[d], h))
                        row.update(ST.delay_summary(dl, det, H))
                        if si == 0 and D == dgp.DURATIONS[0]:
                            w = Sn[d][:, dgp.BREAK_T:dgp.BREAK_T + 50] > h
                            row['placebo_share_alarm_in_125_175_no_shock'] = float(w.any(axis=1).mean())
                        else:
                            row['placebo_share_alarm_in_125_175_no_shock'] = ''
                        rows_f.append(row)
                        alarm = S[d] > h
                        alarm[:, :ST.BURN_IN] = False
                        fdet = np.ones_like(X)
                        fdet[:, 1:] = np.where(alarm[:, :-1], 0.0, 1.0)
                        rows_d.append(downstream(X, fdet, dict(aii=aii, level=lev, shock=shock, D=D, detector=d)))
                if aii == AII[0]:
                    for bname in ('always_in', 'oracle'):
                        fdet = np.ones_like(X)
                        if bname == 'oracle':
                            fdet[:, dgp.BREAK_T:dgp.BREAK_T + D] = 0.0
                        rows_d.append(downstream(X, fdet, dict(aii='', level='', shock=shock, D=D, detector=bname)))
                print(f'  a: A_ii={aii} shock={shock} D={D} done', flush=True)

    # contrasts
    rows_c = []
    for (aii, lev, shock, D, d), _ in list(delays.items()):
        if d != 'hmm_ret':
            continue
        for cname, (x1, x2) in CONTRASTS.items():
            diff = delays[(aii, lev, shock, D, x1)] - delays[(aii, lev, shock, D, x2)]
            obs = abs(diff.mean())
            null = np.abs(signs @ diff) / diff.size
            p = float((1 + (null >= obs - 1e-12).sum()) / (signs.shape[0] + 1))
            bm = diff[boot_idx].mean(axis=1)
            primary = (aii == PRIMARY_CELL['aii'] and lev == PRIMARY_CELL['level'] and shock == PRIMARY_CELL['shock']
                       and D == PRIMARY_CELL['D'] and cname in ('A1', 'A2', 'A3', 'A4'))
            rows_c.append(dict(contrast=cname, a=x1, b=x2, aii=aii, level=lev, shock=shock, D=D,
                               mean_diff=float(diff.mean()), ci_lo=float(np.quantile(bm, 0.025)),
                               ci_hi=float(np.quantile(bm, 0.975)), p_signflip=p, primary=primary,
                               share_a_faster=float((diff < 0).mean()), share_tied=float((diff == 0).mean())))
    sec = [i for i, r in enumerate(rows_c) if not r['primary'] and r['level'] != 'native']
    q = ST.bh([rows_c[i]['p_signflip'] for i in sec])
    for i, qi in zip(sec, q):
        rows_c[i]['q_bh_secondary'] = float(qi)
    for r in rows_c:
        r.setdefault('q_bh_secondary', '')
    write_csv(os.path.join(outdir, 'a_frontier.csv'), rows_f)
    write_csv(os.path.join(outdir, 'a_downstream.csv'), rows_d)
    write_csv(os.path.join(outdir, 'a_contrasts.csv'), rows_c)
    if thresholds_in is None:
        with open(os.path.join(outdir, 'thresholds.json'), 'w') as fh:
            json.dump(thresholds, fh, indent=1)
    return [r for r in rows_c if r['primary']], info


# ============================================================================== part b
def part_b(seeds, outdir, chunk=500):
    out = {}
    rows = []
    for within in ('diffusive', 'jump', 'placebo'):
        acc = {(v, n, m, pk): [] for v in VARIANTS for n in N_LIST for m in M_LIST for pk in PEAK_MODES}
        acc_unb = {m: [] for m in M_LIST}
        for c0 in range(0, len(seeds), chunk):
            sd = seeds[c0:c0 + chunk]
            if within == 'placebo':
                Y = dgp.fine_log_increments(sd, break_t=dgp.T_STEPS, within='diffusive')
            else:
                Y = dgp.fine_log_increments(sd, within=within)
            for v, lmin in VARIANTS.items():
                for n in N_LIST:
                    for m in M_LIST:
                        for pk in PEAK_MODES:
                            acc[(v, n, m, pk)].append(CP.cppi_paths(Y, n_reb=n, m=m, l_min=lmin,
                                                                    peak_every_rebalance=(pk == 'peak_every')))
            if within == 'diffusive':
                Rs = Y.sum(axis=2)          # step log returns used directly as simple returns (unbounded)
                for m in M_LIST:
                    acc_unb[m].append(CP.cppi_simple_returns(Rs, m=m))
        cat = {k: {f: np.concatenate([d[f] for d in L]) for f in L[0]} for k, L in acc.items()}
        for (v, n, m, pk), d in cat.items():
            N = d['breach_step'].size
            k = int(d['breach_step'].sum())
            kf = int(d['breach_fine'].sum())
            fm = d['first_breach_lambda_at_min']
            rows.append(dict(dgp=within, variant=v, n_rebalance=n, m=m, peak=pk, model='bounded', n_paths=N,
                             share_breaches_lambda_at_min=float((fm == 1).sum() / k) if k else '',
                             breach_rate=k / N, breach_ci_lo=ST.wilson(k, N)[0], breach_ci_hi=ST.wilson(k, N)[1],
                             breach_fine_rate=kf / N, ruin_rate=float(d['ruined'].mean()),
                             practical_ruin_rate=float((d['min_w'] <= 0.10).mean()),
                             median_terminal_wealth=float(np.median(d['terminal'])),
                             median_mdd=float(np.median(d['mdd_step']))))
        if within == 'diffusive':
            for m, L in acc_unb.items():
                d = {f: np.concatenate([x[f] for x in L]) for f in L[0]}
                N = d['breach_step'].size
                k = int(d['breach_step'].sum())
                rows.append(dict(dgp='diffusive', variant='V1_code', n_rebalance=1, m=m, peak='', model='unbounded',
                                 n_paths=N, share_breaches_lambda_at_min='', breach_rate=k / N, breach_ci_lo=ST.wilson(k, N)[0],
                                 breach_ci_hi=ST.wilson(k, N)[1], breach_fine_rate='',
                                 ruin_rate=float(d['ruined'].mean()),
                                 practical_ruin_rate=float((d['min_w'] <= 0.10).mean()),
                                 median_terminal_wealth=float(np.median(d['terminal'])),
                                 median_mdd=float(np.median(d['mdd_step']))))
        if within != 'placebo':
            for v, pk in [(v, pk) for v in VARIANTS for pk in PEAK_MODES]:
                arr = np.stack([np.stack([cat[(v, n, m, pk)]['breach_step'] for m in M_LIST], axis=1)
                                for n in N_LIST], axis=1)          # (paths, n, m)
                reg = ST.gap_risk_regression(arr, N_LIST, M_LIST)
                r1 = arr[:, 0, M_LIST.index(3.0)].mean()
                r50 = arr[:, -1, M_LIST.index(3.0)].mean()
                reg['rate_n1_m3'] = float(r1)
                reg['rate_n50_m3'] = float(r50)
                reg['ratio_n50_to_n1_m3'] = float(r50 / r1) if r1 > 0 else None
                reg['toward_zero_criterion_met'] = (bool(r50 <= 0.10 * r1) if r1 > 0 else None)
                out[f'{within}/{v}/{pk}'] = reg
        print(f'  b: {within} done', flush=True)
    write_csv(os.path.join(outdir, 'b_gap.csv'), rows)
    with open(os.path.join(outdir, 'b_regression.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


# ============================================================================== part c
def part_c(seeds, outdir, procs=8):
    from multiprocessing import get_context
    from ext2lib import thesis_agents as TA
    tasks = [(a, s, mdl, seeds) for s in TA.SCENARIOS for mdl in ('bounded', 'unbounded') for a in TA.AGENTS]
    with get_context('spawn').Pool(procs) as pool:
        res = pool.map(TA.run_agent, tasks, chunksize=1)
    rows = []
    for r in res:
        n = r['terminal'].size
        k = int(r['ruined'].sum())
        pr = int((r['min_w'] <= TA.PRACTICAL_RUIN).sum())
        br = int((r['mdd'] > 0.20).sum())
        lo, hi = ST.bootstrap_ci(r['terminal'])
        rows.append(dict(scenario=r['scenario'], model=r['model'], agent=r['agent'], n=n, ruin_rate=k / n,
                         ruin_ci_lo=ST.wilson(k, n)[0], ruin_ci_hi=ST.wilson(k, n)[1],
                         practical_ruin_rate=pr / n, median_terminal_wealth=float(np.median(r['terminal'])),
                         mtw_ci_lo=lo, mtw_ci_hi=hi, breach_rate=br / n,
                         terminal_sha256=sha(r['terminal'])))
    write_csv(os.path.join(outdir, 'c_agents.csv'), rows)
    print('  c: done', flush=True)
    return rows


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['exploratory', 'confirmatory', 'scratch'])
    ap.add_argument('--parts', default='a,b,c')
    ap.add_argument('--n-seeds', type=int, default=None, help='exploratory/scratch only: use the first N seeds')
    args = ap.parse_args()
    parts = args.parts.split(',')
    outdir = os.path.join(HERE, 'results', args.phase)
    os.makedirs(outdir, exist_ok=True)
    if args.phase == 'confirmatory':
        assert args.n_seeds is None
        assert 'FREEZE ext2' in git('log', '--format=%s'), 'no FREEZE ext2 commit'
        dirty = [l for l in git('status', '--porcelain', '--', 'ext2').splitlines() if 'ext2/results/' not in l]
        assert not dirty, f'uncommitted changes in ext2: {dirty}'
        assert not os.path.exists(os.path.join(outdir, 'primary.json')), 'confirmatory run already done'
        seeds = dgp.CONFIRMATORY_SEEDS
        with open(os.path.join(HERE, 'results', 'exploratory', 'thresholds.json')) as fh:
            thr = json.load(fh)
    else:
        seeds = dgp.EXPLORATORY_SEEDS if args.n_seeds is None else dgp.EXPLORATORY_SEEDS[:args.n_seeds]
        assert seeds.max() < 100000, 'exploratory phase may not touch confirmatory seeds'
        thr = None
    t0 = time.time()
    info = {'phase': args.phase, 'git_head': git('rev-parse', 'HEAD'), 'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'python': platform.python_version(), 'numpy': np.__version__,
            'numba': __import__('numba').__version__, 'seeds': [int(seeds.min()), int(seeds.max()), int(seeds.size)],
            'parts': parts}
    primary = {}
    if 'a' in parts:
        prim_a, ainfo = part_a(seeds, args.phase, outdir, thr)
        info['a'] = ainfo
        primary['a'] = prim_a
    if 'b' in parts:
        primary['b'] = part_b(seeds, outdir)
    if 'c' in parts:
        part_c(seeds, outdir)
    if 'a' in primary and 'b' in primary:
        fam = [(f"{r['contrast']} {r['a']} - {r['b']}", r['p_signflip'], r) for r in primary['a']]
        for v in VARIANTS:
            reg = primary['b'][f'diffusive/{v}/peak_step']
            fam.append((f'{v} b1(log n) < 0', reg['p_one_sided_b1_lt_0'], reg))
            fam.append((f'{v} b2(m) > 0', reg['p_one_sided_b2_gt_0'], reg))
        adj = ST.holm([f[1] for f in fam])
        tests = [{'test': f[0], 'p_raw': f[1], 'p_holm': float(a)} for f, a in zip(fam, adj)]
        for t, f in zip(tests, fam):
            if 'mean_diff' in f[2]:
                t.update(mean_diff=f[2]['mean_diff'], ci95=[f[2]['ci_lo'], f[2]['ci_hi']])
        decisions = {}
        for v in VARIANTS:
            reg = primary['b'][f'diffusive/{v}/peak_step']
            p1 = next(t['p_holm'] for t in tests if t['test'] == f'{v} b1(log n) < 0')
            p2 = next(t['p_holm'] for t in tests if t['test'] == f'{v} b2(m) > 0')
            slopes = reg['b1_logn'] < 0 and reg['b2_m'] > 0 and p1 < 0.05 and p2 < 0.05
            tz = reg['toward_zero_criterion_met']
            decisions[v] = 'SUPPORTED' if slopes and tz else ('PARTLY SUPPORTED' if slopes else 'NOT SUPPORTED')
        with open(os.path.join(outdir, 'primary.json'), 'w') as fh:
            json.dump({'tests': tests, 'gap_risk_decisions': decisions,
                       'b_regressions': {k: v for k, v in primary['b'].items()}}, fh, indent=1)
    info['runtime_s'] = round(time.time() - t0, 1)
    with open(os.path.join(outdir, 'run_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    print('done in', info['runtime_s'], 's')


if __name__ == '__main__':
    main()
