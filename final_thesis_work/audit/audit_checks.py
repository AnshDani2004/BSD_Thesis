"""Resume-audit checks (added 2026-10-12). Read-only use of the thesis code: nothing in
simulation/ or experiments/ is changed, and the thesis's own outputs are not overwritten.

Every check replays the Adversarial Shock scenario of experiments/generate_all_figures.py
exactly: 100 paths, T = 250, bull N(0.08, 0.15^2) from seed 42, bear Student-t(3) with
mu = -0.10, scale 0.30 from seed 142, break at t = 125.

1. lag_ablation: detection lag of simulation.hmm_refined.VolAugmentedHMM with the emission
   parameters of create_vol_augmented_hmm_kelly(1, 0.08, 0.15, -0.10, 0.30), switching the
   volatility feature and the CUSUM trigger on and off, at A_ii in {0.99, 0.98, 0.95, 0.80}.
   "Vol off" gives both regimes the same Gamma volatility emission, so that term cancels when
   the beliefs are normalised (a return-only HMM). "CUSUM off" sets cusum_threshold = inf.
   Lag = (first t >= 125 with P(bear) > 0.5) - 125, the convention of
   bayesian-sequential-decisions-thesis/experiments/exp03_hmm_sensitivity.py. Pre-break false
   alarms are reported alongside.
2. vol_hmm_ruin_by_aii: ruin and median terminal wealth of the Vol-HMM agent as A_ii varies
   (A_ii = 0.95 is the factory default and must reproduce Adversarial_stats.json).
3. cppi_floor: the CPPI agent replayed as generate_all_figures.py calls it
   (agent.update(np.array([r])), no wealth argument), and with wealth passed in.

Usage (from final_thesis_work/):  python audit/audit_checks.py [path_to_final_thesis_work]
Writes audit/results/audit_checks.json. Deterministic (fixed seeds, no other randomness).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from simulation.environment import MarketEnvironment
from simulation.hmm_refined import VolAugmentedHMM, create_vol_augmented_hmm_kelly
from simulation.risk_constrained import RiskConstrainedKelly
from simulation.wealth_tracker import WealthTracker

N, T = 100, 250
B = T // 2
env_bull = MarketEnvironment({'bull': {'mu': 0.08, 'sigma': 0.15}}, np.array([[1.0]]), seed=42)
env_bear = MarketEnvironment({'bear': {'mu': -0.10, 'sigma': 0.30, 'df': 3}}, np.array([[1.0]]), seed=142)
R = np.copy(env_bull.simulate_batch(N, T).returns)
R[:, B:] = env_bear.simulate_batch(N, T).returns[:, B:]
AII = [0.99, 0.98, 0.95, 0.80]


def emission_params(vol_on):
    bull = {'mu': 0.08, 'sigma': 0.15, 'vol_shape': 2.0, 'vol_scale': 0.15 / 2}
    bear = {'mu': -0.10, 'sigma': 0.30, 'vol_shape': 4.0, 'vol_scale': 0.30 / 2}
    if not vol_on:
        bear = dict(bear, vol_shape=bull['vol_shape'], vol_scale=bull['vol_scale'])
    return {'bull': bull, 'bear': bear}


def lag_row(vol_on, cusum_on, aii):
    lags, censored, bear_at_124, frac_pre, cusum_pre = [], 0, 0, [], 0
    for s in range(N):
        h = VolAugmentedHMM(emission_params(vol_on), max_persistence=aii,
                            cusum_threshold=(3.0 if cusum_on else np.inf))
        det, pre = None, 0
        for t in range(T):
            h.update(float(R[s, t]))
            p = h.get_regime_probability('bear')
            if t < B and p > 0.5:
                pre += 1
            if t == B - 1:
                bear_at_124 += int(p > 0.5)
                cusum_pre += int(h.cusum_triggered)
            if t >= B and det is None and p > 0.5:
                det = t - B
        if det is None:
            det, censored = T - B, censored + 1
        lags.append(det)
        frac_pre.append(pre / B)
    lags = np.array(lags)
    return {'vol_feature': vol_on, 'cusum': cusum_on, 'A_ii': aii,
            'mean_lag': round(float(lags.mean()), 2), 'median_lag': float(np.median(lags)),
            'p90_lag': round(float(np.percentile(lags, 90)), 2), 'n_never_detected': censored,
            'paths_in_bear_at_t124': bear_at_124,
            'mean_share_prebreak_steps_flagged_bear': round(float(np.mean(frac_pre)), 4),
            'paths_cusum_fired_before_break': cusum_pre}


def vol_hmm_ruin(aii):
    fin, ruined = [], 0
    for s in range(N):
        a = create_vol_augmented_hmm_kelly(1, 0.08, 0.15, -0.10, 0.30)
        p = {r.name: {'mu': r.mu, 'sigma': r.sigma, 'vol_shape': r.vol_shape, 'vol_scale': r.vol_scale}
             for r in a.hmm.regimes}
        a.hmm = VolAugmentedHMM(p, vol_window=5, max_persistence=aii)
        tr = WealthTracker(1.0)
        for t in range(T):
            f = a.act()[0]
            tr.update(f, R[s, t])
            a.update(np.array([R[s, t]]))
        fin.append(tr.wealth)
        ruined += int(tr.get_state().is_ruined)
    return {'A_ii': aii, 'ruin_rate': ruined / N, 'median_terminal_wealth': round(float(np.median(fin)), 4)}


def cppi(feed_wealth):
    levs, mdds, fin, ruined, fmax = [], [], [], 0, 0.0
    for s in range(N):
        a = RiskConstrainedKelly(1, max_drawdown=0.20, cppi_multiplier=3.0)
        tr = WealthTracker(1.0)
        peak, mdd = 1.0, 0.0
        for t in range(T):
            f = a.act()[0]
            fmax = max(fmax, f)
            tr.update(f, R[s, t])
            if feed_wealth:
                a.update(np.array([R[s, t]]), wealth=tr.wealth)
            else:
                a.update(np.array([R[s, t]]))
            peak = max(peak, tr.wealth)
            mdd = max(mdd, (peak - tr.wealth) / peak)
        levs.extend(a.leverage_history)
        mdds.append(mdd)
        fin.append(tr.wealth)
        ruined += int(tr.get_state().is_ruined)
    mdds = np.array(mdds)
    return {'wealth_passed_to_agent': feed_wealth,
            'cppi_multiplier_min': round(float(np.min(levs)), 6), 'cppi_multiplier_max': round(float(np.max(levs)), 6),
            'max_bet_fraction': round(fmax, 6), 'ruin_rate': ruined / N,
            'median_terminal_wealth': round(float(np.median(fin)), 4),
            'paths_max_drawdown_over_20pct': int((mdds > 0.20).sum()),
            'median_max_drawdown': round(float(np.median(mdds)), 4), 'worst_max_drawdown': round(float(mdds.max()), 4)}


if __name__ == '__main__':
    out = {'scenario': {'n_paths': N, 'T': T, 'break_t': B, 'seeds': {'bull': 42, 'bear': 142},
                        'source': 'experiments/generate_all_figures.py Adversarial'},
           'lag_ablation': [], 'vol_hmm_ruin_by_aii': [], 'cppi_floor': []}
    for aii in AII:
        for vol_on, cusum_on in [(True, True), (False, False), (False, True), (True, False)]:
            out['lag_ablation'].append(lag_row(vol_on, cusum_on, aii)); print(out['lag_ablation'][-1])
    for aii in AII:
        out['vol_hmm_ruin_by_aii'].append(vol_hmm_ruin(aii)); print(out['vol_hmm_ruin_by_aii'][-1])
    for fw in (False, True):
        out['cppi_floor'].append(cppi(fw)); print(out['cppi_floor'][-1])
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'audit_checks.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
