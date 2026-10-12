"""Independent replication: Risk-Constrained Kelly (CPPI) on the Adversarial Shock scenario.

Written from the thesis text only (04_methodology.tex Def. 4.x / Prop. cppi_bound,
03_foundations.tex eq. gaussian_kelly, 05_experiments.tex Exp. 4). Return paths and the
wealth/ruin definition come from verbatim copies of simulation/environment.py,
distributions.py and wealth_tracker.py (in ./simcopy, SHA-256 in REPORT.md).

Agent (primary reading):
  base Kelly fraction f_base = clip(mu_bull / sigma_bull^2, 0, 1)   (= 1.0)
  running peak  Wbar_t = max_{s<=t} W_s (includes W_0 = 1 and the current wealth)
  floor F_t = (1 - Dmax) * Wbar_t, cushion C_t = W_t - F_t
  lambda_t = clip(m * C_t / W_t, 0, 1)      (min(1, .) from the thesis; [0,1] from the notation table)
  f_t = lambda_t * f_base
Run: python3 replicate_cppi_adversarial.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simcopy.environment import MarketEnvironment  # noqa: E402
from simcopy.wealth_tracker import WealthTracker  # noqa: E402

N, T, T_SHOCK = 100, 250, 125
DMAX, M = 0.20, 3.0
MU_BULL, SIG_BULL = 0.08, 0.15


def make_returns():
    bull = MarketEnvironment({'bull': {'mu': 0.08, 'sigma': 0.15}}, np.array([[1.0]]), seed=42)
    bear = MarketEnvironment({'bear': {'mu': -0.10, 'sigma': 0.30, 'df': 3}}, np.array([[1.0]]), seed=142)
    R = bull.simulate_batch(N, T).returns.copy()
    Rb = bear.simulate_batch(N, T).returns
    R[:, T_SHOCK:] = Rb[:, T_SHOCK:]
    return R


class CPPIKelly:
    """base: 'oracle_bull' (clip(mu/sigma^2,0,1)), 'half' (0.5), or 'beta' (Beta(1,1) posterior on
    y=1[r>0], binary Kelly with b=1: clip(2p-1,0,1))."""

    def __init__(self, base='oracle_bull', wealth_updates=True):
        self.base = base
        self.wealth_updates = wealth_updates
        self.W = 1.0
        self.peak = 1.0
        self.a, self.b = 1.0, 1.0

    def f_base(self):
        if self.base == 'oracle_bull':
            return float(np.clip(MU_BULL / SIG_BULL ** 2, 0.0, 1.0))
        if self.base == 'half':
            return 0.5
        if self.base == 'beta':
            p = self.a / (self.a + self.b)
            return float(np.clip(2 * p - 1, 0.0, 1.0))
        raise ValueError(self.base)

    def leverage(self):
        floor = (1.0 - DMAX) * self.peak
        cushion = self.W - floor
        return float(np.clip(M * cushion / self.W, 0.0, 1.0))

    def act(self):
        lam = self.leverage()
        return lam * self.f_base(), lam

    def observe(self, r, new_wealth):
        if r > 0:
            self.a += 1
        else:
            self.b += 1
        if self.wealth_updates:
            self.W = new_wealth
            self.peak = max(self.peak, new_wealth)


def run(R, base, wealth_updates):
    term = np.empty(N)
    mdd = np.empty(N)
    ruined = np.zeros(N, bool)
    lam_all = []
    f_all = []
    for i in range(N):
        ag = CPPIKelly(base, wealth_updates)
        wt = WealthTracker(initial_wealth=1.0)
        path_mdd = 0.0
        for t in range(T):
            f, lam = ag.act()
            lam_all.append(lam)
            f_all.append(f)
            r = R[i, t]
            ok = wt.update(f, r)
            path_mdd = max(path_mdd, wt.current_drawdown)  # 1.0 if ruined
            if not ok:
                break
            ag.observe(r, wt.wealth)
        ruined[i] = wt.is_ruined
        term[i] = wt.wealth
        mdd[i] = path_mdd
    lam_all = np.array(lam_all)
    f_all = np.array(f_all)
    return {
        'base': base,
        'wealth_updates': wealth_updates,
        'ruin_rate': float(ruined.mean()),
        'n_ruined': int(ruined.sum()),
        'median_terminal_wealth': float(np.median(term)),
        'mean_terminal_wealth': float(np.mean(term)),
        'n_paths_mdd_gt_20pct': int((mdd > 0.20).sum()),
        'median_mdd': float(np.median(mdd)),
        'worst_mdd': float(mdd.max()),
        'lambda_min': float(lam_all.min()),
        'lambda_max': float(lam_all.max()),
        'f_min': float(f_all.min()),
        'f_max': float(f_all.max()),
    }


def main():
    R = make_returns()
    info = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'returns_sha256': __import__('hashlib').sha256(np.ascontiguousarray(R).tobytes()).hexdigest(),
        'returns_min': float(R.min()),
        'n_bear_returns_below_-1/0.6': int((R[:, T_SHOCK:] < -1 / 0.6).sum()),
        'n_bear_returns_below_-1/3': int((R[:, T_SHOCK:] < -1 / 3).sum()),
    }
    out = {'info': info, 'runs': []}
    for base in ['oracle_bull', 'half', 'beta']:
        for wu in [True, False]:
            out['runs'].append(run(R, base, wu))
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, 'results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
