"""Expected one-step log growth E[log(1 + f r)] of a constant bet f in each regime (amendment 3).

Bounded model: r = exp(x) - 1, log(1 + f r) = logaddexp(log(1 - f), log f + x), integrated by quadrature
against the regime density of x. Unbounded (thesis) model: r = x; the expectation is -inf for every f > 0
under Student-t shocks because P(x <= -1/f) > 0, so the per-step ruin probability is reported instead.
Deterministic (no random numbers). Writes results/bear_log_growth.json.
Usage (from final_thesis_work/): python3 ext2/bear_log_growth.py
"""
import json
import os

import numpy as np
from scipy import integrate, optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
REGIMES = {'bull': ('norm', 0.08, 0.15), 'small': ('t3', 0.00, 0.20), 'thesis': ('t3', -0.10, 0.30),
           'large': ('t3', -0.20, 0.45)}
FS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def growth(f, kind, mu, s):
    dist = stats.norm() if kind == 'norm' else stats.t(3)
    if f >= 1.0:
        return mu
    g = lambda z: dist.pdf(z) * np.logaddexp(np.log1p(-f), np.log(f) + mu + s * z)  # noqa: E731
    return integrate.quad(g, -np.inf, np.inf, limit=500)[0]


def main():
    out = {}
    for name, (kind, mu, s) in REGIMES.items():
        row = {'kind': kind, 'location': mu, 'scale': s,
               'bounded_E_log_growth': {str(f): growth(f, kind, mu, s) for f in FS}}
        res = optimize.minimize_scalar(lambda f: -growth(f, kind, mu, s), bounds=(1e-6, 1 - 1e-9),
                                       method='bounded', options={'xatol': 1e-6})
        row['bounded_kelly_fraction_in_0_1'] = float(res.x)
        row['bounded_kelly_growth'] = float(-res.fun)
        dist = stats.norm() if kind == 'norm' else stats.t(3)
        row['unbounded_P_r_le_minus_1_over_f'] = {str(f): float(dist.cdf((-1.0 / f - mu) / s)) for f in FS}
        out[name] = row
    with open(os.path.join(HERE, 'results', 'bear_log_growth.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
