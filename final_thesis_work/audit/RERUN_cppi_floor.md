# CPPI floor wiring fix and rerun (branch audit-cppi-floor, 2026-10-12)

NOT REPLICATED YET. Do not quote any number below until an independent re-implementation agrees.

## The fix
`experiments/generate_all_figures.py` and `dashboard/backend/app.py` called `agent.update(np.array([r]))`
for every agent, so `RiskConstrainedKelly` never saw its wealth: its peak and floor stayed at 1.0 and
its CPPI multiplier stayed at the 0.5 cap on every step. Both call sites now pass
`wealth=tracker.wealth` to the CPPI agent only (other agents' `update` takes no wealth argument).
No other code changed.

## Rerun
`cd final_thesis_work && MPLBACKEND=Agg python3 experiments/generate_all_figures.py` (about 30 s;
Python 3.10, numpy as installed). It regenerates the figures and `*_stats.json` under
`dashboard/frontend/public/results/`. Drawdown statistics: `python3 audit/audit_checks.py` function
`cppi(True)` (same paths, seeds 42 and 142).

Adversarial scenario (100 paths, T = 250, break at t = 125), Risk-Constrained CPPI:

| | before (floor not wired) | after (floor wired) |
|---|---|---|
| ruin rate | 0.00 | 0.00 |
| median terminal wealth | 1.073 | 4.105 |
| CPPI multiplier range | 0.5 to 0.5 | 0.01 to 0.5 |
| paths with max drawdown > 20% | 100 | 98 |
| median max drawdown | 0.794 | 0.220 |
| worst max drawdown | 0.990 | 0.845 |

So the floor now moves, but a 20% drawdown floor with one rebalance per step does not keep drawdown
within 20%: Student-t(3) shocks gap through it on 98 of 100 paths. Ruin stays at 0%.

## Other numbers that changed, and why (not caused by the fix)
* Thompson Sampling (all scenarios) and every agent in the Drift scenario use the unseeded global
  NumPy RNG (`np.random.normal` in `generate_all()`; Thompson's posterior draws), so they differ on every
  run. Example: Adversarial Thompson ruin 0.16 in the committed file, 0.12 in this run. These numbers are
  not reproducible as committed.
* Seeded agents not using the CPPI fix (Vol-HMM, EXP3, Naive Bayes, Softmax in Stationary and
  Adversarial) reproduce to floating-point precision.

The defended thesis PDF and its tables are unchanged by this branch.
