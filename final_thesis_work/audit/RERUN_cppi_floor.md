# CPPI floor wiring fix and rerun (branch audit-cppi-floor, 2026-10-12)

NOT REPLICATED. An independent re-implementation from the thesis text (2026-10-12, `replication_2026-10-12/`)
did not reproduce these numbers. Do not quote any number below.

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

## Independent replication (2026-10-12): does not match

A separate agent re-implemented the CPPI agent from the thesis text only (it could read the thesis, the
return-path generator `simulation/environment.py` and `distributions.py`, and `wealth_tracker.py`; not
`risk_constrained.py`, the experiment scripts or the audit). Same 100 paths (return-matrix SHA-256
4987c90a...e9c8). Script and output: `replication_2026-10-12/replicate_cppi_adversarial.py`, `results.json`. The script imports
verbatim copies of the three allowed files from `./simcopy/`; to rerun, copy `simulation/environment.py`,
`distributions.py` and `wealth_tracker.py` (with an empty `__init__.py`) into `replication_2026-10-12/simcopy/`.

The thesis text does not pin down the CPPI agent: which Kelly fraction is scaled (the replica's primary
reading is the Gaussian fraction mu/sigma^2 clipped to 1; the code uses a Beta-posterior binary Kelly
fraction), and the leverage bounds (the replica uses lambda in [0, 1]; the code clips to [0.01, 0.5]).
Under the replica's readings, with the wealth update on, ruin is 1% (not 0%) and median terminal wealth
is 26.4x (primary), 10.2x (base fraction 0.5) or 5.05x (Beta-posterior base); 75 to 100 paths exceed a
20% drawdown. All readings agree that the 20% floor is breached on most paths, so the thesis's
drawdown-bound proposition does not hold on these paths. The rerun's 0% ruin and 4.10x are therefore
specific to the code's unstated choices and are not replicated.

To replicate properly, the spec must state the base Kelly fraction and the leverage bounds; then rerun
both implementations.
