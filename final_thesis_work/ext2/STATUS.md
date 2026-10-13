# STATUS: Extension 2 (honors thesis ablation), 2026-10-12

Every number produced so far is NOT YET REPLICATED. Synthetic data only. Nothing has been pushed.

## Where

Thesis clone `existing/BSD_Thesis`, local branch `ext2-ablation` (created from `audit-cppi-floor` at
388ebf5). Everything new is under `final_thesis_work/ext2/`. The only change outside it is
`simulation/agents.py`: an optional `rng` argument for Thompson Sampling and epsilon-greedy, which
defaults to the old behaviour. Not pushed, main untouched.

| commit | what |
|---|---|
| cd20b2c | CPPI_SPEC.md written from the code; explicit-rng option in agents.py |
| a5ebfc0 | PREREG version 1, before any outcome computation; library code |
| 94d8cb8 | exploratory run 1 outputs (`results/exploratory_v1/`); amendments 1 to 3 |
| a8bd9ff | FREEZE ext2 (amendments 1 to 5, exploratory run 2, frozen thresholds; 22/22 tests pass) |
| 5a50295 | confirmatory run (seeds 100000 to 101999, run once from a8bd9ff), figures, digest |
| eeb3fcc | RESULTS.md |

## Done

* CPPI agent spec from the code (`CPPI_SPEC.md`). PREREG with an amendment log (`PREREG.md`).
* All random generators seeded per path through `SeedSequence([seed, stream])`, including Thompson's
  posterior draws and the Drift scenario (in the ext2 runners).
* Bounded return model (log return x, r = exp(x) - 1), with the thesis model kept as a labelled
  comparison.
* 22 known-answer and fidelity tests in `tests/`. All pass, both before the freeze and at the freeze.
* Exploratory runs on seeds 0 to 999, then a single confirmatory run. Results in `RESULTS.md`; figures in
  `results/confirmatory/fig_frontier.png`, `fig_frontier_shocks.png` and `fig_gap.png`.

## Not done / planned (someone else)

* Independent replication from PREREG.md and CPPI_SPEC.md only, and an adversarial review.
* The thesis dashboard script `experiments/generate_all_figures.py` is unchanged: its Drift scenario and
  Thompson agent still use the unseeded global RNG, and its committed outputs are not regenerated.

## How to rerun (from `final_thesis_work/`; about 20 min for the confirmatory run on 8 cores)

    export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/ext2_numba
    (cd ext2 && python3 -m pytest -q -p no:cacheprovider tests)          # 22 tests
    python3 ext2/run_ext2.py --phase exploratory                          # seeds 0-999
    python3 ext2/run_ext2.py --phase confirmatory                         # refuses if already run or tree dirty
    python3 ext2/bear_log_growth.py
    MPLBACKEND=Agg python3 ext2/make_figures.py confirmatory
    python3 ext2/summarize.py confirmatory > ext2/results/confirmatory/digest.json

The confirmatory guard refuses to run if `results/confirmatory/primary.json` already exists. To
reproduce the run, move that file aside first. `NUMBA_CACHE_DIR` keeps numba from rewriting the
`__pycache__` files that are tracked in this repo.

## Open questions for Ansh

* [NEEDS ANSH: which return model should the thesis narrative use? Under the prescribed bounded
  log-Student-t model every regime, including the thesis bear, has positive expected log growth for small
  bets (Kelly fraction 0.072 in the bear), so "go to cash" is never optimal and the detectors have almost
  no wealth value for a persistent shock. Option 1: keep the bounded model and say so. Option 2: add a
  bounded bear with a genuinely negative small-bet growth (for example a bounded-above shock) as a new,
  separately pre-registered study.]
* [NEEDS ANSH: amendment 2 changed the primary gap-risk analysis after the exploratory data (peak tracked
  at step ends instead of at every rebalance). Under the original version the thesis agent would have been
  rated SUPPORTED; under the amended one it is PARTLY SUPPORTED with a negligible effect. Both are reported.
  Do you accept the amendment as the primary?]
* [NEEDS ANSH: the 0.01 minimum leverage in RiskConstrainedKelly causes most of the floor breaches (88%).
  Should the thesis text or erratum describe the CPPI agent as it is coded (minimum 0.01, maximum 0.5,
  Beta-posterior binary Kelly base), and should the 0.01 minimum be called a design flaw?]
* [NEEDS ANSH: should the dashboard outputs be regenerated with seeded Thompson and Drift draws (on a
  branch, unpushed)?]
* [NEEDS ANSH: nothing here goes on the resume or website until it is replicated. Which, if any, of the
  three headline candidates in RESULTS.md section 6 would you want considered afterwards?]

## Ansh's decisions (2026-10-13)
Bounded log-return model primary; amendment 2 recorded as a disclosed deviation with both versions
reported; CPPI agent described as coded in the README. Push of `ext2-ablation` (which includes the
`audit-cppi-floor` commits) only after Ansh approves the README.

