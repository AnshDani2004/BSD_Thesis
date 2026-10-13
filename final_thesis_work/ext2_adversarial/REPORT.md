# Adversarial review of Extension 2 (honors thesis ablation, synthetic), 2026-10-13

Saved by the main session (the subagent could not write report files). Target: final_thesis_work/ext2/,
branch ext2-ablation; confirmatory run 5a50295 from FREEZE a8bd9ff; RESULTS.md eeb3fcc. Check scripts in
this folder (adv_a.py, adv_b.py, adv_tune.py, adv_cusum.py, out_*.json, pytest.log) import ext2lib
read-only; evaluation on confirmatory seeds 100000 to 101999, all tuning on exploratory seeds 0 to 999;
bootstrap seed 12345, 2,000 resamples. These are post hoc sensitivity checks, not confirmatory tests.

## Verdict
Machinery sound and reproducible; two plain-language conclusions overreach (the explanation of the
downstream null; "volatility makes detection slower" beyond A_ii 0.95). The CPPI headline survives and is
stronger stated causally; most breaches overshoot the floor by under 1 point. BOCPD and jump-model gaps are
real in sign, much smaller after tuning.

## Reproduced (no threat)
Primary-cell mean delays at the frozen thresholds (2.148, 2.292, 2.5235, 6.006, 4.1685, 10.594); breach
rates V1 0.607 / 0.5625, V2 0.0835 / 0.0015; two executions byte-identical (scores SHA-256 1cf23fc5...);
tests 22/22; thresholds.json identical in exploratory runs 1 and 2; detectors.py, dgp.py, stats.py
unchanged since PREREG v1 (a5ebfc0); seeds disjoint. An alarm already on at t = 124 occurs on 0.15 to 0.6%
of paths (fresh-onset rule moves delays by at most 0.1 step). Exact in-sample matching at 5.000 per 1,000:
A1 +0.150 [0.099, 0.204], A2 +0.245 [0.186, 0.306], A3 +3.18, A4 +7.33 (the realised-rate mismatch had
favoured the Vol-HMM variants).

## Threats, ranked
T1 (high): the downstream null comes from the CPPI floor, not the bounded model (a_downstream.csv, A_ii
0.95, target 5, thesis shock, D = 125). Bounded, no floor: always in 15.0, return-only HMM 120.7, Vol-HMM
137.6, Vol-HMM + CUSUM 131.7, jump model 16.7, oracle 223.9. Unbounded, floor wired: 29.1, 32.1, 32.4, 32.8,
28.8, 41.0. "Detectors slightly ahead (28.8 to 32.8 vs 29.1)" is wrong for the jump model.
T2 (high): A1 depends on A_ii. Significant secondary A1 cells (q < 0.05), faster / slower: 0.95 0 / 17;
0.97 6 / 15; 0.99 15 / 5. Primary-analogue cell at 0.99: -0.049 [-0.109, 0.013]. "Ordering same in all A_ii
panels" false at 0.99 (Vol-HMM 2.33 vs return-only 2.38; BOCPD run length 3.67 vs segment mean 4.68). 87% of
paths tied in the primary cell; matched on onsets A1 = +0.062 [0.011, 0.114].
T3 (high, framing): V1 breaches are knife-edge (n = 1, m = 3): median max drawdown among breachers 0.204,
P90 0.213; 10.2% of paths exceed 21%, 0.35% exceed 25%. Breach by minimum leverage on the same paths: 0 ->
0.0835, 0.001 -> 0.365, 0.005 -> 0.522, 0.01 -> 0.607, 0.02 -> 0.6945. V2 breaches are a strict subset of V1's
(167 of 167); minimum 0 removes 86.2% of V1 breaches at n = 1 and 99.7% at n = 50.
T4 (medium): A3/A4 from censoring and untuned defaults. 66.7% of A3 (0.95) from paths censored at 50 (93% at
0.99); medians tie at 2. BOCPD run-length mean delay 3.35 / 6.01 / 7.95 at horizons 10 / 50 / 125. Best of
111 variants tuned on exploratory seeds: BOCPD (hazard 1/250, vague prior, L = 5) mean 2.80, 0.4% never
detected, minus Vol-HMM + CUSUM +0.27 [0.11, 0.46]; jump model (lambda 0.5, W = 120) 5.09. Both on the grid
edge. Ranking holds.
T5 (medium): A2 depends on the false-alarm metric. The thesis CUSUM latches before the break on 65.95% of
paths. At matched per-path false-alarm probability (0.25): A2 +0.040 [-0.025, 0.104] (0.95), +0.062
[-0.003, 0.125] (0.97), +0.117 [0.052, 0.179] (0.99); matched on onsets +0.061 [0.004, 0.118]; a
non-latching CUSUM under occupancy matching +0.161 [0.108, 0.216].
T6 (medium): the bounded bear's small-bet growth comes from extreme right-tail steps. Quadrature reproduces
Kelly 0.0720 (growth 0.00225 per step); truncating t3 at |z| <= 5 makes growth negative at every f tested
(f = 0.05: -0.0011); excluding one-step log returns above 2 (simple returns above +639%, 0.30% of steps)
also makes f = 0.05 negative. Log-t3 implies an infinite mean simple return.
T7 (low): amendment 2 justified (under peak_every the agent's floor sits above the evaluated floor and the
jump negative control fails, b1 -0.437); it moved V1 against the prediction; the causal minimum-leverage
result is at n = 1, where both peak modes coincide. Under peak_every V1 at n = 50 drops to 0.037.
T8 (low): with 2,000 common-random-number paths any nonzero difference hits the p floor; claims should rest
on effect sizes. The logit is misspecified in m (breach 0.0045 at m = 1, 0.89 at m = 5): fine for the sign
test, do not read b2 as a size. run_ext2.py first committed after exploratory run 1 (94d8cb8). CPPI_SPEC
section 10 cites tests/test_fidelity.py, which does not exist (the check is
tests/test_cppi.py::test_matches_thesis_class_on_step_returns).

## Wording supported
See the "Review outcome" section at the top of ext2/RESULTS.md (items 1 to 5).
