# Extension 2: independent replication (from the documents only)

What I read: `ext2/PREREG.md` (amendments 1 to 5) and `ext2/CPPI_SPEC.md`. Nothing else in ext2,
no results, no thesis code. The PREREG does not point to `simulation/environment.py` or
`distributions.py` for the DGP (it defines the DGP in section 2), so I did not read them either.
Code: `rep_a.py`, `rep_b.py`, `rep_c.py` in this directory; all numbers are in `replica.json`
(raw pieces in `a_out.json`, `b_out.json`, `c_out.json`). NumPy 2.4.4, SciPy 1.17.1, Python 3.13.2
(the PREREG names NumPy 2.2; I assume the PCG64 / standard_normal / chisquare / integers streams are
the same). Confirmatory seeds 100000 to 101999 (2,000 paths). Thresholds re-tuned by me on exploratory
seeds 0 to 999 as in PREREG section 5 (I did not see `thresholds.json`). Runtime: a 25 s, b 217 s,
c about 3 s, one process each.

## (a) Detection, primary cell (A_ii = 0.95, thesis shock, D = 125, H = 50), target 5 per 1,000

| detector | h (exploratory) | mean delay | median | p90 | never detected | FA per 1,000 (confirmatory) |
|---|---|---|---|---|---|---|
| hmm_ret | 1.3595 | 2.148 | 1 | 5 | 0.000 | 4.42 |
| volhmm | -0.8107 | 2.292 | 2 | 6 | 0.000 | 4.62 |
| volhmm_cusum | 3.1894 | 2.523 | 2 | 6 | 0.000 | 4.67 |
| bocpd_rl | 1.7877 | 6.006 | 2 | 11.1 | 0.050 | 4.36 |
| bocpd_mean | 0.0638 | 4.168 | 2 | 10 | 0.002 | 4.31 |
| jm | 18.4533 | 10.594 | 5 | 38 | 0.089 | 4.40 |

Exploratory FA at the tuned h is 5.00 per 1,000 for every detector (by construction). Targets 1 and 10
and the native h = 0 point are in `replica.json` (for example native h = 0: hmm_ret 1.540 delay at
26.4 FA per 1,000; volhmm 2.655 at 2.14; volhmm_cusum 1.659 at 41.9; jm FA 514 per 1,000).

Primary contrasts (mean paired difference of censored delays, positive = first detector slower):

| contrast | mean diff | 95% bootstrap CI | sign-flip p (10,000) | Holm p (family of 8) | decision |
|---|---|---|---|---|---|
| A1 volhmm - hmm_ret | +0.144 | [0.088, 0.199] | 1/10001 | 0.0008 | reject |
| A2 volhmm_cusum - volhmm | +0.232 | [0.171, 0.295] | 1/10001 | 0.0008 | reject |
| A3 bocpd_rl - volhmm_cusum | +3.483 | [3.022, 3.984] | 1/10001 | 0.0008 | reject |
| A4 jm - volhmm_cusum | +8.071 | [7.491, 8.678] | 1/10001 | 0.0008 | reject |

Reading: at matched false alarms the volatility feature makes detection slightly SLOWER, not faster
(A1 > 0), the CUSUM also slows it (A2 > 0), and BOCPD and the jump model are much slower than the
thesis Vol-HMM + CUSUM. Secondary A5 (bocpd_mean - volhmm_cusum) = +1.645 [1.416, 1.899].

### (a') Thesis-native threshold h = 0 (no tuning), A_ii = 0.95, thesis shock, D = 125, confirmatory

FA definition: share of pre-break steps t in [10, 125) with s_t > 0 (each flagged step counts), per
1,000 such steps, pooled over the 2,000 paths. Alarm onsets (s_t > 0 and s_{t-1} <= 0) reported
separately. Delay as above (first t >= 125 with s_t > 0, minus 125, censored at 50). For the HMMs
s_t > 0 is P(bear) > 0.5; for BOCPD and jm, h = 0 is simply the same cut applied to their scores
(P(l <= 10) > 0.5 for bocpd_rl; posterior segment mean < 0 for bocpd_mean; bear cheaper for jm).

| detector | FA steps per 1,000 | onsets per 1,000 | mean delay | median delay | never detected |
|---|---|---|---|---|---|
| hmm_ret | 26.41 | 13.69 | 1.540 | 1 | 0 |
| volhmm | 2.14 | 1.28 | 2.655 | 2 | 0 |
| volhmm_cusum | 41.88 | 10.06 | 1.659 | 1 | 0 |
| bocpd_rl | 90.70 | 36.08 | 1.464 | 1 | 0 |
| bocpd_mean | 43.88 | 24.36 | 2.329 | 1 | 0 |
| jm | 513.84 | 83.92 | 1.245 | 0 | 0 |

Alternative readings: volhmm_cusum with +2 from the next step 38.57 FA, 1.738 delay; jm with the
0.05 step-0 vol feature 513.15 FA, 1.246 delay. Stored under `native_threshold_h0` in `replica.json`.

## (b) Gap risk, confirmatory, step-grid breach rate (diffusive DGP, peak_step, primary)

| variant | n | m=1 | m=2 | m=3 | m=5 |
|---|---|---|---|---|---|
| V1_code | 1 | 0.0045 | 0.2185 | 0.6070 | 0.8945 |
| V1_code | 50 | 0.0040 | 0.1900 | 0.5625 | 0.8825 |
| V2_pure | 1 | 0.0000 | 0.0015 | 0.0835 | 0.5605 |
| V2_pure | 50 | 0.0000 | 0.0005 | 0.0015 | 0.0670 |

(full 5 x 4 grids for all DGPs, both peak modes, plus fine-grid breach, median terminal wealth and
mean max drawdown are in `replica.json`.)

Logistic regression logit P = b0 + b1 log n + b2 m, path bootstrap 2,000:

| run | b1 [95% CI] | b2 [95% CI] | p(b1<0), p(b2>0) |
|---|---|---|---|
| diffusive V1_code peak_step | -0.0435 [-0.0529, -0.0348] | 1.378 [1.322, 1.438] | 0.0005, 0.0005 |
| diffusive V2_pure peak_step | -0.7076 [-0.7449, -0.6721] | 1.588 [1.524, 1.660] | 0.0005, 0.0005 |
| jump V1_code peak_step (neg. control) | +0.0233 [0.0189, 0.0280] | 1.410 [1.348, 1.478] | |
| jump V2_pure peak_step (neg. control) | -0.0463 [-0.0559, -0.0373] | 1.505 [1.445, 1.572] | |
| diffusive V1_code peak_every (secondary) | -1.110 [-1.170, -1.051] | 0.893 [0.867, 0.922] | |
| diffusive V2_pure peak_every (secondary) | -1.546 [-1.641, -1.466] | 1.474 [1.410, 1.548] | |

Holm over the 8 primary tests: all reject (largest adjusted p 0.002).
Decisions: V1_code PARTLY SUPPORTED (slopes pass; n = 50 / n = 1 at m = 3 = 0.5625 / 0.6070 = 0.93,
fails the 10% toward-zero criterion). V2_pure SUPPORTED (0.0015 / 0.0835 = 0.018).
Jump negative control: |b1| < 0.1 for both variants (the CIs exclude 0, so it passes only through the
|b1| < 0.1 branch). Checks: V2_pure m = 1 has zero step-grid breaches at every n under all DGPs
(theorem holds); no-break placebo has zero step-grid breaches in every cell. Unbounded comparison,
V1_code n = 1: breach 0.322 / 0.8625 / 0.9755 / 0.9935 for m = 1/2/3/5, ruin 0 / 0 / 0.001 / 0.0015.

## (c) Thesis agents

Only the CPPI agent (CPPI_SPEC.md, n = 1, m = 3) is specified in the documents I may read. The other
five thesis agents (including Thompson Sampling, which PREREG stream 4 implies) are not specified, so
I did not replicate them. CPPI agent on stream-1 paths:

| scenario | model | thesis ruin | practical ruin | median terminal wealth [95% CI] | floor breach |
|---|---|---|---|---|---|
| Stationary | bounded | 0 | 0 | 77.49 [73.05, 81.08] | 0 |
| Stationary | unbounded | 0 | 0 | 36.72 [34.99, 38.82] | 0 |
| Adversarial | bounded | 0 | 0 | 8.10 [7.70, 8.50] | 0.6155 |
| Adversarial | unbounded | 0.001 | 0.001 | 4.61 [4.49, 4.77] | 0.971 |
| Drift | bounded | 0 | 0 | 2.83 [2.73, 2.91] | 0.0005 |
| Drift | unbounded | 0 | 0 | 2.06 [2.01, 2.11] | 0.0015 |

Thesis ruin is exactly 0 under bounded returns, as the PREREG requires.

## Ambiguities (and their effect when I computed both readings)

1. CUSUM timing (PREREG 5, "+2 from that step on"): primary reading applies +2 at the step where S_t
   first exceeds 3; alternative from the next step. volhmm_cusum mean delay 2.523 vs 2.583; A2 +0.232 vs
   +0.291, A3 +3.483 vs +3.424, A4 +8.071 vs +8.012. No decision changes.
2. Jump-model volatility feature at step 0 (population std of a single observation): primary reading 0;
   alternative uses the Vol-HMM convention (0.05 for fewer than 2 observations, +1e-6). This only
   touches windows that contain step 0 (t < 60, pre-break), but it moves the tuned threshold (18.45 vs
   17.18) and so the jm delay: 10.594 vs 8.730; A4 +8.071 vs +6.207. Sign and decision unchanged; the
   size of A4 depends on this. Worth stating explicitly in the PREREG.
3. JM convergence check "from the second pass on": comparing pass 1 with the median split or not gives
   identical scores (if pass 1 reproduces the split, the centroids and pass 2 are unchanged).
4. Sign-flip p formula not stated: I report (1 + count)/(10001); the raw count/10000 is 0 for all four.
   Bootstrap and sign-flip draw order from the 20261012 Generator is not stated, so CIs and p-values
   will not match bitwise (a fresh Generator per contrast / per regression here).
5. HMM emission scale 0.15 / 0.30 read as standard deviations, full log densities (normalising
   constants included). Bootstrap CIs are percentile intervals.
6. Ridge 1e-8 applied to all three coefficients (subtracted from the gradient, added to the Hessian);
   negligible here since no cell configuration is separated.
7. Fine-grid breach: wealth at every substep from buy-and-hold inside each rebalance interval, fine-grid
   running peak starting at W_0 = 1.
8. Not computed (not requested): outcome (a) downstream wealth, other A_ii / shock cells, the
   lambda-at-minimum diagnostic, the fake-break placebo.
