# RESULTS: Extension 2 (detector ablation, CPPI gap risk, bounded returns)

EVERY NUMBER IN THIS FILE IS NOT YET REPLICATED. An independent replication (from `PREREG.md` and
`CPPI_SPEC.md` only) and an adversarial review have not been run. Do not quote any number anywhere
until they have.

Synthetic data only. Confirmatory run: seeds 100000 to 101999 (2,000 paths per cell), run once from the
FREEZE commit a8bd9ff on branch `ext2-ablation`, outputs committed in 5a50295. Paths below are relative
to `final_thesis_work/ext2/results/confirmatory/`. `digest.json` (from `summarize.py`) collects every
number quoted here; the file and key are given for each one.

## 0. Summary in plain language

1. **At matched false-alarm rates, the thesis's additions do not detect the break faster.** The
   return-only HMM is the fastest of the six detectors. Adding the rolling-volatility feature makes
   detection 0.14 steps slower on average, and the CUSUM trigger adds another 0.23 steps. BOCPD and the
   jump model are much slower. The one benefit of the volatility feature is fewer false alarms at the
   thesis's own threshold (P(bear) > 0.5): 2.1 versus 26.4 per 1,000 pre-break steps.
2. **The thesis CPPI agent's floor breaches are mostly not gap risk.** Under bounded returns it still
   breaches its 20% floor on 60.7% of paths, and rebalancing 50 times per step only brings that down to
   56.3%. In 88% of the breaches the leverage multiplier was sitting at its 0.01 minimum. That minimum
   keeps a position open at the floor, so wealth can drift through the floor slowly. The textbook CPPI
   (minimum 0) breaches on 8.35% of paths at one rebalance per step and on 0.15% at 50, which is what gap
   risk predicts.
3. **Under bounded returns, ruin as the thesis defines it is exactly zero for every agent, because it has
   to be.** Every agent bets a fraction between 0 and 1, and r > -1, so 1 + f r > 0 always. The thesis's
   nonzero ruin rates come entirely from simulated returns below -100%.
4. **Under the bounded model, detection has almost no wealth value when the shock lasts.** With a shock
   that persists to the end, staying invested ("always in") has a median terminal wealth (120.3) as high
   as any detector's. The reason is in the model: with log-Student-t returns, small bets have positive
   expected log growth even in the "bear" regime (Kelly fraction 0.072 in the thesis bear), so going to
   cash is never Kelly-optimal. Detection does pay when the shock is short (5 or 20 steps) and severe.

## 1. Outcome a: detection frontier

### 1.1 Primary contrasts (pre-registered, Holm over 8 primary tests)

Cell: A_ii = 0.95, false-alarm target 5 per 1,000, thesis shock (Student-t(3), location -0.10,
scale 0.30) lasting to the end, horizon 50. Statistic: mean over 2,000 paths of the paired difference in
censored detection delay (steps). Source: `primary.json` `tests`; `digest.json`
`a_contrasts_primary_cell_all`.

| test | mean difference (a - b) | 95% bootstrap CI | Holm p | share of paths where a is faster / tied |
|---|---|---|---|---|
| A1 Vol-HMM - return-only HMM | +0.144 | [0.088, 0.199] | 0.0008 | 0.051 / 0.870 |
| A2 Vol-HMM + CUSUM - Vol-HMM | +0.232 | [0.171, 0.295] | 0.0008 | 0.054 / 0.777 |
| A3 BOCPD (run length) - Vol-HMM + CUSUM | +3.48 | [3.02, 3.98] | 0.0008 | 0.092 / 0.604 |
| A4 jump model - Vol-HMM + CUSUM | +8.07 | [7.49, 8.68] | 0.0008 | 0.012 / 0.271 |

All four reject "no difference", and all four go the same way: the more elaborate detector is slower.
The A1 and A2 effects are small (a fraction of a step) and most paths are tied: 87% tie for A1, 78% for
A2. The p-values are at the floor of the 10,000-flip sign-flip test (raw p = 1/10,001).

### 1.2 Delay at the primary cell (`table_frontier_thesis_D125.md`; `digest.json` `a_frontier_thesis_D125`)

| detector (A_ii 0.95, target 5) | realised FA /1,000 | mean delay | median | P90 | never detected within 50 |
|---|---|---|---|---|---|
| return-only HMM | 4.42 | 2.15 | 1 | 5 | 0.000 |
| Vol-HMM | 4.62 | 2.29 | 2 | 6 | 0.000 |
| Vol-HMM + CUSUM | 4.67 | 2.52 | 2 | 6 | 0.000 |
| BOCPD (run length) | 4.36 | 6.01 | 2 | 11.1 | 0.050 |
| BOCPD (segment mean) | 4.31 | 4.17 | 2 | 10 | 0.002 |
| jump model | 4.40 | 10.59 | 5 | 38 | 0.089 |

Thesis-native threshold (h = 0), A_ii = 0.95: false alarms per 1,000 pre-break steps were 26.41
(return-only HMM), 2.14 (Vol-HMM), 41.88 (Vol-HMM + CUSUM), 90.70 (BOCPD run length), 43.88 (BOCPD
segment mean) and 513.84 (jump model). Mean delays were 1.54, 2.65, 1.66, 1.46, 2.33 and 1.25. So at the
thesis's own threshold the CUSUM version looks fast only because it raises about 20 times as many false
alarms as the Vol-HMM without CUSUM.

Figure: `fig_frontier.png` (one panel per A_ii, thesis shock to the end) and `fig_frontier_shocks.png`
(every shock cell at A_ii = 0.95). The ordering is the same in all three A_ii panels. The three HMMs sit
close together near 2 to 3 steps, BOCPD is above them and the jump model is highest.

### 1.3 All 81 matched cells (secondary, BH; `digest.json` `a_contrast_summary_all_matched_cells`)

The 81 cells are 3 A_ii x 3 targets x 9 shock cells. The counts below are significant cells (q < 0.05).

| contrast | a faster | a slower | range of mean difference |
|---|---|---|---|
| A1 Vol-HMM - return-only | 21 | 37 | -0.141 to +1.909 |
| A2 + CUSUM - Vol-HMM | 8 | 69 | -1.208 to +1.253 |
| A3 BOCPD run length - Vol-HMM + CUSUM | 4 | 70 | -0.063 to +15.502 |
| A4 jump model - Vol-HMM + CUSUM | 0 | 81 | +0.661 to +28.517 |
| A5 BOCPD segment mean - Vol-HMM + CUSUM | 0 | 81 | +0.157 to +9.710 |

The volatility feature helps in some cells (21 of 81) and hurts in more (37 of 81). This is a mixed
result, not a clean null in either direction.

### 1.4 Checks on the frontier

* **Realised false-alarm rates came in below target on the confirmatory seeds.** Ranges across
  detectors and A_ii (`digest.json` `a_realised_fa_range`): target 1 gave 0.59 to 0.94, target 5 gave
  3.84 to 4.76, target 10 gave 8.12 to 10.08. This is within the pre-set tolerance of the held-out
  tuning test, but "matched" is approximate. At target 5 the detectors differ by up to about 0.9 false
  alarms per 1,000.
* **Fake-break placebo** (no shock; share of paths with an alarm in steps 125 to 174; `a_frontier.csv`,
  rows shock = small, D = 5): at target 5 it was 0.104 to 0.148 for five detectors and 0.0545 for the
  jump model. That is what false alarms alone produce, so a "detection" can come from a false alarm on
  roughly one path in ten.
* **The pre-break scores are identical across all 9 shock cells for every detector** (SHA-256 recorded
  in `run_info.json` `a`). The no-look-ahead unit test passes.

### 1.5 Downstream wealth (descriptive; `digest.json` `a_downstream_thesis_D125_aii095`)

Kelly sizing under the wired CPPI floor (m = 3), base fraction 1 when the alarm is off and 0 when it is
on. Thesis shock to the end, A_ii = 0.95, target 5, bounded model:

| sizing | median terminal wealth [95% CI] | floor breach | ruin | unbounded-model median wealth / ruin |
|---|---|---|---|---|
| return-only HMM | 114.1 [108.9, 121.0] | 0.999 | 0 | 32.1 / 0.0070 |
| Vol-HMM | 113.0 [108.6, 119.2] | 0.994 | 0 | 32.4 / 0.0065 |
| Vol-HMM + CUSUM | 112.8 [108.2, 119.6] | 0.994 | 0 | 32.8 / 0.0050 |
| BOCPD run length | 116.5 [109.8, 122.5] | 1.000 | 0 | 29.6 / 0.0070 |
| BOCPD segment mean | 115.4 [108.8, 120.3] | 1.000 | 0 | 31.1 / 0.0070 |
| jump model | 116.3 [109.4, 123.2] | 1.000 | 0 | 28.8 / 0.0070 |
| always in (no detector) | 120.3 [111.7, 126.0] | 1.000 | 0 | 29.1 / 0.0070 |
| oracle timing (cash during the shock) | 122.9 [115.2, 129.1] | 0.038 | 0 | 41.0 / 0 |

* Thesis-definition ruin is 0 in every one of the downstream cells (`digest.json` `a_downstream_max_ruin_bounded` = 0.0).
* **Null for a persistent shock.** No detector beats "always in" on median wealth, and the confidence
  intervals overlap. Under the unbounded comparison model the detectors are slightly ahead of
  always-in (28.8 to 32.8 versus 29.1), and the oracle timing benchmark is clearly ahead (41.0).
* **Short shocks** (`a_downstream.csv`, A_ii 0.95, target 5). Thesis shock lasting 20 steps:
  return-only HMM 1,005, Vol-HMM + CUSUM 997, always in 714, oracle 6,853. Large shock lasting 20
  steps: 553, 564, 338 and 6,853. Small shock lasting to the end: always in 148.3, return-only HMM 138.8.
  Detection helps against short, severe shocks, but every detector stays far below the oracle. After even
  a detected shock, the CPPI floor (80% of the old peak) keeps the agent de-levered for a long time.
* With the floor wired in, the downstream agents breach the floor on almost every path when the shock
  persists, because the base fraction is 1, three times the cushion.

## 2. Outcome b: CPPI gap risk

Thesis bear from step 125, 50 substeps per step, bounded model. "Breach" means wealth on the step-end
grid falls more than 20% below its running peak. Primary peak tracking is `peak_step` (amendment 2).
Sources: `b_gap.csv`, `b_regression.json`, `primary.json`.

### 2.1 Primary tests and decisions

| variant (diffusive DGP, peak_step) | breach at n = 1, m = 3 [Wilson CI] | at n = 50, m = 3 | b1 (log n) [95% CI] | b2 (m) [95% CI] | Holm p (b1, b2) | decision |
|---|---|---|---|---|---|---|
| V1_code (thesis agent, lambda >= 0.01) | 0.607 [0.585, 0.628] | 0.5625 | -0.0435 [-0.053, -0.035] | 1.378 [1.322, 1.438] | 0.002, 0.002 | PARTLY SUPPORTED |
| V2_pure (lambda >= 0) | 0.0835 | 0.0015 | -0.705 [-0.743, -0.670] | 1.588 [1.524, 1.659] | 0.002, 0.002 | SUPPORTED |

* **What "PARTLY SUPPORTED" means for the thesis agent:** both slopes have the predicted sign and pass
  Holm, but the effect of rebalancing is tiny (a 7% relative drop from n = 1 to n = 50, ratio 0.927 against
  the pre-registered maximum of 0.10). In plain terms, more frequent rebalancing barely helps the thesis
  agent. Its breaches are not mainly gap risk.
* **Why:** in 88.2% of V1 breaches at n = 1, m = 3, lambda was at its 0.01 minimum (or the cushion was
  already gone) in the interval where wealth crossed the floor (`b_gap.csv`
  `share_breaches_lambda_at_min`). At n = 50 that share rises to 1.00, so the breaches that rebalancing
  cannot remove are all of this kind. V2 never has lambda at the minimum when it breaches (share 0.00).
  The minimum keeps exposure above m times the cushion near the floor, and the bear's drift then carries
  wealth through the floor without any gap.
* **Pure CPPI behaves as gap-risk theory says:** the breach rate falls 98% (0.0835 to 0.0015) as
  rebalancing goes from 1 to 50 per step, and it rises with m (m = 5: 0.5605 at n = 1, 0.0685 at n = 50).
* **Theorem check:** V2 with m = 1 breaches on 0 paths in every n and DGP cell, as it must under bounded
  returns. V1 with m = 1 breaches on 0.0045 at n = 1 (`b_gap.csv`), all with lambda at the minimum.

### 2.2 Negative control and secondary analyses

* **Jump DGP (whole shock in one substep), peak_step:** b1 = +0.023 [0.019, 0.028] (V1) and
  -0.046 [-0.056, -0.037] (V2). Both are small (|b1| < 0.1, the pre-registered "near zero"), though their
  CIs exclude 0. V2 at m = 3 goes from 0.0835 to 0.072. Rebalancing does not help against a jump, as
  expected.
* **The version first written in the spec (`peak_every`, the peak updated at every rebalance) would have
  given the opposite headline for the thesis agent.** V1 at m = 3 falls from 0.607 to 0.037 (b1 = -1.110),
  which passes the toward-zero rule. Under the jump negative control it still falls (0.607 to 0.2105,
  b1 = -0.437). The fall comes from the agent seeing intra-step highs and raising its own floor, not from
  rebalancing exposure faster. This is why amendment 2 was made before the freeze. Both versions are
  reported here.
* **Fine-grid breach** (any of the 50 substeps below 80% of the fine-grid peak): V1 n = 1, m = 3: 0.9685.
  Most paths dip through the floor inside a step even when the step-end grid shows no breach.
* **Unbounded (thesis) model, V1, n = 1:** breach 0.9755 [0.968, 0.981] at m = 3 versus 0.607 bounded.
  Bounding the returns removes about 38% of the breaching paths (0.9755 to 0.607) but not most of them.
* **No-break placebo:** 0 step-grid breaches in every variant, n and m cell.

## 3. Outcome c: thesis agents, bounded returns, all RNGs seeded (`c_agents.csv`; `digest.json` `c_agents`)

2,000 paths per cell. Thesis ruin under bounded returns is 0 for all 18 agent-scenario cells (Wilson upper
bound 0.0019), as pre-stated. Median terminal wealth [95% bootstrap CI]:

| agent | Stationary bounded | Adversarial bounded | Drift bounded | Adversarial unbounded (thesis model): median / ruin |
|---|---|---|---|---|
| Vol-HMM | 13,785 | 131.0 [124.4, 138.4] | 11.69 | 49.4 / 0.0170 |
| Risk-Constrained CPPI | 77.5 | 8.10 [7.71, 8.53] | 2.83 | 4.61 / 0.0010 |
| Thompson Sampling (now seeded) | 5,167 | 35.1 [31.6, 38.8] | 7.40 | 0.884 / 0.0730 |
| EXP3 | 54,888 | 15.0 [12.9, 17.3] | 14.11 | 0.000 / 0.3860 |
| Naive Bayes | 4,917 | 35.3 [32.1, 38.3] | 7.10 | 0.993 / 0.0600 |
| Softmax | 6,727 | 45.5 [40.5, 51.5] | 8.63 | 1.124 / 0.0630 |

* CPPI floor-breach rate in the Adversarial scenario: 0.6155 bounded, 0.9710 unbounded. In Stationary
  bounded it is 0.0000, and in Drift bounded 0.0005.
* Practical ruin (wealth ever at or below 10% of the start), Adversarial bounded: 0.0405 for EXP3 and 0
  for every other agent.
* The ranking in the Adversarial scenario is the same under both models: Vol-HMM is highest and CPPI is
  lowest among the ones that are not ruined. Under bounded returns the Beta-posterior agents no longer
  lose money in the crash (medians 35 to 46 instead of about 1).
* These are new seeded, 2,000-path versions of the thesis scenarios. They are not reruns of the
  committed 100-path tables (different seeds and return model), so they should not be read as
  corrections of specific thesis numbers.

## 4. Failed checks, surprises and deviations (as prominent as the findings)

1. **Version-1 negative-control expectation failed on exploratory data.** With the peak updated at
   every rebalance, the jump DGP still showed a large rebalancing effect (exploratory b1 = -0.42). This
   led to amendment 2, which changed the primary analysis after looking at exploratory data. Under
   version 1 the thesis agent would have been rated SUPPORTED. Under the amended primary it is PARTLY
   SUPPORTED with a negligible effect. Both are reported.
2. **The bounded model changes the economics** (amendment 3, `../bear_log_growth.json`). The thesis bear
   has expected log growth +0.0021 per step at f = 0.05 and 0.10, and -0.0204 at f = 0.5. Its Kelly
   fraction in [0, 1] is 0.072. The small shock is favourable at every f below 1 (Kelly fraction 0.5).
   The "bear" is therefore only bad for large bets, which is why detection has little wealth value under
   this model.
3. **Matched false-alarm rates were below target out of sample** (section 1.4).
4. **Downstream with a persistent shock: null** (section 1.5).
5. The 0.01 minimum leverage, an undocumented code choice, drives most of the thesis agent's breaches.
6. `summarize.py` was written, and a label-collision fix applied to `make_figures.py`, after the FREEZE
   commit. Both only read the confirmatory outputs. The exploratory `run_info.json` was overwritten by a
   part-b-only rerun (diagnostic fix, amendment 4). The exploratory `primary.json` is from the full run.

## 5. Limitations

* Synthetic only. The regimes, break time and shocks are stylised. The HMMs are given the true bull and
  thesis-bear emission parameters, which is a best case for them. BOCPD and the jump model are given no
  regime parameters, so they are disadvantaged by construction, and their specifications (prior, L = 10,
  window 60, features, penalty mapping) were fixed a priori and not tuned.
* The delay is censored at min(D, 50). Means depend on the horizon, while medians and shares do not.
* "Matched" false-alarm rates are approximate (section 1.4).
* The downstream sizing rule (base fraction 0 or 1) is one simple choice. Other sizing rules could rank
  the detectors differently.
* In outcome c the thesis agents observe simple returns, while their emission parameters were written
  for the log-return values, so the Vol-HMM is mildly misspecified under the bounded model.
* In outcome b, how a step's shock arrives inside the step (diffusive versus jump) is a modelling choice.
  The two DGPs bracket it but do not settle it.

## 6. Headline candidates (all NOT YET REPLICATED; resume use is Ansh's decision)

1. At matched false-alarm rates (5 per 1,000), a return-only HMM detects the simulated break as fast as or
   faster than the thesis Vol-HMM (+0.144 steps [0.088, 0.199]) and the Vol-HMM + CUSUM (+0.232 more).
   The volatility feature's value is fewer false alarms at the thesis threshold (2.14 versus 26.41 per
   1,000).
2. The thesis CPPI agent's floor breaches (60.7% of paths under bounded returns) are mostly caused by its
   0.01 minimum-leverage clip (88% of breaches), not gap risk. 50x rebalancing only reaches 56.3%, while a
   pure CPPI goes from 8.35% to 0.15%.
3. Under bounded returns thesis-definition ruin is zero for every agent by construction. The thesis's
   ruin rates are artifacts of simulated returns below -100%.

Null results with the same standing: detection has no median-wealth value for a persistent shock under
the bounded model (always in 120.3 versus 112.8 to 116.5), and the volatility feature is faster in only 21
of 81 matched cells.
