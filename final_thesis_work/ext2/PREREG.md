# PREREG: Extension 2 of the honors thesis (detector ablation, CPPI gap risk, bounded returns)

Status: frozen (amendments 1 to 5 below). Version 1 was written and committed (a5ebfc0) before any
exploratory or confirmatory computation of the outcomes below (only fidelity smoke checks on test seeds 900000 to 900999 had been run). It is frozen by
the commit whose message starts with `FREEZE ext2`; the confirmatory run happens once, after that commit,
on the confirmatory seeds only. Changes after version 1 are listed in the amendment log (section 11).

Everything is synthetic. No market data is used. The CPPI agent is specified in `CPPI_SPEC.md` (same
directory); this document and `CPPI_SPEC.md` together are meant to be enough to re-implement every
number without reading the code.

## 1. Questions

a. Detection frontier. At matched false-alarm rates, how fast do these online detectors flag a simulated
   regime break: a return-only HMM, the thesis Vol-HMM (rolling-volatility feature), the Vol-HMM with its
   CUSUM trigger, Bayesian online changepoint detection (BOCPD) and an online statistical jump model?
   And what do their alarms do to ruin and median wealth when they drive Kelly sizing under the CPPI
   floor? (The audit found that at the thesis threshold a return-only HMM detects the thesis break as
   fast as the Vol-HMM; this asks whether that survives matching false-alarm rates.)
b. Gap risk. Under bounded returns, how often does the CPPI agent fall more than 20% below its running
   peak, and does that rate fall toward zero as it rebalances more often and as the multiplier m falls?
c. Thesis agents. Ruin and median terminal wealth of the six thesis agents under bounded returns with
   every random generator seeded.

## 2. Data-generating processes

Time steps t = 0, ..., T-1 with T = 250. Break time tau = 125 (the thesis's t_steps // 2).

Return model (primary, "bounded"). Each step draws a LOG return x_t; the simple return applied to wealth
is r_t = exp(x_t) - 1, which is always above -1. Regimes, as distributions of x_t:

* bull: x = 0.08 + 0.15 Z, Z ~ N(0, 1).
* shock(mu, s): x = mu + s Z sqrt(3 / V), Z ~ N(0, 1), V ~ chi-square(3) independent (Student-t with
  3 degrees of freedom, location mu, scale s). The thesis bear is shock(-0.10, 0.30).

Comparison only ("unbounded", the thesis model): the same draw x is used directly as the simple return,
r_t = x_t, so r_t <= -1 can occur. It is always labelled as such and never used for a headline.

Scenarios:

* a (detection): bull for t < 125; shock(mu, s) for t in [125, 125 + D); bull again for t >= 125 + D.
  Shock grid (mu, s): small (0.00, 0.20), thesis (-0.10, 0.30), large (-0.20, 0.45). Durations
  D in {5, 20, 125} (D = 125 means the shock lasts to the end, the thesis setup). 9 shock cells.
* b (gap risk): bull for t < 125, thesis bear for t >= 125, generated on a fine grid of K = 50 substeps
  per step. The step's log return is exactly as above; the substeps split it as follows.
  Fine normals e_{t,k} ~ N(0, 1) iid, Z_t = sum_k e_{t,k} / sqrt(50) (so Z_t ~ N(0, 1)); V_t ~ chi2(3).
  Primary, "diffusive": y_{t,k} = mu_t / 50 + s_t c_t e_{t,k} / sqrt(50) with (mu_t, s_t, c_t) =
  (0.08, 0.15, 1) in the bull and (-0.10, 0.30, sqrt(3 / V_t)) in the bear. Given V_t this is a Gaussian
  random walk inside the step and sum_k y_{t,k} = x_t exactly, so the per-step distribution is held
  exactly fixed (Student-t is a normal variance mixture). Sensitivity, "jump": bull steps as above; in a
  bear step the drift -0.10 / 50 is spread over the substeps and the whole shock 0.30 sqrt(3 / V_t) Z_t
  arrives at one substep drawn uniformly from 0..49. Again the step sums are exactly x_t.
  Placebo: the same fine grid with bull at all t (no break).
* c (thesis agents): Stationary (bull at all t); Adversarial (bull for t < 125, thesis bear after);
  Drift (x_t = mu_t + 0.15 Z_t, Gaussian, mu_t = linspace(0.08, -0.05, 250)). Same as
  `experiments/generate_all_figures.py` except that x is a log return and the seeds are per path.

## 3. Seeds, random streams, exploratory and confirmatory samples

The unit of inference is one simulated path, identified by an integer seed. Every random draw of path s
comes from `numpy.random.default_rng(numpy.random.SeedSequence([s, stream]))` (NumPy 2.2 PCG64):

| stream | draws (in this order) | used by |
|---|---|---|
| 1 | Z = standard_normal(250), then V = chisquare(3, 250) | a and c |
| 2 | e = standard_normal((250, 50)), then V = chisquare(3, 250) | b |
| 3 | integers(0, 50, size=250) (the jump substep of each step) | b, jump DGP |
| 4 | Thompson Sampling's Beta posterior draws (`rng.beta` once per step) | c |

All shock cells and all detectors use the same paths for a given seed (common random numbers), so
contrasts are paired by seed. All resampling (bootstrap, sign flips) uses Generators seeded with
20261012.

* Exploratory seeds: 0 to 999 (1,000 paths). Used for threshold tuning, development and every
  refinement. Thresholds tuned here are frozen and reused, never re-tuned on confirmatory seeds.
* Confirmatory seeds: 100000 to 101999 (2,000 paths per cell). Used once, after the FREEZE commit.
  The code refuses confirmatory seeds unless run with `--phase confirmatory`, and the confirmatory phase
  refuses to run if the ext2 directory has uncommitted changes.
* Test seeds for unit tests: 900000 and up, plus 0 to 999 where a test needs exploratory draws.

## 4. Cells

Outcome a: detector (6) x A_ii (3) x false-alarm target (3) x shock cell (9), 2,000 paths each.
Outcome b: DGP (diffusive, jump, placebo) x agent variant (2) x n (5) x m (4), 2,000 paths each.
Outcome c: scenario (3) x return model (2) x agent (6), 2,000 paths each.

## 5. Detectors (outcome a)

All detectors read the log returns x_0..x_t and output a score s_t after observing x_t (larger means more
evidence of the bad regime or of a change). No detector reads x_{t+1} or later. An alarm at t is
s_t > h. Emission parameters are the generating bull and thesis-bear values (as in the thesis); the
detectors are NOT told the small or large shock parameters.

* `hmm_ret` (return-only HMM): two states (bull, bear), transition matrix with A_ii on the diagonal and
  1 - A_ii off it, start (0.5, 0.5), Gaussian emissions N(x; 0.08, 0.15) and N(x; -0.10, 0.30).
  Forward filter in log space, in the order of `simulation/hmm_refined.py`:
  l_j <- e_j(x_t) + logsumexp_i(l_i + log A_ij), then normalise. s_t = log P(bear) - log P(bull).
* `volhmm` (thesis Vol-HMM, no CUSUM): as `hmm_ret` but each emission also multiplies a Gamma density of
  v_t = population standard deviation of x over the last min(5, t+1) steps plus 1e-6 (v_t = 0.05 when
  fewer than 2 observations): Gamma(v; shape 2, scale 0.075) for bull, Gamma(v; shape 4, scale 0.15)
  for bear (the values of `create_vol_augmented_hmm_kelly(1, 0.08, 0.15, -0.10, 0.30)`).
* `volhmm_cusum` (thesis Vol-HMM with CUSUM, exactly `VolAugmentedHMM` with cusum_threshold 3 and drift
  0.5): S_0 = 0, S_t = max(0, S_{t-1} - (x_t - 0.08)/0.15 - 0.5); once S_t > 3 the trigger latches for the
  rest of the path and +2 is added to the bear log-emission from that step on.
* `bocpd_rl` (Adams and MacKay 2007): constant hazard H = 1 - A_ii. Within a segment x is iid Normal with
  unknown mean and precision, Normal-Gamma prior mu0 = 0, kappa0 = 1, alpha0 = 2, beta0 = 0.04; the
  predictive is Student-t with 2 alpha degrees of freedom, location mu_n and squared scale
  beta_n (kappa_n + 1) / (alpha_n kappa_n). Run length l after observing x_t = number of observations of
  the current segment including x_t. Recursion: P(l = 1) is proportional to H pred_prior(x_t) times the
  total previous mass, plus (at t = 0 only) the growth term from l = 0; P(l + 1) is proportional to
  P_prev(l) (1 - H) pred_l(x_t); normalise. Updates: kappa' = kappa + 1, mu' = (kappa mu + x)/(kappa + 1),
  alpha' = alpha + 1/2, beta' = beta + kappa (x - mu)^2 / (2 (kappa + 1)). No truncation.
  s_t = log P(l <= 10) - log P(l > 10) ("a change within the last 10 observations"); s_t = +1e9 when
  P(l > 10) = 0 (only possible for t <= 9, inside the burn-in).
* `bocpd_mean`: the same filter; s_t = - sum_l P(l) mu_l (minus the posterior mean of the current
  segment's mean). This is a state-like score (stays high while the segment mean is believed negative).
* `jm` (online statistical jump model; Bemporad et al. 2018, Nystrup et al. 2020): at each t, take the
  window of the last min(60, t + 1) steps (skip, score 0, if fewer than 3). Features per step:
  (x_t, population std of x over the last min(5, t + 1) steps). Z-score each feature within the window
  (population std; 1 if zero). K = 2 states, loss = squared Euclidean distance to the state centroid,
  jump penalty lambda = 2 log(A_ii / (1 - A_ii)) per state change (the Viterbi penalty of an HMM with
  unit-variance Gaussian emissions: 5.89, 6.95, 9.19 for A_ii = 0.95, 0.97, 0.99). Fit: initial states by
  a median split of the return feature (at or below the median = state 0), centroids = state means; then
  repeat up to 10 times: dynamic programme V_j(k) = loss_j(k) + min(V_{j-1}(k), V_{j-1}(1-k) + lambda)
  (ties keep the state), backtrack from the cheaper final state; stop if the states did not change
  (from the second pass on); else update centroids (keep a centroid if its state is empty). If the 10th
  pass ends with a centroid update, the DP costs are recomputed with the final centroids. The bear state
  is the one with the lower return-feature centroid (state 0 on ties). s_t = V_end(bull) - V_end(bear).

A_ii in {0.95, 0.97, 0.99} for every detector: the HMM diagonal, the BOCPD hazard 1 - A_ii, and the jump
penalty above.

Burn-in: steps 0 to 9 are excluded from false-alarm counting, from detection and from sizing (alarm
forced off).

False-alarm rate: the share of pre-break steps t in [10, 125) with s_t > h, per 1,000 such steps, pooled
over paths. Targets: 1, 5 and 10 per 1,000. Secondary: alarm onsets (s_t > h and s_{t-1} <= h) per 1,000
pre-break steps.

Threshold tuning (exploratory seeds only): pool the pre-break scores of the 1,000 exploratory paths
(115,000 values per detector and A_ii; pre-break scores do not depend on the shock cell, which is checked),
sort them in descending order and set h to the (k+1)-th largest with k = floor(target / 1000 x 115,000).
Then at most k pooled scores exceed h. The thesis-native threshold h = 0 (P(bear) > 0.5 for the HMMs) is
reported as a fourth, unmatched operating point.

Detection delay: first t >= 125 with s_t > h, minus 125 (delay 0 = alarm right after the first shock
return), censored at horizon H = min(D, 50): a path with no alarm in [125, 125 + H) gets delay H and counts
as "never detected". Reported per cell: mean and median of the censored delay, its 90th percentile, share
never detected, and the realised false-alarm rate on the same paths.

Downstream sizing (Kelly under the CPPI floor wired in; amendment 1 adds the unbounded comparison): base fraction b_t = 0 if the alarm was on at
t - 1, else 1 (b_0 = 1; the Gaussian bull Kelly fraction 0.08 / 0.15^2 = 3.56 clipped to 1, and the bear
fraction clipped to 0). Bet f_t = lambda_t b_t with lambda_t the thesis CPPI multiplier of CPPI_SPEC.md
section 3 (m = 3, floor 80% of the running peak, lambda in [0.01, 0.50]); one rebalance per step; wealth
W_{t+1} = W_t (1 + f_t r_t) with r_t = exp(x_t) - 1. Benchmarks on the same paths: "always in" (b_t = 1)
and "oracle timing" (b_t = 0 exactly for t in [125, 125 + D)). Secondary: "no floor" (f_t = 0.5 b_t), and
(amendment 1) the same alarms and bets applied to the unbounded model's simple returns r_t = x_t (thesis
comparison; the detectors see the same x either way, so only the wealth changes). Context (amendment 3):
`bear_log_growth.py` computes E[log(1 + f r)] for constant bets in each regime by quadrature.

## 6. Outcomes and tests

### 6a. Detection frontier (outcome a)

Primary contrasts, at the operating point A_ii = 0.95 (the thesis value), target 5 per 1,000, thesis
shock, D = 125, H = 50; statistic = mean over paths of the paired difference of censored delays;
two-sided sign-flip randomisation test (10,000 random sign vectors), 95% path-bootstrap CI (2,000):

* A1: volhmm - hmm_ret (does the volatility feature speed detection at matched false alarms?)
* A2: volhmm_cusum - volhmm (does the CUSUM trigger?)
* A3: bocpd_rl - volhmm_cusum
* A4: jm - volhmm_cusum

Secondary: A1 to A4 and A5 (bocpd_mean - volhmm_cusum) at every other cell of A_ii x target x shock x D;
the native-threshold delays and false-alarm rates; onsets per 1,000; and the downstream outcomes below.
Downstream outcomes per cell (descriptive, no tests): thesis ruin rate (1 + f r <= 0; zero by
construction under bounded returns since f <= 0.5 and r > -1), practical-ruin rate (wealth ever at or
below 0.10 on the step grid), median terminal wealth (95% bootstrap CI), floor-breach rate (CPPI_SPEC.md
section 8; Wilson 95% CI).

### 6b. Gap risk (outcome b)

Agent variants (CPPI_SPEC.md sections 10 and 11): V1_code (the thesis agent, lambda in [0.01, 0.50]) and
V2_pure (lambda in [0, 0.50]). Grid: n rebalances per step in {1, 2, 5, 10, 50}; m in {1, 2, 3, 5}.
Peak tracking (amendment 2): primary `peak_step` (peak updated at step ends only, exposure rebalanced n
times); secondary `peak_every` (peak updated at every rebalance). Diagnostic (amendment 2): for each
breaching path, whether lambda sat at its minimum (or the cushion was <= 0) at the start of the rebalance
interval in which wealth first ended below 80% of the step-grid peak, within the first breaching step.
Breach (primary outcome): wealth on the step-end grid falls more than 20% below its running peak
(CPPI_SPEC.md section 8). Secondary: breach on the fine grid (all 50 substeps per step, against the
fine-grid running peak), median terminal wealth, maximum drawdown. Comparison: V1_code at n = 1 for each m
on the unbounded model (r = x) with the same stream-2 draws.

Test, per (DGP, variant, peak tracking): grouped binomial logistic regression of breach on log(n) and m over the 20
(n, m) cells on common paths, logit P = b0 + b1 log n + b2 m (Newton-Raphson, ridge 1e-8). CIs and
one-sided p-values from a path bootstrap (2,000 resamples of the 2,000 paths, each path's 20 cells kept
together): p(b1 < 0) = (1 + #{boot b1 >= 0}) / 2001, p(b2 > 0) = (1 + #{boot b2 <= 0}) / 2001.

The pre-registered gap-risk prediction: if the remaining breaches are gap risk, breach falls toward zero
as n grows and as m falls (with one rebalance per interval, a breach needs a one-interval loss larger than
1/m of the exposure; for V2_pure with m <= 1 and r > -1 a step-grid breach is impossible, a theorem used as
a check). Decision rule per variant, under the diffusive DGP with `peak_step` tracking:

* SUPPORTED if b1 < 0 and b2 > 0, both with Holm-adjusted p < 0.05 (within the primary family below), and
  the breach rate at n = 50 is at most 10% of the rate at n = 1 for m = 3 ("toward zero").
* PARTLY SUPPORTED if both slopes pass but the toward-zero criterion fails.
* NOT SUPPORTED otherwise.

Under the jump DGP the prediction is reversed for n (the shock arrives inside one interval whatever n is),
so the jump runs are a negative control: with `peak_step` tracking, b1 near zero is expected (95% CI
covering 0 or |b1| < 0.1); reported, not tested for support. The `peak_every` regressions are reported
as secondary.

### 6c. Thesis agents (outcome c)

Per scenario x return model x agent: thesis ruin rate (Wilson 95% CI), practical-ruin rate, median
terminal wealth (bootstrap 95% CI), and for the CPPI agent the floor-breach rate. Descriptive; no tests.
Pre-stated expectation (follows from the code): every thesis agent bets a fraction in [0, 1], so under
bounded returns thesis ruin is exactly 0 for every agent; any nonzero value is a bug.

## 7. Multiple testing

Primary family (Holm, alpha 0.05, 8 tests): A1 to A4 (two-sided) and, under the diffusive DGP with
`peak_step` tracking, b1 < 0 and b2 > 0 for V1_code and V2_pure (one-sided). Secondary contrasts (A1 to A5 at all other cells, jump-DGP
slopes): Benjamini-Hochberg, reported as q-values. Everything else is descriptive.

## 8. Kill criteria

None. This is an ablation and robustness study; every result is reported whatever its sign. A null or
contradicting result is brought to Ansh with two options before anything is claimed.

## 9. Known-answer tests, placebos and sanity checks (must pass before the confirmatory run)

In `tests/` (pytest): each detector raises an alarm within 2 steps of a huge break (x = -2 per step) and
raises few alarms on a constant bull regime at its tuned threshold; the BOCPD run-length posterior puts
more than 0.9 of its mass on run lengths <= 5 within 3 steps of a large mean shift, and on the full run
length before it; the jump model recovers two known segments offline (>= 95% of labels); thresholds tuned
on exploratory seeds 0 to 499 give rates within a pre-set tolerance on seeds 500 to 999; with 1,000
rebalances per step and Gaussian returns the CPPI floor is (almost) never breached; with m = 0 (V2_pure)
the floor is never breached; V2_pure with m = 1 never breaches on bounded returns; seeded runs are
bit-identical across two separate executions; the re-implemented HMMs and CPPI agent match the thesis
classes on the same returns; no detector looks ahead; bounded simple returns are above -1; the fine-grid
increments sum to the step log returns.

Placebos reported with the results: (i) fake break: on paths with no shock, the share of paths with an
alarm in [125, 175) (should be close to what false alarms alone give); (ii) the no-break fine-grid placebo
for the CPPI; (iii) realised confirmatory false-alarm rates versus the targets.

## 10. Outputs

`results/exploratory/` and `results/confirmatory/`: `a_frontier.csv`, `a_downstream.csv`,
`a_contrasts.csv`, `b_gap.csv`, `b_regression.json`, `c_agents.csv`, `primary.json`, `thresholds.json`
(exploratory only), `run_info.json` (git hash, versions, seeds, runtime); `results/bear_log_growth.json`. Figures: `fig_frontier.png`
(delay vs realised false-alarm rate, one panel per A_ii, thesis shock D = 125), `fig_frontier_shocks.png`
(one panel per shock cell at A_ii = 0.95), `fig_gap.png`.

## 11. Amendment log

| # | date | change | reason | seen data? |
|---|---|---|---|---|
| 0 | 2026-10-12 | version 1 (commit a5ebfc0) | | no outcome data |
| 1 | 2026-10-12 | Outcome a downstream: add the same alarms and bets on the unbounded model (r = x) as a labelled comparison. | Exploratory run: under the bounded model "always in" had the highest median terminal wealth (120.9) and the oracle-timing benchmark did not beat it (119.0), so detection has almost no wealth value there; the comparison shows whether that is a property of the bounded model. | yes, exploratory seeds 0 to 999 |
| 2 | 2026-10-12 | Outcome b: primary breach analysis uses `peak_step` tracking (peak updated at step ends only; exposure rebalanced n times); the originally specified `peak_every` version becomes secondary. Add the lambda-at-minimum breach diagnostic. | Exploratory run: with `peak_every`, the jump-DGP negative control still showed a strong n effect (b1 = -0.42, V1 breach at m = 3 from 0.624 at n = 1 to 0.234 at n = 50) because the agent sees intra-step highs and raises its floor. A quick exploratory diagnostic with `peak_step` gave V1 0.624 to 0.586 (diffusive) and V2 0.088 to 0.007 (diffusive), 0.088 to 0.085 (jump). So `peak_every` mixes "rebalance exposure more often" with "track a higher floor"; only `peak_step` isolates the first, which is what the gap-risk prediction is about. Under version 1 the V1 decision was SUPPORTED on exploratory seeds; this change is expected to make it less favourable. | yes, exploratory seeds 0 to 999 |
| 3 | 2026-10-12 | Add `bear_log_growth.py` (deterministic quadrature of E[log(1 + f r)] per regime) as context. | Exploratory finding: under the bounded log-Student-t model every regime has positive expected log growth for small enough bets (E[exp(x)] is infinite for Student-t x), e.g. the thesis bear has Kelly fraction 0.072 in [0, 1] with growth +0.0022 per step, so "go to cash in the bear" is never Kelly-optimal under the bounded model. | analytic, no seeds |
| 4 | 2026-10-12 | The lambda-at-minimum diagnostic records the interval in which wealth first ended below 80% of the step-grid peak (not any interval of the breaching step). | Exploratory run 2 showed V2_pure (minimum 0) flagged "at minimum" on all n = 50 breaches, an artifact of intervals after the crossing (cushion already <= 0). Diagnostic only; breach outcomes unchanged. | yes, exploratory seeds 0 to 999 |
| 5 | 2026-10-12 | FREEZE. Thresholds in `results/exploratory/thresholds.json` (tuned on exploratory seeds 0 to 999, identical in exploratory runs 1 and 2) are final. Confirmatory run on seeds 100000 to 101999, once. | | |
