# CPPI agent specification (Extension 2)

Written 2026-10-12 from the code, not from the thesis text. Sources: `simulation/risk_constrained.py`
(class `RiskConstrainedKelly`), `simulation/wealth_tracker.py` (class `WealthTracker`), and the call
site in `experiments/generate_all_figures.py` on branch `audit-cppi-floor` (commit dcafbf1), which
constructs `RiskConstrainedKelly(1, max_drawdown=0.20, cppi_multiplier=3.0)` and passes the wealth in.
This document is what an independent replication should implement. Where the thesis text
(`thesis/04_methodology.tex`, Definition 4.x and Proposition `cppi_bound`) says something different,
the code wins here and the difference is listed in section 9.

## 1. Inputs and state

One risky asset ("one arm", odds b = 1). Per path the agent holds:

| symbol | meaning | initial value |
|---|---|---|
| a, b | Beta posterior counts on "the step return is positive" | a = 1, b = 1 |
| W | wealth as last reported to the agent | 1.0 |
| P | running peak of reported wealth | 1.0 |

Fixed parameters (thesis run): drawdown limit D = 0.20, CPPI multiplier m = 3.0, minimum leverage
L_min = 0.01, maximum leverage L_max = 0.50, transaction cost 0.

## 2. Which Kelly fraction is scaled

The base fraction is the even-money (binary) Kelly fraction of the Beta posterior mean, NOT the Gaussian
fraction mu / sigma^2:

    p_hat  = a / (a + b)
    f_base = clip(p_hat - (1 - p_hat) / 1, 0, 1) = clip(2 p_hat - 1, 0, 1)

(`_compute_kelly` with odds 1; with one arm the "normalise so the sum is at most 1" step never binds.)
At t = 0, p_hat = 0.5 so f_base = 0 and the first bet is 0.

Posterior update after each step's return r: if r > 0 then a += 1, otherwise b += 1. A return of exactly
0 counts as a loss here (unlike `NaiveBayesKelly`, which ignores r = 0). The agent never uses the size of
r, only its sign.

## 3. Floor, cushion and leverage multiplier

Computed fresh at every call to `act()`, from the W and P the agent currently holds:

    F      = (1 - D) * P            (floor = 80% of the running peak)
    C      = W - F                  (cushion)
    if C <= 0:      lambda = L_min  (= 0.01; "emergency delever", still a nonzero bet)
    else:           lambda = clip(m * C / W, L_min, L_max)   (= clip(3 C / W, 0.01, 0.50))

At the peak, C / W = 0.20 and m C / W = 0.60, so lambda sits at the 0.50 cap; the cap stops binding once
W < 0.96 P (C / W < 1/6).

## 4. The bet

    f = lambda * f_base     (then divided by its sum if the sum exceeded 1, which cannot happen here)

So f is in [0, 0.5]. The agent never borrows. The exposure as a multiple of the cushion is
f / (C / W) = (lambda / (C / W)) * f_base, which equals m * f_base <= m whenever lambda is not clipped,
is below that when the 0.50 cap binds, and can exceed m when the 0.01 floor on lambda binds
(C / W < 0.01 / m) or when C <= 0.

## 5. Rebalancing and timing (one rebalance per step)

For each step t = 0, ..., T-1 (Python indexing; `returns[i, t]` is the return realised over step t):

1. `f_t = agent.act()[0]` using (a, b, W, P) as they stand after step t-1.
2. `tracker.update(f_t, r_t)`: wealth multiplier M_t = 1 + f_t r_t (friction 0).
3. `agent.update(np.array([r_t]), wealth=tracker.wealth)`: Beta update on sign(r_t); then W := reported
   wealth; P := max(P, W).

The bet is fixed for the whole step: the position f_t W_t is set at the start of the step and the step's
simple return r_t is applied to it once (equivalently, buy and hold within the step). There is no
intra-step monitoring or rebalancing.

## 6. Wealth update and ruin (`WealthTracker`)

    if M_t = 1 + f_t r_t <= 0:  ruined (absorbing); wealth reported as 0 from then on
    else:                       log W_{t+1} = log W_t + log(M_t);  W_{t+1} = exp(log W_{t+1})

Ruin is only this one-step wipe-out. It is not a drawdown or floor event. After ruin the tracker ignores
further updates; the agent keeps receiving wealth 0 (so C < 0 and lambda = 0.01, irrelevant).
Because f <= 0.5, ruin requires r_t <= -2, which only the unbounded (thesis) return model can produce.
Under the bounded model of Extension 2 (r = exp(x) - 1 > -1), ruin is impossible for any bet in [0, 1].

## 7. Below the floor

Nothing special happens beyond lambda = 0.01: the agent keeps a small long position (0.01 * f_base of
wealth), keeps learning, and its floor stays at 80% of the old peak. It can recover above the floor if
the asset rises. There is no lock-in to cash.

## 8. Drawdown and floor breach (audit / Extension 2 outcome, not part of the agent)

Measured on the step-end wealth grid W_0 = 1, W_1, ..., W_T: running peak P*_t = max_{s <= t} W_s,
drawdown DD_t = (P*_t - W_t) / P*_t; a path "breaches the floor" if max_t DD_t > 0.20. With one rebalance
per step, P*_t equals the agent's P, so a breach is exactly W_t < F_t for some t.

## 9. Differences from the thesis text

| item | thesis text | code (this spec) |
|---|---|---|
| base Kelly fraction | Gaussian Kelly mu / sigma^2 (03_foundations eq. gaussian_kelly) implied | Beta-posterior binary Kelly clip(2 p_hat - 1, 0, 1) |
| leverage rule | lambda = min(1, m C / W), lambda in [0, 1] | lambda = clip(m C / W, 0.01, 0.50); 0.01 when C <= 0 |
| below floor | not stated | lambda = 0.01, position kept |
| drawdown guarantee | Prop. cppi_bound: no breach if m |r_min| <= 1 | holds only if lambda is never floored at 0.01 and r > -1/(m f_base) |

## 10. Extension 2 variants (used in outcome b only, labelled)

* `V1_code`: exactly sections 1 to 7 (L_min = 0.01, L_max = 0.50), with n rebalances per step (section 11).
* `V2_pure`: identical except L_min = 0 (so lambda = 0 when C <= 0 and lambda = clip(m C / W, 0, 0.50)).
  This is the textbook CPPI rule; it isolates gap risk from the 0.01 minimum-leverage clip.

With n = 1, `V1_code` must reproduce `RiskConstrainedKelly` + `WealthTracker` on the same returns to
floating-point precision (checked by `tests/test_fidelity.py`).

## 11. n rebalances per step (outcome b)

Each step is split into K = 50 equal sub-intervals (fine grid). With n in {1, 2, 5, 10, 50}, the agent
rebalances at fine indices 0, K/n, 2K/n, ... of each step. At a rebalance it recomputes lambda from its
current W and P (sections 3 and 4) using the f_base fixed at the start of the step; between rebalances the
position is held (buy and hold): W_end = W_start (1 + f (exp(sum of fine log increments) - 1)), computed in
log space as in section 6. The Beta posterior is updated once per step, at the step end, on the sign of the
step's total log return (the same sign as its simple return).

Peak tracking (PREREG amendment 2):

* `peak_step` (primary): the agent receives its wealth after every rebalance interval (so W is current
  at each rebalance) but updates its peak P := max(P, W) only at step ends. The floor is then tracked on
  the same grid for every n, and n changes only how often the exposure is reset.
* `peak_every` (secondary; the version first written in this spec): P := max(P, W) after every rebalance
  interval. With n > 1 the agent then also sees intra-step highs, so its floor sits higher, which by
  itself lowers step-grid breaches.

With n = 1 the two are identical and equal sections 5 to 7.
