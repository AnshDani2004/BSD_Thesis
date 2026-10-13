# Bayesian Sequential Decision-Making under Regime Switches

**Honors Thesis, School of Mathematics and Statistical Sciences, Arizona State University**

Studied Bayesian sequential decision-making under regime switches: an HMM regime detector with a
CUSUM change-point trigger feeding Kelly position sizing, evaluated by Monte Carlo on detection
delay, ruin and median wealth. This repository contains the code, LaTeX source and slides. All
results are from simulation. A React dashboard replays the experiments.

![Dashboard Preview](final_thesis_work/dashboard/preview.png)

## What the thesis studies

- Why a Bayesian bettor adapts slowly after a regime shift (Thompson Sampling, Naive Bayes and other
  baselines in a simulated bull-to-bear break).
- A volatility-augmented hidden Markov model (Vol-HMM) with a CUSUM change-point trigger as the regime
  detector, and Kelly position sizing driven by it.
- Kelly sizing under a CPPI (constant proportion portfolio insurance) drawdown floor.
- Ruin under heavy tails when the Kelly fraction is computed under a Gaussian assumption.

Each method is evaluated by Monte Carlo on detection delay, ruin and median terminal wealth. The
thesis's own conclusion is in `final_thesis_work/thesis/06_conclusion.tex`.

**Status of the numbers (2026-10-12).** An audit found that the experiment script did not pass the
current wealth to the CPPI agent, so its floor never moved. The fix and a full rerun are on the local
branch `audit-cppi-floor` (see `final_thesis_work/audit/RERUN_cppi_floor.md` there). Until that rerun
is independently replicated, this README quotes no results; the thesis PDF reports the results as
defended.

## Reproduce

From `final_thesis_work/` with Python 3.10 and `requirements.txt` (rerun on 2026-10-12 with
numpy 2.2.6, scipy 1.15.3, numba 0.63.1):

```bash
python experiments/exp06_heavy_tails_fatality.py   # Kelly ruin under heavy tails, about 3 s
python experiments/generate_all_figures.py         # stationary / adversarial / drift tables and figures, about 40 s
python audit/audit_checks.py                       # audit replay of the adversarial scenario, about 3 min
```

Tables are written to `dashboard/frontend/public/results/tables/*_stats.json`. Thompson Sampling and
the Drift scenario use the unseeded global NumPy random generator, so those numbers change from run
to run; the other agents in the Stationary and Adversarial scenarios reproduce to floating-point
precision.

## Limitations

- **Simulation only.** Every result comes from synthetic paths. The S&P 500 epochs are single paths,
  and the historical table in the thesis has no saved output here. Their return arrays are not stored in
  this repository: `python3 scripts/fetch_historical_data.py` (from `final_thesis_work/`, needs yfinance)
  downloads them and checks each against `data/historical/checksums.json`.
- **One stylised break.** A Gaussian bull regime followed by a Student-t(3) bear regime halfway
  through each path. Per-step moves are far larger than daily equity returns, and the two regimes are
  very different, which makes the break easy to detect.
- **The filter knows the true regimes.** The Vol-HMM emission parameters equal the parameters that
  generate the data (they are not estimated), so detection delay is a best case.
- **Small samples.** The adversarial and stationary runs use 100 paths per agent; the heavy-tail
  experiment uses 1,000.
- **Parameter choices.** Persistence, CUSUM threshold and drift, CPPI multiplier and floor were chosen
  by the author; the persistence and CUSUM sensitivity tables in the thesis have no generating script
  in this repository.
- **"Ruin" is a one-step wipeout** (1 + f r <= 0 in `simulation/wealth_tracker.py`), not a breach of
  the drawdown floor.
- **CPPI floor wiring.** On this branch, `generate_all_figures.py` and the dashboard backend call the
  CPPI agent's `update` without the current wealth, so its floor never moves and its leverage
  multiplier stays at its cap. Fixed on `audit-cppi-floor`; a one-step-rebalanced floor can still be
  gapped through by heavy-tailed shocks.
- **Kelly fraction in Experiment 6** is the even-money fraction 2P(r > 0) - 1, not mu/sigma^2 as the
  thesis text describes.
- **Single asset, proportional fees only**, as the thesis notes in its own Limitations section.

## Project architecture

- **Simulation core (Python):** NumPy, SciPy, Numba. `simulation/hmm_refined.py` (Vol-HMM forward
  filter and CUSUM), `simulation/risk_constrained.py` (CPPI-scaled Kelly), `simulation/agents.py`
  (Thompson Sampling, Naive Bayes, EXP3, UCB, Softmax), `simulation/wealth_tracker.py` (log-space wealth).
- **Dashboard:** React/Vite frontend (`dashboard/frontend/`) and a Flask backend (`dashboard/backend/app.py`).

## Repository structure

```text
final_thesis_work/
  simulation/      Vol-HMM, CPPI agent, baseline agents, environments, wealth tracker
  experiments/     experiment scripts (exp01 to exp07) and generate_all_figures.py
  audit/           audit_checks.py and its outputs (added 2026-10-12)
  data/historical/ checksums of the S&P 500 return arrays (fetch with scripts/fetch_historical_data.py)
  dashboard/       React/Vite frontend and Flask backend
  thesis/          LaTeX source and compiled main.pdf
  presentation/    Beamer slides for the defense
```

## Run the dashboard locally

```bash
cd final_thesis_work/dashboard/backend && python3 app.py        # API on http://localhost:5001
cd final_thesis_work/dashboard/frontend && npm install && npm run dev   # UI on http://localhost:5173
```

---

**Author**: Ansh Dani  
**Thesis Committee**: [Dr. Shiwei Lan](https://math.la.asu.edu/~slan/) (Thesis Director) and [Dr. Nicolas Lanchier](https://math.la.asu.edu/~lanchier/) (Second Reader)  
**Institution**: [School of Mathematics and Statistical Sciences](https://math.asu.edu/) (Arizona State University)
