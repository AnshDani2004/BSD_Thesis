# Document 2: State, Replication, & LLM Consistency Guide

## 1. What We Have Done So Far
* **Base Architecture**: The overarching logic rests in `final_thesis_work/simulation`. We have successfully built the `MarketEnvironment`, `WealthTracker` (log-space), and an expansive suite of Bayesian/Bandit betting agents (`agents.py`).
* **The Proposed Methods**: We successfully coded the two core novel algorithms driving the thesis: `VolAugmentedHMM` (for enhanced regime detection) and `RiskConstrainedKelly` (for CPPI drawdown floors). These are fully integrated into the testing suites.
* **The Offline Matplotlib Engine**: Built `generate_all_figures.py` and `plot_engine.py` to aggressively run batches of simulations on stationary, drift, and adversarial shocks, outputting high-fidelity Kernel Density (CDF) and Monte Carlo Fan trajectory charts statically.
* **The Live Dashboard**: Transformed a simple HTML template into a fully React-Router integrated `Vite + React` application, styled as a dark-mode Bloomberg Terminal. The dashboard now features a comprehensive **Executive Summary Home Page** and a persistent **Global Simulation Context** ensuring data survival during navigation. The UI connects via REST API (`127.0.0.1:5001/api/simulate/compare`) to a Python Flask backend that calculates annualized **Sharpe** and **Sortino** Ratios live.
* **Empirical Historical Engine**: Integrated `yfinance` to allow deterministic replay of 4 critical S&P 500 epochs (2000, 2008, 2013, 2020), validating theoretical survival claims against real-world price fractures.
* **LaTeX Thesis Substructure**: The foundational `.tex` files have been finalized, integrating fee audits and risk-adjusted return equations directly into Chapter 3 (Methodology) and Chapter 4 (Experiments).

## 2. LLM Replication Guide & Strict Consistency Rules
If a new LLM agent picks up this project, they **must** abide by the following constraints to ensure math correctness and codebase consistency:

### 2.1 The Common Random Numbers (CRN) Methodology
Agents being compared **must** be evaluated against the exact same return matrices. Never call `simulate_batch()` independently for different agents. Call it once, hold the resulting 2-D matrix `(n_sims, t_steps)`, and feed the same paths to all agents in the loop. Failing to do this destroys the statistical integrity of the Monte Carlo tests by injecting idiosyncratic randomness into the comparison parameters.

### 2.2 Mathematical Boundaries & Log-Space Tracking
* **Log-Wealth Computations:** Never use basic accumulation loops `wealth *= (1 + f * r)`. Always utilize `np.log1p(f * r)`. Financial compounding over $T_{1000}$ iterations with standard floats will generate catastrophic underflow/overflow bounds inside Numpy arrays.
* **No-Lookahead Constraints:** The sequential evaluation must enforce filtering $\Filt_t$. The agent must call `act()` strictly before the return $r_t$ is observed. $r_t$ is only supplied to the agent afterward via `update(r_t)`. 
* **Ruin Traps:** If an agent breaches $1 + f_t r_t \leq 0$, the WealthTracker correctly snaps wealth to `-Infinity` and categorizes the Monte Carlo path as fundamentally ruined (Absorbing State). This must be strictly tracked for Fisher Exact failure probability evaluation.

### 2.3 Environmental Constraints
* **Numba Limitations:** The codebase utilizes `numba` `@njit` compilers for sampling loops. Python lists, dictionaries, SciPy functions, and Numpy random states **cannot** be generated inside `@njit` blocks. Pre-allocate random arrays via NumPy prior to passing them to the Numba functions.
* **Cross-Language Integration:** The React Vite frontend requires Node.js (via NVM). Do NOT rely on MacOS system packages natively since they throw permission/version errors on this machine. The backend requires Python mapped to Port `5001` configured to accept CORS from `localhost`.
* **LaTeX Custom Macros:** Never hardcode base math definitions. Utilize thesis macros (e.g. `\E` for Expected Value, `\Filt` for Filtration) stored explicitly in `main.tex`. 

### 2.4 Empirical Mode Replication Rules
* **Single Path Policy:** When a scenario key matches the historical registry, `n_sims` must be forced to 1. 
* **Calendar Mapping:** Use the `date_start` metadata to generate ISO strings for the time-series.
* **Deterministic Replay:** Ensure agent `update()` and `act()` loops remain functionally identical to synthetic mode to maintain the "one framework" proof.
* **Global State Persistence:** All simulation results (Wealth Paths, Console Logs, Charts) MUST be stored in the `SimulationContext.jsx` provider. Standard React component states will be wiped during tab switching (Simulator -> Academic Logs), breaking the thesis demonstration flow.
* **Risk Metric Annualization:** When calculating Sharpe/Sortino in Python, assume $N=252$ trading days. Use running variance statistics ($E[x^2] - (E[x])^2$) to calculate volatility efficiently within the O(T) tracker loop.
* **Transaction Fee Audit:** All wealth updates MUST subtract $\Delta f_t \cdot c$, where $\Delta f_t$ is the absolute change in exposure and $c$ is the friction in basis points. This is critical for preventing "Zero-Cost Fantasy" results.
