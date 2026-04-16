# Document 1: Project Aims & Components

## 1. Ultimate Objective
The objective of this project is to produce a rigorous, Barrett Honors-level computational mathematics thesis titled **"Bayesian Sequential Decision-Making in Non-Stationary, Heavy-Tailed Environments."** The work bridges theoretical probability, algorithmic optimization, and empirical software engineering. The final deliverables include a formally compiled LaTeX thesis, a journal-ready SIURO paper draft, and a live, interactive Bloomberg-style web dashboard to be used during the thesis defense presentation.

## 2. Core Methodological Claims (The Thesis)
The project aims to mathematically prove and empirically demonstrate three interrelated concepts:
1. **The Vulnerability of Standard Models:** Classical Kelly Criterion and Naive Bayesian updates fail catastrophically in heavy-tailed (infinite variance) and non-stationary (regime-switching) environments due to "sticky priors."
2. **Volatility-Augmented HMM (Phase 5):** The central proposed method. By augmenting a Hidden Markov Model with rolling volatility as a second observational dimension, Kullback-Leibler (KL) divergence increases by $\approx 5\times$, allowing for regime detection lag to drop from 15-20 steps down to 1.9 steps.
3. **Risk-Constrained Kelly (CPPI):** Implementing a Constant Proportion Portfolio Insurance (CPPI) dynamic leverage constraint achieves 100% survival against max-drawdown boundaries, quantifying the exact "cost of survival" (e.g., losing 10 percentage points of theoretical CAGR).
4. **Transaction Cost Sensitivity:** Proving that as trading friction ($c$) increases, the optimal Kelly fraction $\hat{f}$ must be dampened to prevent wealth erosion by turnover—bridging the gap between theoretical growth and net-of-fees reality.
5. **Risk-Adjusted Outperformance:** Proving that while simple Kelly maximizes wealth, our Vol-HMM improves the **Sharpe** and **Sortino** Ratios, providing superior "Investor-Grade" stability compared to standard MAB agents.

## 3. The Live Defense Dashboard (Bloomberg Terminal)
The web application serves as the interactive proving ground during the thesis defense. It is built to impress Mathematics and Statistics professors through sheer computational density.
* **Component 1: Executive Summary Home Page:** A high-impact landing page detailing the "Sticky Prior Paradox," thesis architecture, and implementation roadmap.
* **Component 2: React-Router Multi-Page Architecture:** Includes three distinct pages (Home, Simulator, Academic Logs) with global state persistence via a custom `SimulationContext` provider. Style is a rigid `#000000` pitch-black backend, neon amber/cyan accents, and `Fira Code` monospace typography.
* **Component 3: Live Simulator (Real-Time API):** Executes Monte Carlo arrays testing 8+ agents simultaneously (Thompson, EXP3, UCB, Vol-HMM, CPPI, etc.). Includes sophisticated controls for Transaction Costs (BPS), Drawdown Floors, and HMM Priors.
* **Component 4: Formal Math Console:** A dedicated Bloomberg-style terminal that renders live equity curves and aggregates annualized risk metrics (Sharpe, Sortino) and cumulative fee audits. Includes live interpretation of SciPy-backed statistical tests (Fisher's Exact, Mann-Whitney U).
* **Component 5: Offline Academic Archives:** The "Academic Logs" tab dynamically loads offline-generated figures.

## 4. Academic Deliverables (LaTeX & Paper)
* **The Thesis Document (`main.tex`):** An extensive document spanning 7 chapters (Introduction, Literature, Foundations, Methodology, Optimization, Results, Conclusion), completely cross-referenced with proper mathematical notation (e.g., $F_t$ filtration formulas) and explicitly mapped to the Python code execution arrays.
* **Code-To-Paper Traceability:** Appendices linking the raw Python execution (`compute_kelly_fraction()`) directly to academic formulas ($f^* = \mu/\sigma^2$).
* **Sensitivity Analyses:** Empirical tables computing exact detection lag decay when mutating variables (Rolling Volatility Windows, prior stickiness $A_{ii}$, and CUSUM thresholds $k$).

## 5. Empirical Historical Integration (Real-World Stress Testing)
The project now includes an advanced empirical backtester to bridge the gap between simulation and real-world deployment:
* **Historical Epochs:** 4 distinct S&P 500 epochs (2000 Dot-Com, 2008 GFC, 2020 COVID, 2013-19 Bull) downloaded via Yahoo Finance.
* **Dual-Mode Engine:** A unified API that handles both Monte Carlo synthetic distributions and single-path empirical replays.
* **Real Calendar Logic:** The dashboard dynamically switches from abstract $t$ steps to ISO-formatted calendar dates when in historical mode.
* **Survival Comparison:** Quantifies how agents like `Vol-HMM` and `Risk-Constrained CPPI` navigate true multi-year structural collapses.
