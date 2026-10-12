"""Outcome c: the thesis agents, unchanged, on seeded bounded (and, for comparison, unbounded) paths.

The agents are the classes in simulation/ called exactly as experiments/generate_all_figures.py calls
them on branch audit-cppi-floor (CPPI receives the tracker wealth; every other agent gets
update(np.array([r]))). The only change: Thompson Sampling gets an explicit Generator built from
(path seed, stream 4). Agents observe the simple return r.
"""
import os
import sys
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))      # final_thesis_work
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import dgp  # noqa: E402

AGENTS = ('Vol-HMM (Proposed)', 'Risk-Constrained CPPI', 'Thompson Sampling', 'EXP3', 'Naive Bayes', 'Softmax')
SCENARIOS = ('Stationary', 'Adversarial', 'Drift')
PRACTICAL_RUIN = 0.10   # wealth at or below 10% of the start on the step-end grid


def make_agent(name, seed):
    from simulation.agents import ThompsonKellyAgent, NaiveBayesKelly, EXP3Agent, SoftmaxKellyAgent
    from simulation.hmm_refined import create_vol_augmented_hmm_kelly
    from simulation.risk_constrained import RiskConstrainedKelly
    if name == 'Vol-HMM (Proposed)':
        return create_vol_augmented_hmm_kelly(1, 0.08, 0.15, -0.10, 0.30)
    if name == 'Risk-Constrained CPPI':
        return RiskConstrainedKelly(1, max_drawdown=0.20, cppi_multiplier=3.0)
    if name == 'Thompson Sampling':
        return ThompsonKellyAgent(1, rng=dgp.path_rng(seed, 4))
    if name == 'EXP3':
        return EXP3Agent(1, gamma=0.1)
    if name == 'Naive Bayes':
        return NaiveBayesKelly(1)
    if name == 'Softmax':
        return SoftmaxKellyAgent(1, tau=0.5)
    raise ValueError(name)


def scenario_log_returns(scenario, seeds):
    Z, V = dgp.step_draws(seeds)
    if scenario == 'Stationary':
        return dgp.stationary_log_returns(Z)
    if scenario == 'Adversarial':
        return dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, dgp.T_STEPS - dgp.BREAK_T)
    if scenario == 'Drift':
        return dgp.drift_log_returns(Z)
    raise ValueError(scenario)


def run_agent(args):
    """args = (agent_name, scenario, model, seeds). Returns dict of per-path arrays (lists)."""
    name, scenario, model, seeds = args
    warnings.filterwarnings('ignore')
    from simulation.risk_constrained import RiskConstrainedKelly
    from simulation.wealth_tracker import WealthTracker
    R = dgp.to_simple(scenario_log_returns(scenario, seeds), model)
    n, T = R.shape
    term = np.empty(n)
    minw = np.empty(n)
    ruined = np.zeros(n, bool)
    mdd = np.empty(n)
    for i, s in enumerate(seeds):
        agent = make_agent(name, s)
        tr = WealthTracker(initial_wealth=1.0)
        peak, worst, mw = 1.0, 0.0, 1.0
        for t in range(T):
            f = agent.act()[0]
            r = R[i, t]
            tr.update(f, r)
            if isinstance(agent, RiskConstrainedKelly):
                agent.update(np.array([r]), wealth=tr.wealth)
            else:
                agent.update(np.array([r]))
            w = tr.wealth
            mw = min(mw, w)
            peak = max(peak, w)
            worst = max(worst, 1.0 - w / peak)
        term[i] = tr.wealth
        minw[i] = mw
        ruined[i] = tr.is_ruined
        mdd[i] = worst
    return {'agent': name, 'scenario': scenario, 'model': model, 'terminal': term, 'min_w': minw,
            'ruined': ruined, 'mdd': mdd}
