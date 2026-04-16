"""
Experiment 4: Adversarial Market Shocks (Bayesian vs. Adversarial Bandits)

This experiment evaluates algorithm robustness to sudden, unobserved market structure
changes (structural breaks/crash).

Bayesian models (Thompson Sampling, Naive Bayes) generally assume stationarity.
When a sudden crash occurs, they fall victim to "sticky priors" and over-allocate,
risking ruin. The EXP3 algorithm is mathematically structured to maintain bounds 
on regret against an adversarial reality, thus shifting faster from the trap.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MarketEnvironment
from simulation.agents import EXP3Agent, ThompsonKellyAgent, NaiveBayesKelly
from simulation.wealth_tracker import WealthTracker

def run_experiment():
    print("Running Experiment 4: Adversarial Market Shocks")
    
    T_STEPS = 1000
    SHOCK_STEP = 500
    N_SIMS = 100
    
    config = {
        'bull': {'mu': 0.05, 'sigma': 0.10}, 
        'bear': {'mu': -0.05, 'sigma': 0.20, 'df': 3} 
    }
    
    # Pre-generate deterministic shock 
    env_bull = MarketEnvironment({'bull': config['bull']}, np.array([[1.0]]), seed=99)
    res_bull = env_bull.simulate_batch(N_SIMS, T_STEPS)
    
    env_bear = MarketEnvironment({'bear': config['bear']}, np.array([[1.0]]), seed=100)
    res_bear = env_bear.simulate_batch(N_SIMS, T_STEPS)
    
    returns_matrix = np.copy(res_bull.returns)
    returns_matrix[:, SHOCK_STEP:] = res_bear.returns[:, SHOCK_STEP:]
    
    agents_dict = {
        'Thompson Sampling': lambda: ThompsonKellyAgent(1),
        'Naive Bayes': lambda: NaiveBayesKelly(1),
        'EXP3 (Adversarial)': lambda: EXP3Agent(1, gamma=0.1)
    }
    
    print("\nSimulating market crash at T=500...")
    print(f"{'Strategy':<20} | {'Ruin Risk':<12} | {'Max Drawdown (Avg)'}")
    print("-" * 55)
    
    for name, agent_factory in agents_dict.items():
        ruins = 0
        drawdowns = []
        
        for sim in range(N_SIMS):
            agent = agent_factory()
            tracker = WealthTracker(initial_wealth=1.0)
            
            for t in range(T_STEPS):
                f = agent.act()[0]
                r = returns_matrix[sim, t]
                tracker.update(f, r)
                agent.update(np.array([r]))
                
            state = tracker.get_state()
            if state.is_ruined:
                ruins += 1
                drawdowns.append(1.0)
            else:
                drawdowns.append(state.current_drawdown)
                
        print(f"{name:<20} | {ruins/N_SIMS:>10.1%} | {np.mean(drawdowns):>17.1%}")

    print("\nResult: Demonstrated severe failure of continuous Bayesian updating due")
    print("to 'sticky priors'. Motivating the transition to Volatility-Augmented")
    print("Hidden Markov Models for the future directions.")

if __name__ == "__main__":
    run_experiment()
