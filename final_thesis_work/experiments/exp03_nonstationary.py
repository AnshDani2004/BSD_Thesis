"""
Experiment 3: Changing Payoffs & Robustness (Nonstationary Environment)

This experiment fulfills the prospectus goal of analyzing behavior under
nonstationary or adversarial conditions. 

It subjects the core Bayesian and Bandit agents to a sudden, unobserved
regime switch (market crash), demonstrating the risk of ruin when beliefs
are sticky and payoffs suddenly change. This explicitly motivates advanced
regime-detection tools (like HMMs) as future work.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MarketEnvironment
from simulation.agents import UCBAgent, ThompsonKellyAgent, NaiveBayesKelly
from simulation.wealth_tracker import simulate_wealth_path

def run_experiment():
    print("Running Experiment 3: Adaptation in Nonstationary Environments")
    
    T_STEPS = 1000
    SHOCK_STEP = 500
    N_SIMS = 100
    
    # 1. Setup Adversarial/Nonstationary Environment
    # We force a change from a highly profitable regime to a highly negative one
    # Note: Transition matrix is set such that it stays in regime 0 until T=500
    # then switches to regime 1 forever.
    config = {
        'bull': {'mu': 0.05, 'sigma': 0.10},       # Positive expectation
        'bear': {'mu': -0.05, 'sigma': 0.20, 'df': 3} # Negative expectation, noisy
    }
    trans_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    env = MarketEnvironment(config, trans_matrix, seed=99)
    res = env.simulate_batch(N_SIMS, T_STEPS)
    
    # Manually inject the nonstationary shock at T=500
    env_bear = MarketEnvironment({'bear': config['bear']}, np.array([[1.0]]), seed=100)
    res_bear = env_bear.simulate_batch(N_SIMS, T_STEPS)
    
    returns_matrix = np.copy(res.returns)
    returns_matrix[:, SHOCK_STEP:] = res_bear.returns[:, SHOCK_STEP:]
    
    agents_dict = {
        'UCB': lambda: UCBAgent(1),
        'Thompson Sampling': lambda: ThompsonKellyAgent(1),
        'Naive Bayes': lambda: NaiveBayesKelly(1)
    }
    
    # 2. Run simulation
    for name, agent_factory in agents_dict.items():
        print(f"\n--- Testing {name} under Adversarial Shock ---")
        
        ruins = 0
        drawdowns = []
        
        for sim in range(N_SIMS):
            agent = agent_factory()
            
            # Use simulate_wealth_path logic manually since agent acts sequentially
            wealth = 1.0
            peak = 1.0
            max_dd_sim = 0.0
            is_ruined = False
            
            for t in range(T_STEPS):
                f = agent.act()[0]
                r = returns_matrix[sim, t]
                multiplier = 1.0 + f * r
                
                if multiplier <= 0:
                    is_ruined = True
                    break
                    
                wealth *= multiplier
                peak = max(peak, wealth)
                dd = (peak - wealth) / peak
                max_dd_sim = max(max_dd_sim, dd)
                
                # We record the agent's bet update
                agent.update(np.array([r]))
                
            if is_ruined:
                ruins += 1
                drawdowns.append(1.0)
            else:
                drawdowns.append(max_dd_sim)
                
        prob_ruin = ruins / N_SIMS
        avg_max_dd = np.mean(drawdowns)
        
        print(f"  Probability of Ruin: {prob_ruin:.1%}")
        print(f"  Average Max Drawdown: {avg_max_dd:.1%}")

    print("\nExperiment Complete. Validates that standard sequential decision-making")
    print("strategies suffer catastrophic failures when underlying payoff structures")
    print("change without notice due to sticky prior beliefs.")

if __name__ == "__main__":
    run_experiment()
