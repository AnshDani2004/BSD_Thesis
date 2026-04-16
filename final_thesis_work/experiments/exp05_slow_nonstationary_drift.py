"""
Experiment 5: Continuous Slow Nonstationary Drift

Sudden shocks (Experiment 4) evaluate how algorithms handle structural breaks,
but real-world environments often experience slow degradation or drift.

This experiment evaluates the response of the algorithms to a Gaussian Random 
Walk on the mean expected return. Because the drift is subtle, Bayesian algorithms
might learn it, but their learning rate is fundamentally bottlenecked by their
prior weights.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.agents import EXP3Agent, ThompsonKellyAgent, SoftmaxKellyAgent
from simulation.wealth_tracker import WealthTracker
from scipy.stats import norm

def run_experiment():
    print("Running Experiment 5: Slow Nonstationary Probability Drift")
    
    T_STEPS = 2000
    N_SIMS = 50
    N_ARMS = 2 
    
    # Generate drift for the arms
    # Arm 0 starts good, decays. Arm 1 starts bad, improves.
    # We'll use a deterministic drift for the experiment to measure precise reactions.
    mu_path_0 = np.linspace(0.08, -0.05, T_STEPS) # 8% down to -5%
    mu_path_1 = np.linspace(-0.05, 0.08, T_STEPS) # -5% up to 8%
    sigma = 0.15
    
    # Cross-over point happens at T=1000
    print("Arm 0 slowly decays from 8% to -5% expected return.")
    print("Arm 1 slowly grows from -5% to 8% expected return.")
    print("Crossover point is exactly at T=1000.")
    print("A perfect algorithm shifts allocation from Arm 0 to Arm 1 exactly at T=1000.")
    
    np.random.seed(42)
    returns_matrix = np.zeros((N_SIMS, T_STEPS, N_ARMS))
    
    for sim in range(N_SIMS):
        returns_matrix[sim, :, 0] = np.random.normal(mu_path_0, sigma)
        returns_matrix[sim, :, 1] = np.random.normal(mu_path_1, sigma)
        
    agents_dict = {
        'Thompson Sampling': lambda: ThompsonKellyAgent(N_ARMS),
        'EXP3 (\u03B3=0.1)': lambda: EXP3Agent(N_ARMS, gamma=0.1),
        'Softmax (\u03C4=0.2)': lambda: SoftmaxKellyAgent(N_ARMS, tau=0.2)
    }
    
    print("\nExecuting drift simulation...")
    print(f"{'Strategy':<20} | {'Median Wealth (x)':<18} | {'Avg Alloc Post-Crossover (Arm 1)'}")
    print("-" * 75)
    
    for name, agent_factory in agents_dict.items():
        wealths = []
        allocations_post_crossover = []
        
        for sim in range(N_SIMS):
            agent = agent_factory()
            tracker = WealthTracker(initial_wealth=1.0)
            
            for t in range(T_STEPS):
                f = agent.act()
                r = returns_matrix[sim, t, :]
                
                if t >= 1000 and np.sum(f) > 0:
                    allocations_post_crossover.append(f[1] / np.sum(f))
                    
                total_return = np.sum(f * r)
                tracker.update(f=1.0, r=total_return)
                agent.update(r)
                
            state = tracker.get_state()
            wealths.append(state.wealth)
            
        avg_alloc = np.mean(allocations_post_crossover) if len(allocations_post_crossover) > 0 else 0
        print(f"{name:<20} | {np.median(wealths):>15.2f}x | {avg_alloc:>29.1%}")

if __name__ == "__main__":
    run_experiment()
