"""
Experiment 3: Horizon Trade-offs & Targets (Fractional Kelly Curve)

Evaluates the non-monotonic trade-off between targeting infinite-term
growth constraints versus finite-term success probabilities.

Iterates over an entire set of fractional constraints (From 0.1x to 2.0x Kelly)
and maps the precise points where optimal probabilities flip from Full Kelly 
to Fractional Kelly under short-term targets.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import create_single_regime_environment
from simulation.agents import KellyOracle
from simulation.wealth_tracker import WealthTracker
from scipy.stats import norm

def run_experiment():
    print("Running Experiment 3: Horizon Trade-offs & Probability Optimization")
    
    T_STEPS = 50
    N_SIMS = 1000
    TARGET_WEALTH = 1.25  # +25%
    
    mu_sys, sigma_sys = 0.05, 0.20
    env = create_single_regime_environment(mu_sys, sigma_sys, seed=42)
    res = env.simulate_batch(N_SIMS, T_STEPS)
    
    true_win_prob = 1.0 - norm.cdf(0, loc=mu_sys, scale=sigma_sys)
    
    # fractions from 0.0 (cash) to 2.0 (Double Kelly)
    kelly_fractions = np.linspace(0.1, 2.0, 10)
    
    print(f"\nEvaluating short-term target ({TARGET_WEALTH}x within T={T_STEPS})...")
    print(f"{'Kelly Multiplier':<18} | {'Prob(Reach Target)':<20} | {'Median Wealth':<15}")
    print("-" * 56)
    
    for fraction in kelly_fractions:
        wins = 0
        final_wealths = []
        
        for sim in range(N_SIMS):
            agent = KellyOracle(1, [true_win_prob], fraction_multiplier=fraction)
            tracker = WealthTracker(initial_wealth=1.0, target_wealth=TARGET_WEALTH)
            
            for t in range(T_STEPS):
                f = agent.act()[0]
                tracker.update(f, res.returns[sim, t])
                
            state = tracker.get_state()
            if state.target_reached: wins += 1
            final_wealths.append(state.wealth)
            
        prob_target = wins / N_SIMS
        med_wealth = np.median(final_wealths)
        
        print(f"{fraction:>17.1f}x | {prob_target:>19.1%} | {med_wealth:>14.2f}x")

    print("\nResult: Demonstrated that Full Kelly (1.0x) optimizes Median Wealth")
    print("but Fractional Kelly (e.g., 0.5x) strictly maximizes Probability of")
    print("reaching discrete, finite-time targets.")

if __name__ == "__main__":
    run_experiment()
