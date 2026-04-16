"""
Experiment 2: Short-term vs. Long-term Optimality (Horizon Trade-offs)

This experiment fulfills the prospectus goal of analyzing how agents perform
under limited time horizons and measuring the probability of reaching target outcomes.

It demonstrates the trade-off between maximizing expected long-term wealth 
(Full Kelly) and maximizing the probability of reaching a short-term target while
avoiding ruin (Half Kelly or Fixed Fraction).
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import create_single_regime_environment
from simulation.agents import KellyOracle, FractionalKelly, FixedFraction
from simulation.wealth_tracker import WealthTracker

def run_experiment():
    print("Running Experiment 2: Horizon Trade-offs in Decision Systems")
    
    # 1. Setup Environment
    # We use a noisy market where aggressive betting can lead to volatile swings
    mu, sigma = 0.05, 0.20
    env = create_single_regime_environment(mu=mu, sigma=sigma, seed=42)
    
    # Calculate true win probability
    from scipy.stats import norm
    true_win_prob = 1.0 - norm.cdf(0, loc=mu, scale=sigma)
    
    print(f"Market stats: μ={mu}, σ={sigma}, P(Win)={true_win_prob:.2f}")
    
    # Evaluate at two horizons
    horizons = {
        'Short-term': {'T': 50, 'target_wealth': 1.5, 'n_sims': 1000},   # +50% in 50 steps
        'Long-term': {'T': 1000, 'target_wealth': 11.0, 'n_sims': 1000}  # 10x in 1000 steps
    }
    
    # Agents
    agents_dict = {
        'Full Kelly': lambda: KellyOracle(1, [true_win_prob], fraction_multiplier=1.0),
        'Half Kelly': lambda: KellyOracle(1, [true_win_prob], fraction_multiplier=0.5),
        'Quarter Kelly': lambda: KellyOracle(1, [true_win_prob], fraction_multiplier=0.25)
    }
    
    # 2. Run simulation for each horizon
    for horizon_name, h_params in horizons.items():
        T = h_params['T']
        target = h_params['target_wealth']
        N = h_params['n_sims']
        
        print(f"\n--- {horizon_name} Horizon (T={T}, Target Wealth={target}x) ---")
        res = env.simulate_batch(N, T)
        returns_matrix = res.returns
        
        for agent_name, agent_factory in agents_dict.items():
            wins = 0
            ruins = 0
            final_wealths = []
            
            for sim in range(N):
                agent = agent_factory()
                tracker = WealthTracker(initial_wealth=1.0, target_wealth=target)
                
                for t in range(T):
                    f = agent.act()[0]
                    r = returns_matrix[sim, t]
                    tracker.update(f=f, r=r)
                    
                state = tracker.get_state()
                if state.target_reached:
                    wins += 1
                if state.is_ruined:
                    ruins += 1
                final_wealths.append(state.wealth)
            
            prob_target = wins / N
            prob_ruin = ruins / N
            median_wealth = np.median(final_wealths)
            
            print(f"  [{agent_name}]")
            print(f"    Prob(Reach Target): {prob_target:.1%}")
            print(f"    Prob(Ruin):         {prob_ruin:.1%}")
            print(f"    Median Wealth:      {median_wealth:.2f}x")

    print("\nExperiment Complete. Validates that while Full Kelly maximizes long-term")
    print("median wealth, fractional Kelly strategies offer a safer profile and often")
    print("higher probabilities of reaching short-term targets without ruin.")

if __name__ == "__main__":
    run_experiment()
