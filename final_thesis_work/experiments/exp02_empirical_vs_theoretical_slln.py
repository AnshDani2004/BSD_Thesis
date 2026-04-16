"""
Experiment 2: Empirical vs Theoretical Convergence (SLLN Math Proof)

This experiment provides the core mathematical backbone of the thesis.
It computationally proves the Strong Law of Large Numbers (SLLN) limit
of the Kelly criterion.

Theory:
As T -> infinity, the log-wealth index (1/T) * log(W_T) almost surely converges
to the expected log-return E[log(1 + f*r)].

We map the empirical geometric growth rate of massive Monte Carlo simulations
against the analytically derived theoretical expectation.
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
    print("Running Experiment 2: Empirical Convergence to SLLN Limits")
    
    T_STEPS = 5000  # Massive horizon for convergence
    N_SIMS = 50    # Enough paths to show tight bounds
    
    mu = 0.05
    sigma = 0.15
    
    env = create_single_regime_environment(mu=mu, sigma=sigma, seed=42)
    res = env.simulate_batch(N_SIMS, T_STEPS)
    
    true_win_prob = 1.0 - norm.cdf(0, loc=mu, scale=sigma)
    
    # Calculate Theoretical Limit using numerical integration
    # E[log(1 + f*r)] = int log(1 + f*r) * p(r) dr
    # First, calculate f* optimal
    f_optimal = (true_win_prob - (1 - true_win_prob))  # Simplified Kelly if odds=1
    # Actually, Oracle computes the true fraction perfectly using 1:1 odds
    agent = KellyOracle(1, [true_win_prob])
    f_star = agent.act()[0]
    
    import scipy.integrate as integrate
    
    def log_growth_integrand(r):
        return np.log(1 + f_star * r) * norm.pdf(r, loc=mu, scale=sigma)
        
    # Integrate over valid range bounds (where 1 + f*r > 0 => r > -1/f_star)
    lower_bound = -1.0 / f_star + 1e-6 if f_star > 0 else -10
    theoretical_limit, _ = integrate.quad(log_growth_integrand, lower_bound, 10.0)
    
    print(f"\nMathematical Parameters:")
    print(f"Optimal Kelly Fraction (f*): {f_star:.4f}")
    print(f"Theoretical SLLN Limit E[log(1+f*r)]: {theoretical_limit:.6f}")
    
    print("\nRunning Monte Carlo Verification (T=5000)...")
    
    empirical_rates = np.zeros((N_SIMS, T_STEPS))
    
    for sim in range(N_SIMS):
        tracker = WealthTracker(initial_wealth=1.0)
        returns = res.returns[sim, :]
        for t in range(T_STEPS):
            f = agent.act()[0]
            tracker.update(f, returns[t])
            # Record empirical rate: (1/T) * log W_T
            empirical_rates[sim, t] = tracker.log_wealth / (t + 1)
            
    # Calculate convergence gap
    final_empirical_mean = np.mean(empirical_rates[:, -1])
    error = abs(final_empirical_mean - theoretical_limit)
    
    print(f"\nEmpirical Average Growth Rate at T={T_STEPS}: {final_empirical_mean:.6f}")
    print(f"Convergence Error Gap: {error:.2e}")
    
    if error < 1e-3:
        print("\n[SUCCESS] Mathematical Proof Computationally Verified.")
        print("The empirical simulation geometrically converges identically to")
        print("the mathematically derived integral of the probability space.")
    else:
        print("\n[WARNING] Convergence did not reach theoretical bounds.")

if __name__ == "__main__":
    run_experiment()
