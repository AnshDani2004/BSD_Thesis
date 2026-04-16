"""
Experiment 7: Statistical Significance and Academic Rigor

In sequential decision-making, observing a higher average return is insufficient;
outliers or fat tails can skew means dramatically. This script formally executes
two non-parametric statistical tests to validate our algorithmic superiority:

1. Mann-Whitney U Test: Evaluates if the median final wealth distributions
   between two strategies are significantly different (p < 0.05). Non-parametric,
   thus robust to the log-normal/power-law distribution of compounding wealth.
2. Fisher's Exact Test: Compares the categorical probability of ruin between
   two agents under adversarial shocks.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MarketEnvironment, create_single_regime_environment
from simulation.agents import UCBAgent, NaiveBayesKelly, EXP3Agent, ThompsonKellyAgent
from simulation.wealth_tracker import WealthTracker
import scipy.stats as stats

def get_stationary_results(n_sims=300, t_steps=500):
    env_configs = [
        {'mu': 0.05, 'sigma': 0.10, 'seed': 42}, 
        {'mu': 0.03, 'sigma': 0.20, 'seed': 142},
    ]
    returns_matrix = np.zeros((n_sims, t_steps, 2))
    
    for k, config in enumerate(env_configs):
        env = create_single_regime_environment(mu=config['mu'], sigma=config['sigma'], seed=config['seed'])
        res = env.simulate_batch(n_sims, t_steps)
        returns_matrix[:, :, k] = res.returns
        
    agents_dict = {
        'UCB': lambda: UCBAgent(2),
        'Naive Bayes': lambda: NaiveBayesKelly(2)
    }
    
    results = {}
    for name, agent_factory in agents_dict.items():
        wealths = []
        ruins = 0
        for sim in range(n_sims):
            agent = agent_factory()
            tracker = WealthTracker(initial_wealth=1.0)
            for t in range(t_steps):
                f = agent.act()
                r = returns_matrix[sim, t, :]
                tracker.update(f=1.0, r=np.sum(f * r))
                agent.update(r)
            state = tracker.get_state()
            wealths.append(state.wealth)
            if state.is_ruined: ruins += 1
        results[name] = {'wealths': wealths, 'ruin_count': ruins}
    return results

def get_adversarial_results(n_sims=300, t_steps=500):
    config = {
        'bull': {'mu': 0.05, 'sigma': 0.10}, 
        'bear': {'mu': -0.05, 'sigma': 0.20, 'df': 3} 
    }
    env_bull = MarketEnvironment({'bull': config['bull']}, np.array([[1.0]]), seed=99)
    res_bull = env_bull.simulate_batch(n_sims, t_steps)
    env_bear = MarketEnvironment({'bear': config['bear']}, np.array([[1.0]]), seed=100)
    res_bear = env_bear.simulate_batch(n_sims, t_steps)
    
    returns_matrix = np.copy(res_bull.returns)
    shock_step = t_steps // 2
    returns_matrix[:, shock_step:] = res_bear.returns[:, shock_step:]
    
    agents_dict = {
        'EXP3': lambda: EXP3Agent(1, gamma=0.1),
        'Thompson Sampling': lambda: ThompsonKellyAgent(1)
    }
    
    results = {}
    for name, agent_factory in agents_dict.items():
        wealths = []
        ruins = 0
        for sim in range(n_sims):
            agent = agent_factory()
            tracker = WealthTracker(initial_wealth=1.0)
            for t in range(t_steps):
                f = agent.act()[0]
                r = returns_matrix[sim, t]
                tracker.update(f, r)
                agent.update(np.array([r]))
            state = tracker.get_state()
            wealths.append(state.wealth)
            if state.is_ruined: ruins += 1
        results[name] = {'wealths': wealths, 'ruin_count': ruins}
    return results

def run_experiment():
    print("Running Experiment 7: Academic Statistical Validation\n")
    N_SIMS = 300
    
    print("Test 1: Mann-Whitney U Test (Median Wealth Outperformance)")
    print("Scenario: Stationary MAB (UCB vs. Naive Bayes)")
    stat_res = get_stationary_results(n_sims=N_SIMS)
    
    ucb_wealths = stat_res['UCB']['wealths']
    naive_wealths = stat_res['Naive Bayes']['wealths']
    
    # Mann-Whitney U test (one-sided: UCB > Naive)
    u_stat, p_value_mw = stats.mannwhitneyu(ucb_wealths, naive_wealths, alternative='greater')
    
    print(f"  UCB Median Wealth:         {np.median(ucb_wealths):.2f}")
    print(f"  Naive Bayes Median Wealth: {np.median(naive_wealths):.2f}")
    print(f"  Mann-Whitney U-statistic:  {u_stat:.1f}")
    print(f"  p-value:                   {p_value_mw:.3e}")
    if p_value_mw < 0.05:
        print("  Conclusion: UCB final wealth is stochastically strictly greater (Significant)")
    else:
        print("  Conclusion: No significant outperformance.")
        
    print("\n--------------------------------------------------------------\n")
        
    print("Test 2: Fisher's Exact Test (Probability of Ruin)")
    print("Scenario: Adversarial Market Crash (EXP3 vs. Thompson Sampling)")
    adv_res = get_adversarial_results(n_sims=N_SIMS)
    
    exp3_ruins = adv_res['EXP3']['ruin_count']
    exp3_survives = N_SIMS - exp3_ruins
    
    ts_ruins = adv_res['Thompson Sampling']['ruin_count']
    ts_survives = N_SIMS - ts_ruins
    
    # Contingency Table
    #           | Ruined | Survived |
    # -------------------------------
    # TS        |   ...  |   ...    |
    # EXP3      |   ...  |   ...    |
    table = np.array([[ts_ruins, ts_survives], [exp3_ruins, exp3_survives]])
    oddsratio, p_value_fisher = stats.fisher_exact(table, alternative='two-sided')
    
    print(f"  Thompson Sampling Ruin: {ts_ruins}/{N_SIMS} ({ts_ruins/N_SIMS:.1%})")
    print(f"  EXP3 Ruin:              {exp3_ruins}/{N_SIMS} ({exp3_ruins/N_SIMS:.1%})")
    print(f"  Odds Ratio:             {oddsratio:.2f}")
    print(f"  p-value:                {p_value_fisher:.3e}")
    if p_value_fisher < 0.05:
        if exp3_ruins < ts_ruins:
            print("  Conclusion: EXP3 strictly significantly reduces ruin risk during market regime changes.")
        else:
            print("  Conclusion: Thompson Sampling strictly significantly reduces ruin risk during market regime changes compared to EXP3.")
    else:
        print("  Conclusion: No significant difference in ruin risk.")

if __name__ == "__main__":
    run_experiment()
