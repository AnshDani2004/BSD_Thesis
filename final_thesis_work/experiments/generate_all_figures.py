import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MarketEnvironment, create_single_regime_environment
from simulation.agents import UCBAgent, ThompsonKellyAgent, NaiveBayesKelly, EXP3Agent, EpsilonGreedyKelly, SoftmaxKellyAgent
from simulation.wealth_tracker import WealthTracker
from experiments.plot_engine import plot_wealth_fan, plot_cdf_final_wealth, dump_experiment_data, plot_regret_dynamics

def run_experiment(name, returns_matrix, t_steps, n_sims, oracle_f_path):
    from simulation.agents import UCBAgent, ThompsonKellyAgent, NaiveBayesKelly, EXP3Agent, EpsilonGreedyKelly, SoftmaxKellyAgent
    from simulation.hmm_refined import create_vol_augmented_hmm_kelly
    from simulation.risk_constrained import RiskConstrainedKelly
    
    agents_dict = {
        'Vol-HMM (Proposed)': lambda: create_vol_augmented_hmm_kelly(1, 0.08, 0.15, -0.10, 0.30),
        'Risk-Constrained CPPI': lambda: RiskConstrainedKelly(1, max_drawdown=0.20, cppi_multiplier=3.0),
        'Thompson Sampling': lambda: ThompsonKellyAgent(1),
        'EXP3': lambda: EXP3Agent(1, gamma=0.1),
        'Naive Bayes': lambda: NaiveBayesKelly(1),
        'Softmax': lambda: SoftmaxKellyAgent(1, tau=0.5)
    }
    
    wealth_paths_dict = {}
    regret_paths_dict = {}
    final_wealth_dict = {}
    stats_dict = {'metadata': {'experiment': name, 't_steps': t_steps, 'n_sims': n_sims}, 'agents': {}}
    
    for agent_name, factory in agents_dict.items():
        wealths = []
        regrets = []
        ruins = 0
        for sim in range(n_sims):
            agent = factory()
            tracker = WealthTracker(initial_wealth=1.0)
            sim_path = []
            sim_regret = []
            log_w_oracle = 0.0
            for t in range(t_steps):
                f = agent.act()[0]
                r = returns_matrix[sim, t]
                tracker.update(f, r)
                agent.update(np.array([r]))
                sim_path.append(tracker.wealth)
                
                # Oracle Math
                f_opt = oracle_f_path[t]
                log_w_oracle += np.log1p(f_opt * r)
                sim_regret.append(log_w_oracle - tracker.log_wealth)
                
            state = tracker.get_state()
            wealths.append(sim_path)
            regrets.append(sim_regret)
            if state.is_ruined: ruins += 1
            
        wealth_paths_dict[agent_name] = wealths
        regret_paths_dict[agent_name] = regrets
        finals = [w[-1] for w in wealths]
        final_wealth_dict[agent_name] = finals
        
        stats_dict['agents'][agent_name] = {
            'median_wealth': float(np.median(finals)),
            'ruin_risk': ruins / n_sims,
            'max_wealth': float(np.max(finals))
        }
        
    print(f"Generating Plots for {name}...")
    plot_wealth_fan(wealth_paths_dict, t_steps, name)
    plot_cdf_final_wealth(final_wealth_dict, name)
    plot_regret_dynamics(regret_paths_dict, t_steps, name)
    dump_experiment_data(name, stats_dict)

def generate_all():
    t_steps = 250
    n_sims = 100
    
    # Exp 1: Stationary MAB
    print("\n--- Running Stationary Scenario ---")
    env_stat = create_single_regime_environment(mu=0.08, sigma=0.15, seed=42)
    res_stat = env_stat.simulate_batch(n_sims, t_steps)
    oracle_stat = np.clip(0.08 / (0.15**2), 0, 1) * np.ones(t_steps)
    run_experiment("Stationary", res_stat.returns, t_steps, n_sims, oracle_stat)
    
    # Exp 4: Adversarial Shock
    print("\n--- Running Adversarial Scenario ---")
    config = {
        'bull': {'mu': 0.08, 'sigma': 0.15}, 
        'bear': {'mu': -0.10, 'sigma': 0.30, 'df': 3} 
    }
    env_bull = MarketEnvironment({'bull': config['bull']}, np.array([[1.0]]), seed=42)
    env_bear = MarketEnvironment({'bear': config['bear']}, np.array([[1.0]]), seed=142)
    
    res_adv = np.copy(env_bull.simulate_batch(n_sims, t_steps).returns)
    res_adv[:, t_steps//2:] = env_bear.simulate_batch(n_sims, t_steps).returns[:, t_steps//2:]
    
    oracle_adv = np.zeros(t_steps)
    oracle_adv[:t_steps//2] = np.clip(0.08 / (0.15**2), 0, 1)
    oracle_adv[t_steps//2:] = np.clip(-0.10 / (0.30**2), 0, 1)
    run_experiment("Adversarial", res_adv, t_steps, n_sims, oracle_adv)

    # Exp 5: Slow Drift (Regime Shift slowly)
    print("\n--- Running Slow Drift Scenario ---")
    mu_path = np.linspace(0.08, -0.05, t_steps)
    res_drift = np.zeros((n_sims, t_steps))
    oracle_drift = np.zeros(t_steps)
    
    for sim in range(n_sims):
        res_drift[sim, :] = np.random.normal(mu_path, 0.15)
        
    for t in range(t_steps):
        oracle_drift[t] = np.clip(mu_path[t] / (0.15**2), 0, 1)
        
    run_experiment("Drift", res_drift, t_steps, n_sims, oracle_drift)
    
    print("\n✅ All Static Experiment Figures & Tables Generated Successfully.")

if __name__ == "__main__":
    generate_all()
