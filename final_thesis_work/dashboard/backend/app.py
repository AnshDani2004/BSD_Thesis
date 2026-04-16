import sys
import os
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

# Add parent to path to import simulation codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.environment import MarketEnvironment, create_single_regime_environment
from simulation.agents import UCBAgent, ThompsonKellyAgent, NaiveBayesKelly, EXP3Agent
from simulation.wealth_tracker import WealthTracker
import scipy.stats as stats

# ===== Historical Data Registry =====
HIST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'historical'
)

HISTORICAL_SCENARIOS = {
    "real_2000_dotcom": {
        "label": "2000 Dot-Com Bubble Collapse",
        "date_start": "2000-01-03",
        "description": "Prolonged multi-year tech equity unwinding. Tests HMM sticky-prior detection across 709 trading days."
    },
    "real_2008_crisis": {
        "label": "2008 Global Financial Crisis",
        "date_start": "2007-06-01",
        "description": "Systemic fat-tailed collapse. Extreme Student-t returns with sudden regime transitions over 523 trading days."
    },
    "real_2020_covid": {
        "label": "2020 COVID-19 Crash & Recovery",
        "date_start": "2020-01-02",
        "description": "33% drawdown in 33 days then V-shaped recovery. Tests CPPI survival floors over 251 trading days."
    },
    "real_2013_bull": {
        "label": "2013-2019 Extended Bull Market",
        "date_start": "2013-01-02",
        "description": "Sustained low-volatility stationary regime over 1,760 days. Validates Kelly growth in stable empirical environments."
    },
}

def load_historical_returns(scenario_key):
    """Load cached .npy log-return array for the given historical scenario key."""
    path = os.path.join(HIST_DATA_DIR, f"{scenario_key}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

def generate_timestamps(t_steps):
    """Generate daily timestamps for TradingView charts."""
    base = datetime(2025, 1, 1)
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(t_steps)]


@app.route('/api/simulate/compare', methods=['POST'])
def simulate_compare():
    data = request.json or {}
    t_steps = int(data.get('t_steps', 250))
    n_sims = int(data.get('n_sims', 20))
    scenario = data.get('scenario', 'adversarial')
    friction = float(data.get('friction', 0.0))
    
    # Advanced Model Calibration
    adv_params = data.get('advanced_params', {})
    cppi_drawdown = float(adv_params.get('cppi_drawdown', 0.20))
    cppi_multiplier = float(adv_params.get('cppi_multiplier', 3.0))
    hmm_bull_mu = float(adv_params.get('hmm_bull_mu', 0.08))
    hmm_bear_mu = float(adv_params.get('hmm_bear_mu', -0.10))
    
    shock_step = t_steps // 2
    is_historical = scenario in HISTORICAL_SCENARIOS
    
    # 1. Setup Environment — Branch: Historical vs Synthetic
    if is_historical:
        hist_returns = load_historical_returns(scenario)
        if hist_returns is None:
            return jsonify({'status': 'error', 'message': f'Historical data not found for {scenario}. Run scripts/fetch_historical_data.py first.'}), 404
        # Single empirical path — tile it as n_sims=1
        t_steps = len(hist_returns)
        n_sims = 1
        returns_matrix = hist_returns.reshape(1, t_steps)
        hist_info = HISTORICAL_SCENARIOS[scenario]
        # Generate real calendar timestamps from actual start date
        from datetime import datetime as dt
        base_date = dt.strptime(hist_info['date_start'], '%Y-%m-%d')
    elif scenario == 'stationary':
        env = create_single_regime_environment(mu=0.08, sigma=0.15, seed=42)
        returns_matrix = env.simulate_batch(n_sims, t_steps).returns
    elif scenario == 'drift':
        mu_path = np.linspace(0.08, -0.05, t_steps)
        returns_matrix = np.zeros((n_sims, t_steps))
        for sim in range(n_sims):
            returns_matrix[sim, :] = np.random.normal(mu_path, 0.15)
    else:  # adversarial
        config = {
            'bull': {'mu': 0.08, 'sigma': 0.15}, 
            'bear': {'mu': -0.10, 'sigma': 0.30, 'df': 3} 
        }
        env_bull = MarketEnvironment({'bull': config['bull']}, np.array([[1.0]]), seed=42)
        res_bull = env_bull.simulate_batch(n_sims, t_steps)
        env_bear = MarketEnvironment({'bear': config['bear']}, np.array([[1.0]]), seed=142)
        res_bear = env_bear.simulate_batch(n_sims, t_steps)
        returns_matrix = np.copy(res_bull.returns)
        returns_matrix[:, shock_step:] = res_bear.returns[:, shock_step:]
    
    # 2. All 8 Agents Context (Including Custom Thesis Proposals)
    from simulation.agents import EpsilonGreedyKelly, SoftmaxKellyAgent
    from simulation.hmm_refined import create_vol_augmented_hmm_kelly
    from simulation.risk_constrained import RiskConstrainedKelly
    
    agents_dict = {
        'Vol-Augmented HMM (Proposed)': lambda: create_vol_augmented_hmm_kelly(
            1, bull_mu=hmm_bull_mu, bear_mu=hmm_bear_mu, bull_sigma=0.15, bear_sigma=0.30
        ),
        'Risk-Constrained CPPI': lambda: RiskConstrainedKelly(
            1, max_drawdown=cppi_drawdown, cppi_multiplier=cppi_multiplier
        ),
        'Thompson Sampling': lambda: ThompsonKellyAgent(1),
        'EXP3': lambda: EXP3Agent(1, gamma=0.1),
        'Naive Bayes': lambda: NaiveBayesKelly(1),
        'UCB': lambda: UCBAgent(1),
        'Softmax': lambda: SoftmaxKellyAgent(1, tau=0.5)
    }
    
    results = {}
    # Use real calendar dates for historical, synthetic step-labels otherwise
    if is_historical:
        timestamps = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(t_steps)]
    else:
        timestamps = generate_timestamps(t_steps)
    
    # Run the models
    for name, agent_factory in agents_dict.items():
        wealth_paths = []
        ruins = 0
        total_fees_list = []
        sharpe_list = []
        sortino_list = []
        
        for sim in range(n_sims):
            agent = agent_factory()
            tracker = WealthTracker(initial_wealth=1.0, friction_bps=friction)
            
            sim_wealth_path = []
            for t in range(t_steps):
                f = agent.act()[0]
                r = returns_matrix[sim, t]
                tracker.update(f, r)
                agent.update(np.array([r]))
                sim_wealth_path.append(tracker.wealth)
                
            state = tracker.get_state()
            wealth_paths.append(sim_wealth_path)
            total_fees_list.append(state.total_fees)
            sharpe_list.append(state.sharpe)
            sortino_list.append(state.sortino)
            if state.is_ruined:
                ruins += 1
                
        # Calculate Aggregates
        wealth_paths = np.array(wealth_paths)
        median_path = np.median(wealth_paths, axis=0)
        mean_fees = float(np.mean(total_fees_list))
        mean_sharpe = float(np.mean(sharpe_list))
        mean_sortino = float(np.mean(sortino_list))
        
        chart_data = [{"time": ts, "value": v} for ts, v in zip(timestamps, median_path)]
        
        results[name] = {
            'chart_data': chart_data,
            'ruin_risk': ruins / n_sims,
            'final_median_wealth': float(median_path[-1]),
            'mean_total_fees': mean_fees,
            'mean_sharpe': mean_sharpe,
            'mean_sortino': mean_sortino
        }
        
    # 3. Dynamic Output Console Formulation
    p_value = 1.0
    sig = False
    
    if is_historical:
        hist_meta = HISTORICAL_SCENARIOS[scenario]
        raw_out = f""">> MODE: EMPIRICAL HISTORICAL DATA
>> SCENARIO: {hist_meta['label']}
>> CALIBRATION: CPPI_DD={cppi_drawdown:.1%} | H_BULL={hmm_bull_mu:+.2f}
>> TRADING DAYS EVALUATED: {t_steps}
>> NOTE: Single-path empirical replay — Transaction Fees tracked.

[CONTEXT] {hist_meta['description']}

[CAUTION] Fisher Exact Test requires n_sims > 1. Displaying direct terminal outcomes.

>> AGENT TERMINAL OUTCOMES (EMPIRICAL):\n"""
        for alg in agents_dict.keys():
            w = results[alg]['final_median_wealth']
            f_paid = results[alg]['mean_total_fees']
            sr = results[alg]['mean_sharpe']
            raw_out += f"  [{alg.upper()[:16]:<16}] -> WEALTH: {w:>7.3f}x | FEES: {f_paid:>5.3f}x | SR: {sr:>5.2f}\n"
    else:
        exp3_ruins = int(results['EXP3']['ruin_risk'] * n_sims)
        hmm_ruins = int(results['Vol-Augmented HMM (Proposed)']['ruin_risk'] * n_sims)
        table = np.array([[hmm_ruins, n_sims - hmm_ruins], [exp3_ruins, n_sims - exp3_ruins]])
        _, p_value = stats.fisher_exact(table, alternative='two-sided')
        sig = bool(p_value < 0.05)
        raw_out = f""">> SCENARIO GENERATED: [{scenario.upper()}]
>> CALIBRATION: CPPI_DD={cppi_drawdown:.1%} | H_BULL={hmm_bull_mu:+.2f}
>> HORIZON: {t_steps} | PATHS: {n_sims} | FRICTION: {friction} bps

[MATH] FISHER EXACT TEST (Vol-HMM vs EXP3):
   P-Value: {p_value:.3e} | Null Rejected: {sig}

"""
        if scenario == "stationary":
            raw_out += ">> INTERPRETATION: Flat parameters. Thompson Sampling trivially converges due to correct prior structure.\n"
        elif scenario == "drift":
            raw_out += ">> INTERPRETATION: Slow drift. Sticky-prior agents (Naive Bayes) accumulate regret as they fail to unlearn past regimes.\n"
        else:
            raw_out += ">> INTERPRETATION: Adversarial discontinuity at T=" + str(t_steps//2) + ". Fat-tailed Student-t Bear Market decimates geometric wealth bounds.\n"
        raw_out += "\n>> ALGORITHM SYNTHESIS:\n"
        for alg in agents_dict.keys():
            w = results[alg]['final_median_wealth']
            rr = results[alg]['ruin_risk'] * 100
            f_paid = results[alg]['mean_total_fees']
            sr = results[alg]['mean_sharpe']
            sor = results[alg]['mean_sortino']
            raw_out += f"  [{alg.upper()[:12]:<12}] -> M_WEALTH: {w:>8.2f}x | RUIN: {rr:>5.1f}% | SR: {sr:>5.2f} | SOR: {sor:>5.2f}\n"

    stats_data = {
        "p_value_ruin": p_value,
        "is_significant": sig,
        "is_historical": is_historical,
        "raw_output": raw_out
    }

    return jsonify({
        "status": "success",
        "strategies": results,
        "statistics": stats_data
    })


if __name__ == '__main__':
    print("Starting Thesis Simulation API on port 5001...")
    # Port 5001 to avoid MacOS AirPlay conflicts on 5000
    app.run(port=5001, debug=True)
