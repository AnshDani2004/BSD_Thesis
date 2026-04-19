import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# Force stable plotting backend and style
import matplotlib
matplotlib.use('Agg') # Headless, avoids NSWindow errors on MacOS

# Bloomberg/Dark Mode Academic Aesthetic
plt.style.use('dark_background')
sns.set_context("paper", font_scale=1.2)

COLORS = {
    'Vol-HMM (Proposed)':            '#00FF41',  # Matrix Green  (proposed method 1)
    'Risk-Constrained CPPI':         '#FF33FF',  # Neon Magenta  (proposed method 2)
    'Thompson Sampling':             '#FFB000',  # Amber
    'EXP3':                          '#00E5FF',  # Cyan
    'Naive Bayes':                   '#FF0033',  # Red
    'UCB':                           '#FFFFFF',  # White
    'Epsilon-Greedy':                '#A64D79',  # Purple
    'Softmax':                       '#FF9900',  # Orange
    # Flask API uses these full names:
    'Vol-Augmented HMM (Proposed)':  '#00FF41',
}

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dashboard', 'frontend', 'public', 'results', 'figures'
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dashboard', 'frontend', 'public', 'results', 'tables'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def configure_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color='#00E5FF', weight='bold', pad=15)
    ax.set_xlabel(xlabel, color='#CCCCCC')
    ax.set_ylabel(ylabel, color='#CCCCCC')
    ax.grid(color='#333333', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')

def plot_wealth_fan(wealth_paths_dict, t_steps, experiment_name):
    """Plots the 10th, 50th, and 90th percentile wealth paths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    time_arr = np.arange(t_steps)
    
    for agent_name, paths in wealth_paths_dict.items():
        if len(paths) == 0: continue
        paths_arr = np.array(paths)
        p10 = np.percentile(paths_arr, 10, axis=0)
        p50 = np.percentile(paths_arr, 50, axis=0)
        p90 = np.percentile(paths_arr, 90, axis=0)
        
        c = COLORS.get(agent_name, '#FFFFFF')
        ax.plot(time_arr, p50, label=f"{agent_name} (Median)", color=c, lw=2)
        ax.fill_between(time_arr, p10, p90, color=c, alpha=0.15)

    configure_axes(ax, f"Monte Carlo Wealth Trajectories ({experiment_name})", "Time step (t)", "Log-Wealth (w_t)")
    ax.set_yscale('symlog')
    ax.legend(loc='upper left', frameon=True, facecolor='#050505', edgecolor='#333333')
    
    out_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_wealth_fan.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#000000')
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_cdf_final_wealth(final_wealth_dict, experiment_name):
    """Plots empirical CDF of final wealth to show heavy tails/ruin risk."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for agent_name, w_final in final_wealth_dict.items():
        if len(w_final) == 0: continue
        w_sorted = np.sort(np.array(w_final))
        p = 1. * np.arange(len(w_sorted)) / (len(w_sorted) - 1)
        c = COLORS.get(agent_name, '#FFFFFF')
        ax.plot(w_sorted, p, label=agent_name, color=c, lw=2)

    configure_axes(ax, f"Empirical CDF of Final Wealth ({experiment_name})", "Final Wealth (Log Scale)", "P(W <= w)")
    ax.set_xscale('symlog')
    ax.axhline(0.5, color='#FF0033', linestyle=':', label='Median (50%)', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='#050505')
    
    out_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_cdf.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor='#000000')
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_regret_dynamics(regret_paths_dict, t_steps, experiment_name):
    """Plots median cumulative regret over time against the Oracle path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    time_arr = np.arange(t_steps)
    
    for agent_name, paths in regret_paths_dict.items():
        if len(paths) == 0: continue
        p50 = np.percentile(paths, 50, axis=0)
        c = COLORS.get(agent_name, '#FFFFFF')
        ax.plot(time_arr, p50, label=f"{agent_name}", color=c, lw=2)

    configure_axes(ax, f"Cumulative Regret Deviations ({experiment_name})", "Time step (t)", "Regret (Oracle Log-Wealth - Agent Log-Wealth)")
    ax.legend(loc='upper right', frameon=True, facecolor='#050505', edgecolor='#333333')
    
    out_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_regret_dynamics.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#000000')
    plt.close(fig)
    print(f"Saved {out_path}")

def dump_experiment_data(test_name, stats_dict):
    """Dumps formal statistical findings into a JSON table."""
    out_path = os.path.join(DATA_DIR, f"{test_name}_stats.json")
    with open(out_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Saved {out_path}")
