"""
Simulation package for Bayesian Sequential Decision-Making thesis.

This package contains the core simulation infrastructure:
- distributions: Numba-optimized random number generators
- environment: MarketEnvironment class 
- agents: Agent hierarchy (Kelly, Thompson, UCB, EXP3, EpsilonGreedy, Softmax, etc.)
- wealth_tracker: Log-space wealth tracking 
"""

from .distributions import (
    sample_markov_chain_batch,
    sample_regime_returns_batch,
    generate_random_samples,
)
from .environment import MarketEnvironment, create_single_regime_environment
from .agents import (
    BaseAgent,
    KellyOracle,
    NaiveBayesKelly,
    ThompsonKellyAgent,
    UCBAgent,
    FixedFraction,
    EpsilonGreedyKelly,
    EXP3Agent,
    SoftmaxKellyAgent,
    compute_kelly_fraction,
)
from .wealth_tracker import WealthTracker
from .hmm_refined import VolAugmentedHMM, create_vol_augmented_hmm_kelly
from .risk_constrained import RiskConstrainedKelly, create_risk_constrained_kelly

__all__ = [
    'sample_markov_chain_batch',
    'sample_regime_returns_batch',
    'generate_random_samples',
    'MarketEnvironment',
    'create_single_regime_environment',
    'BaseAgent',
    'KellyOracle',
    'NaiveBayesKelly',
    'ThompsonKellyAgent',
    'UCBAgent',
    'FixedFraction',
    'EpsilonGreedyKelly',
    'EXP3Agent',
    'SoftmaxKellyAgent',
    'compute_kelly_fraction',
    'WealthTracker',
    'VolAugmentedHMM',
    'create_vol_augmented_hmm_kelly',
    'RiskConstrainedKelly',
    'create_risk_constrained_kelly',
]
