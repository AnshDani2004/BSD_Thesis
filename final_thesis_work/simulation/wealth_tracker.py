"""
Wealth Tracker with Log-Space Arithmetic

Provides numerically stable wealth tracking for long simulations.
Uses log-wealth internally to prevent overflow/underflow.

Key Features:
- Log-space updates: log_w += log1p(f * r)
- Explicit ruin handling when 1 + f*r <= 0
- Drawdown tracking from peak wealth
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class WealthState:
    """Current state of wealth tracker."""
    wealth: float
    log_wealth: float
    peak_wealth: float
    current_drawdown: float
    is_ruined: bool
    n_steps: int
    total_fees: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    target_reached: bool = False
    target_reached_step: Optional[int] = None


class WealthTracker:
    """
    Log-space wealth tracking for numerical stability.
    
    In long simulations, direct multiplicative updates can overflow:
        wealth *= (1 + f * r)  # Unstable for large T
    
    Log-space is stable:
        log_wealth += log1p(f * r)  # Stable for any T
    
    Example:
        >>> tracker = WealthTracker(initial_wealth=1.0)
        >>> tracker.update(f=0.5, r=0.10)  # 10% return, half Kelly
        >>> print(tracker.wealth)  # ~1.05
        >>> tracker.update(f=0.5, r=-0.20)  # -20% return
        >>> print(tracker.wealth)  # ~0.945
    """
    
    def __init__(self, initial_wealth: float = 1.0, target_wealth: Optional[float] = None, friction_bps: float = 0.0):
        """
        Initialize wealth tracker.
        
        Args:
            initial_wealth: Starting wealth (default: 1.0)
            target_wealth: Optional target sequence wealth goal
            friction_bps: Transaction cost in basis points (1 bps = 0.0001)
        """
        if initial_wealth <= 0:
            raise ValueError("Initial wealth must be positive")
        
        self._initial_wealth = initial_wealth
        self._log_wealth = np.log(initial_wealth)
        self._peak_log_wealth = self._log_wealth
        self._target_wealth = target_wealth
        self._target_log_wealth = np.log(target_wealth) if target_wealth else np.inf
        self._friction = friction_bps / 10000.0
        
        self._is_ruined = False
        self._target_reached = False
        self._n_steps = 0
        self._last_f = 0.0
        self._total_fees = 0.0
        self._sum_r = 0.0
        self._sum_r2 = 0.0
        self._sum_r_neg2 = 0.0
        self._ruin_step: Optional[int] = None
        self._target_reached_step: Optional[int] = None
    
    def update(self, f: float, r: float) -> bool:
        """
        Update wealth with betting fraction and return, deducting friction.
        
        Uses log-space arithmetic with friction deduction:
            cost = friction * |f_t - f_prev_adjusted|
            multiplier = 1 + f_t * r - cost
            log_wealth += log(multiplier)
        
        If multiplier <= 0, marks as ruined.
        
        Args:
            f: Betting fraction (0 <= f <= 1)
            r: Return for this period
        
        Returns:
            True if still solvent, False if ruined
        """
        if self._is_ruined:
            return False
        
        # Calculate friction-adjusted f from previous step
        # After the price move r, the previous position f_old has effectively changed
        denom = (1.0 + self._last_f * r)
        f_prev_adj = (self._last_f * (1.0 + r)) / denom if denom > 0 else self._last_f
        
        # Cost is applied to the turnover (change in fraction)
        cost = self._friction * abs(f - f_prev_adj)
        
        multiplier = 1.0 + f * r - cost
        
        if multiplier <= 0:
            # Ruin: loss (including cost) exceeds current wealth
            self._is_ruined = True
            self._log_wealth = -np.inf
            self._ruin_step = self._n_steps
            self._n_steps += 1
            return False
        
        # Accumulate absolute fees in currency units
        # applied to current wealth before the update
        self._total_fees += cost * self.wealth
        
        # Log-space update (numerically stable)
        current_r = multiplier - 1.0
        self._sum_r += current_r
        self._sum_r2 += current_r**2
        self._sum_r_neg2 += min(0.0, current_r)**2
        
        self._log_wealth += np.log(multiplier)
        self._peak_log_wealth = max(self._peak_log_wealth, self._log_wealth)
        self._last_f = f
        self._n_steps += 1
        
        # Check target
        if not self._target_reached and self._log_wealth >= self._target_log_wealth:
            self._target_reached = True
            self._target_reached_step = self._n_steps
            
        return True
    
    def update_batch(self, f: np.ndarray, r: np.ndarray) -> int:
        """
        Batch update for multiple periods.
        
        Args:
            f: Array of betting fractions
            r: Array of returns
        
        Returns:
            Number of steps before ruin (or len(f) if no ruin)
        """
        for i, (fi, ri) in enumerate(zip(f, r)):
            if not self.update(fi, ri):
                return i + 1
        return len(f)
    
    @property
    def wealth(self) -> float:
        """Current wealth (converted from log-space)."""
        if self._is_ruined:
            return 0.0
        return np.exp(self._log_wealth)
    
    @property
    def log_wealth(self) -> float:
        """Current log-wealth (for direct access)."""
        return self._log_wealth
    
    @property
    def peak_wealth(self) -> float:
        """Peak wealth reached (high-water mark)."""
        return np.exp(self._peak_log_wealth)
    
    @property
    def current_drawdown(self) -> float:
        """
        Current drawdown from peak.
        
        Returns:
            Drawdown as fraction: (peak - current) / peak
        """
        if self._is_ruined:
            return 1.0
        if self._peak_log_wealth == -np.inf:
            return 0.0
        
        # Compute in log-space for stability
        log_drawdown = self._peak_log_wealth - self._log_wealth
        return 1.0 - np.exp(-log_drawdown)
    
    @property
    def is_ruined(self) -> bool:
        """Whether the tracker has hit ruin."""
        return self._is_ruined
    
    @property
    def n_steps(self) -> int:
        """Number of steps executed."""
        return self._n_steps
    @property
    def target_reached(self) -> bool:
        """Whether the target wealth has been reached."""
        return self._target_reached
        
    @property
    def total_fees(self) -> float:
        """Total cumulative transaction fees paid."""
        return self._total_fees
        
    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio (assuming risk-free = 0)."""
        if self._n_steps < 2: return 0.0
        mean = self._sum_r / self._n_steps
        var = (self._sum_r2 / self._n_steps) - mean**2
        std = np.sqrt(max(1e-9, var))
        return (mean / std) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino Ratio (downside risk only)."""
        if self._n_steps < 2: return 0.0
        mean = self._sum_r / self._n_steps
        downside_std = np.sqrt(max(1e-9, self._sum_r_neg2 / self._n_steps))
        return (mean / downside_std) * np.sqrt(252)

    @property
    def target_reached_step(self) -> Optional[int]:
        """Step at which target was reached."""
        return self._target_reached_step
    
    def get_state(self) -> WealthState:
        """Get current state as dataclass."""
        return WealthState(
            wealth=self.wealth,
            log_wealth=self._log_wealth,
            peak_wealth=self.peak_wealth,
            current_drawdown=self.current_drawdown,
            is_ruined=self._is_ruined,
            n_steps=self._n_steps,
            total_fees=self._total_fees,
            sharpe=self.sharpe_ratio,
            sortino=self.sortino_ratio,
            target_reached=self._target_reached,
            target_reached_step=self._target_reached_step
        )
    
    def reset(self, initial_wealth: Optional[float] = None) -> None:
        """Reset tracker to initial state."""
        if initial_wealth is not None:
            self._initial_wealth = initial_wealth
        
        self._log_wealth = np.log(self._initial_wealth)
        self._peak_log_wealth = self._log_wealth
        self._target_reached = False
        self._n_steps = 0
        self._ruin_step = None
        self._target_reached_step = None
    
    def __repr__(self) -> str:
        status = "RUINED" if self._is_ruined else "ACTIVE"
        return (
            f"WealthTracker({status}, "
            f"wealth={self.wealth:.4f}, "
            f"drawdown={self.current_drawdown:.1%}, "
            f"steps={self._n_steps})"
        )


def simulate_wealth_path(
    fractions: np.ndarray,
    returns: np.ndarray,
    initial_wealth: float = 1.0,
    friction_bps: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Simulate wealth path with log-space arithmetic and friction.
    
    Args:
        fractions: Betting fractions at each step
        returns: Returns at each step
        initial_wealth: Starting wealth
        friction_bps: Transaction cost in bps
    
    Returns:
        wealth: Array of wealth at each step
        drawdowns: Array of drawdowns at each step
        is_ruined: Whether ruin occurred
    """
    T = len(returns)
    log_wealth = np.zeros(T + 1)
    log_wealth[0] = np.log(initial_wealth)
    
    friction = friction_bps / 10000.0
    last_f = 0.0
    total_fees = 0.0
    sum_r = 0.0
    sum_r2 = 0.0
    sum_r_neg2 = 0.0
    
    peak_log_wealth = log_wealth[0]
    drawdowns = np.zeros(T + 1)
    is_ruined = False
    
    for t in range(T):
        f = fractions[t]
        r = returns[t]
        
        # Turnover adj
        denom = (1.0 + last_f * r)
        f_prev_adj = (last_f * (1.0 + r)) / denom if denom > 0 else last_f
        cost = friction * abs(f - f_prev_adj)
        
        multiplier = 1.0 + f * r - cost
        
        if multiplier <= 0:
            log_wealth[t + 1:] = -np.inf
            drawdowns[t + 1:] = 1.0
            is_ruined = True
            break
        
        # Track metrics
        total_fees += cost * np.exp(log_wealth[t])
        current_r = multiplier - 1.0
        sum_r += current_r
        sum_r2 += current_r**2
        sum_r_neg2 += min(0.0, current_r)**2
        
        log_wealth[t + 1] = log_wealth[t] + np.log(multiplier)
        peak_log_wealth = max(peak_log_wealth, log_wealth[t + 1])
        drawdowns[t + 1] = 1.0 - np.exp(log_wealth[t + 1] - peak_log_wealth)
        last_f = f
    
    wealth = np.exp(log_wealth)
    wealth[np.isinf(log_wealth)] = 0.0
    
    return wealth, drawdowns, is_ruined, total_fees
