"""
Advanced Risk Management

- Kelly Criterion for optimal position sizing
- CVaR (Conditional Value at Risk)
- Omega Ratio
- Ulcer Index
- Advanced performance metrics

References:
- Kelly (1956) - A new interpretation of information rate
- Rockafellar & Uryasev (2000) - CVaR optimization
- Keating & Shadwick (2002) - Omega ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing

    Formula: f* = (p * b - q) / b
    where:
    - f* = optimal fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = win/loss ratio
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Initialize Kelly Criterion calculator

        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly, more conservative)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_from_trades(self, trades: List[Dict]) -> float:
        """
        Calculate Kelly position size from trade history

        Args:
            trades: List of trade dictionaries with 'pnl' field

        Returns:
            Optimal position size (fraction of capital)
        """
        if not trades:
            return 0.0

        pnls = [t['pnl'] for t in trades]

        # Win rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses:
            return 0.0

        p = len(wins) / len(pnls)  # Win probability
        q = 1 - p  # Loss probability

        # Win/loss ratio
        avg_win = np.mean([abs(w) for w in wins])
        avg_loss = np.mean([abs(l) for l in losses])
        b = avg_win / avg_loss if avg_loss > 0 else 0

        # Kelly formula
        kelly = (p * b - q) / b

        # Apply Kelly fraction (for safety)
        kelly_adjusted = max(0, kelly * self.kelly_fraction)

        # Cap at reasonable maximum
        kelly_capped = min(kelly_adjusted, 0.25)  # Max 25% of capital

        return kelly_capped

    def calculate_from_probabilities(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly from win probability and win/loss ratio

        Args:
            win_prob: Probability of winning trade
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal position size
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        if win_loss_ratio <= 0:
            return 0.0

        p = win_prob
        q = 1 - p
        b = win_loss_ratio

        kelly = (p * b - q) / b
        kelly_adjusted = max(0, kelly * self.kelly_fraction)
        kelly_capped = min(kelly_adjusted, 0.25)

        return kelly_capped


class AdvancedRiskMetrics:
    """Advanced risk and performance metrics"""

    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)

        CVaR is the expected loss given that the loss exceeds VaR.

        Args:
            returns: Series of returns
            alpha: Confidence level (0.05 = 95% confidence)

        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0

        # Calculate VaR
        var = returns.quantile(alpha)

        # CVaR = average of returns below VaR
        cvar = returns[returns <= var].mean()

        return float(abs(cvar))

    @staticmethod
    def calculate_omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega Ratio

        Omega = (Gains above threshold) / (Losses below threshold)

        Args:
            returns: Series of returns
            threshold: Return threshold (usually 0 or risk-free rate)

        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0

        omega = gains.sum() / losses.sum()

        return float(omega)

    @staticmethod
    def calculate_ulcer_index(equity_curve: pd.Series) -> float:
        """
        Calculate Ulcer Index

        Measures the depth and duration of drawdowns.

        Args:
            equity_curve: Series of equity values

        Returns:
            Ulcer Index
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown percentage
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        # Ulcer Index = sqrt(mean of squared drawdowns)
        ulcer = np.sqrt((drawdown_pct ** 2).mean())

        return float(abs(ulcer))

    @staticmethod
    def calculate_pain_index(equity_curve: pd.Series) -> float:
        """
        Calculate Pain Index

        Sum of all drawdowns divided by number of periods.

        Args:
            equity_curve: Series of equity values

        Returns:
            Pain Index
        """
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        pain = abs(drawdown_pct[drawdown_pct < 0].sum()) / len(equity_curve)

        return float(pain)

    @staticmethod
    def calculate_mar_ratio(
        total_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calculate MAR Ratio (Return / Max Drawdown)

        Args:
            total_return: Total return (e.g., 0.5 = 50%)
            max_drawdown: Maximum drawdown (e.g., 0.2 = 20%)

        Returns:
            MAR ratio
        """
        if max_drawdown == 0:
            return 0.0

        mar = total_return / max_drawdown

        return float(mar)

    @staticmethod
    def calculate_tail_ratio(returns: pd.Series) -> float:
        """
        Calculate Tail Ratio

        95th percentile / abs(5th percentile)

        Args:
            returns: Series of returns

        Returns:
            Tail ratio
        """
        if len(returns) == 0:
            return 0.0

        upper = returns.quantile(0.95)
        lower = abs(returns.quantile(0.05))

        if lower == 0:
            return 0.0

        tail_ratio = upper / lower

        return float(tail_ratio)

    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """
        Calculate kurtosis (fat tails indicator)

        Args:
            returns: Series of returns

        Returns:
            Kurtosis value
        """
        if len(returns) < 4:
            return 0.0

        return float(stats.kurtosis(returns))

    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """
        Calculate skewness (asymmetry)

        Args:
            returns: Series of returns

        Returns:
            Skewness value
        """
        if len(returns) < 3:
            return 0.0

        return float(stats.skew(returns))

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate all advanced risk metrics

        Args:
            equity_curve: Series of equity values
            trades: List of trade dictionaries

        Returns:
            Dictionary with all metrics
        """
        if len(equity_curve) == 0:
            return {}

        returns = equity_curve.pct_change().dropna()

        metrics = {
            # Advanced risk metrics
            'cvar_95': self.calculate_cvar(returns, alpha=0.05),
            'cvar_99': self.calculate_cvar(returns, alpha=0.01),
            'omega_ratio': self.calculate_omega_ratio(returns),
            'ulcer_index': self.calculate_ulcer_index(equity_curve),
            'pain_index': self.calculate_pain_index(equity_curve),
            'tail_ratio': self.calculate_tail_ratio(returns),
            'kurtosis': self.calculate_kurtosis(returns),
            'skewness': self.calculate_skewness(returns),
        }

        # MAR ratio (if we have max drawdown)
        if trades:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(drawdown.min())

            metrics['mar_ratio'] = self.calculate_mar_ratio(total_return, max_dd)

        return metrics


class DynamicPositionSizer:
    """
    Dynamic position sizing based on market conditions

    Adjusts position size based on:
    - Volatility regime
    - Win rate
    - Consecutive wins/losses
    - Current drawdown
    """

    def __init__(
        self,
        base_position_size: float = 0.05,
        kelly_criterion: Optional[KellyCriterion] = None
    ):
        """
        Initialize dynamic position sizer

        Args:
            base_position_size: Base position size (fraction)
            kelly_criterion: Kelly criterion calculator
        """
        self.base_position_size = base_position_size
        self.kelly_criterion = kelly_criterion or KellyCriterion()

        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_drawdown = 0.0

    def calculate_position_size(
        self,
        trades: List[Dict],
        current_volatility: float,
        market_regime: str = "normal"
    ) -> float:
        """
        Calculate optimal position size

        Args:
            trades: Recent trades
            current_volatility: Current market volatility
            market_regime: Market regime (bull/bear/sideways/high_vol)

        Returns:
            Position size (fraction of capital)
        """
        # Start with Kelly-based size
        kelly_size = self.kelly_criterion.calculate_from_trades(trades)

        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier(market_regime)

        # Adjust for volatility
        vol_multiplier = self._get_volatility_multiplier(current_volatility)

        # Adjust for consecutive losses (reduce after losses)
        loss_multiplier = 1.0 - (min(self.consecutive_losses, 3) * 0.1)

        # Adjust for drawdown (reduce during drawdown)
        dd_multiplier = 1.0 - min(self.current_drawdown, 0.5)

        # Final position size
        position_size = (
            kelly_size *
            regime_multiplier *
            vol_multiplier *
            loss_multiplier *
            dd_multiplier
        )

        # Bounds
        position_size = np.clip(position_size, 0.01, 0.25)

        return float(position_size)

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get multiplier based on market regime"""
        multipliers = {
            'bull': 1.2,
            'bear': 0.5,
            'sideways': 0.8,
            'high_vol': 0.3,
            'normal': 1.0
        }
        return multipliers.get(regime.lower(), 1.0)

    def _get_volatility_multiplier(self, volatility: float) -> float:
        """Get multiplier based on volatility"""
        # Reduce position size when volatility is high
        if volatility < 0.01:  # Low vol
            return 1.2
        elif volatility < 0.02:  # Normal vol
            return 1.0
        elif volatility < 0.03:  # Elevated vol
            return 0.8
        else:  # High vol
            return 0.5

    def update_trade_result(self, pnl: float):
        """Update consecutive wins/losses"""
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def update_drawdown(self, current_drawdown: float):
        """Update current drawdown"""
        self.current_drawdown = abs(current_drawdown)
