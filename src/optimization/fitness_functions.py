"""
Fitness Functions for Genetic Algorithm Optimization

Evaluates strategy performance using various metrics:
- Sharpe Ratio
- Calmar Ratio
- Sortino Ratio
- Profit Factor
- Win Rate
- Maximum Drawdown
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Evaluate trading strategy fitness using various metrics"""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize fitness evaluator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve"""
        return equity_curve.pct_change().fillna(0)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (252 for daily, adjust for other timeframes)

        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total std dev)

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio (annualized)
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)

        return float(sortino)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            equity_curve: Series of equity values

        Returns:
            Maximum drawdown as a fraction (0.2 = 20% drawdown)
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Return maximum drawdown (most negative value)
        max_dd = drawdown.min()

        return float(abs(max_dd))

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio (annualized return / maximum drawdown)

        Args:
            returns: Series of returns
            equity_curve: Series of equity values
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        max_dd = self.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        # Annualized return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1

        calmar = annualized_return / max_dd

        return float(calmar)

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate Profit Factor (gross profit / gross loss)

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in trades if t['pnl'] < 0]

        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0

        if gross_loss == 0:
            return float(gross_profit) if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss

        return float(profit_factor)

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate win rate

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Win rate (fraction of winning trades)
        """
        if not trades:
            return 0.0

        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning_trades / len(trades)

        return float(win_rate)

    def calculate_average_win_loss_ratio(self, trades: List[Dict]) -> float:
        """
        Calculate average win to loss ratio

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Average win / average loss ratio
        """
        if not trades:
            return 0.0

        winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in trades if t['pnl'] < 0]

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = np.mean(winning_trades)
        avg_loss = abs(np.mean(losing_trades))

        if avg_loss == 0:
            return 0.0

        return float(avg_win / avg_loss)

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """
        Calculate trade expectancy (average PnL per trade)

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Expectancy
        """
        if not trades:
            return 0.0

        return float(np.mean([t['pnl'] for t in trades]))

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        periods_per_year: int = 105120  # For 5-minute bars: 365*24*12
    ) -> Dict[str, float]:
        """
        Calculate all fitness metrics

        Args:
            equity_curve: Series of equity values over time
            trades: List of trade dictionaries
            periods_per_year: Number of periods per year (adjust based on timeframe)

        Returns:
            Dictionary with all metrics
        """
        if len(equity_curve) == 0:
            return self._empty_metrics()

        returns = self.calculate_returns(equity_curve)

        metrics = {
            'total_return': float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.calculate_sortino_ratio(returns, periods_per_year),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'calmar_ratio': self.calculate_calmar_ratio(returns, equity_curve, periods_per_year),
            'num_trades': len(trades),
            'win_rate': self.calculate_win_rate(trades),
            'profit_factor': self.calculate_profit_factor(trades),
            'avg_win_loss_ratio': self.calculate_average_win_loss_ratio(trades),
            'expectancy': self.calculate_expectancy(trades),
            'final_equity': float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else 0.0
        }

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win_loss_ratio': 0.0,
            'expectancy': 0.0,
            'final_equity': 0.0
        }

    def calculate_fitness_score(
        self,
        metrics: Dict[str, float],
        optimization_metric: str = 'sharpe_ratio',
        min_trades: int = 30
    ) -> float:
        """
        Calculate overall fitness score

        Args:
            metrics: Dictionary with all metrics
            optimization_metric: Primary metric to optimize
            min_trades: Minimum number of trades required

        Returns:
            Fitness score (higher is better)
        """
        # Penalty for insufficient trades
        if metrics['num_trades'] < min_trades:
            penalty = metrics['num_trades'] / min_trades
        else:
            penalty = 1.0

        # Get primary metric value
        primary_score = metrics.get(optimization_metric, 0.0)

        # Additional penalties/bonuses
        # Penalty for excessive drawdown
        if metrics['max_drawdown'] > 0.5:  # More than 50% drawdown
            drawdown_penalty = 0.5
        else:
            drawdown_penalty = 1.0

        # Bonus for positive profit factor
        if metrics['profit_factor'] > 1.0:
            profit_factor_bonus = 1.0 + (metrics['profit_factor'] - 1.0) * 0.1
        else:
            profit_factor_bonus = 0.5

        # Calculate final fitness
        fitness = primary_score * penalty * drawdown_penalty * profit_factor_bonus

        return float(fitness)
