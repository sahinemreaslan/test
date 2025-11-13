"""
Performance Metrics Calculator

Comprehensive performance analysis including:
- Return metrics
- Risk metrics
- Risk-adjusted metrics
- Trade statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        benchmark: pd.Series = None,
        periods_per_year: int = 105120
    ) -> Dict:
        """
        Calculate all performance metrics

        Args:
            equity_curve: Equity curve over time
            trades: List of trade dictionaries
            benchmark: Optional benchmark returns
            periods_per_year: Number of periods per year

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Basic metrics
        metrics.update(self._calculate_return_metrics(equity_curve))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_curve, periods_per_year))

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(
            equity_curve, periods_per_year
        ))

        # Trade statistics
        if trades:
            metrics.update(self._calculate_trade_statistics(trades))

        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics(equity_curve))

        # Time-based metrics
        metrics.update(self._calculate_time_metrics(equity_curve))

        # Benchmark comparison (if provided)
        if benchmark is not None:
            metrics.update(self._calculate_benchmark_metrics(equity_curve, benchmark))

        return metrics

    def _calculate_return_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate return-based metrics"""
        if len(equity_curve) == 0:
            return {}

        initial_equity = equity_curve.iloc[0]
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity / initial_equity) - 1

        returns = equity_curve.pct_change().fillna(0)

        return {
            'initial_equity': float(initial_equity),
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'mean_return': float(returns.mean()),
            'median_return': float(returns.median()),
            'positive_returns_pct': float((returns > 0).sum() / len(returns) * 100)
        }

    def _calculate_risk_metrics(
        self,
        equity_curve: pd.Series,
        periods_per_year: int
    ) -> Dict:
        """Calculate risk metrics"""
        returns = equity_curve.pct_change().fillna(0)

        return {
            'volatility': float(returns.std()),
            'annualized_volatility': float(returns.std() * np.sqrt(periods_per_year)),
            'downside_volatility': float(returns[returns < 0].std()),
            'var_95': float(returns.quantile(0.05)),  # Value at Risk (95%)
            'cvar_95': float(returns[returns <= returns.quantile(0.05)].mean())  # Conditional VaR
        }

    def _calculate_risk_adjusted_metrics(
        self,
        equity_curve: pd.Series,
        periods_per_year: int
    ) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        returns = equity_curve.pct_change().fillna(0)

        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = (excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
                 if returns.std() > 0 else 0)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
                  if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)

        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(equity_curve)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'information_ratio': float(sharpe)  # Simplified (assuming benchmark = risk-free rate)
        }

    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate trade statistics"""
        if not trades:
            return {}

        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': float(len(winning_trades) / len(trades) * 100),
            'avg_win': float(np.mean(winning_trades)) if winning_trades else 0.0,
            'avg_loss': float(np.mean(losing_trades)) if losing_trades else 0.0,
            'largest_win': float(max(winning_trades)) if winning_trades else 0.0,
            'largest_loss': float(min(losing_trades)) if losing_trades else 0.0,
            'avg_pnl': float(np.mean(pnls)),
            'total_pnl': float(sum(pnls)),
            'profit_factor': (float(sum(winning_trades) / abs(sum(losing_trades)))
                            if losing_trades and sum(losing_trades) != 0 else 0.0),
            'avg_win_loss_ratio': (float(np.mean(winning_trades) / abs(np.mean(losing_trades)))
                                  if winning_trades and losing_trades else 0.0),
            'expectancy': float(np.mean(pnls))
        }

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return float(abs(drawdown.min()))

    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown-related metrics"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Find all drawdown periods
        in_drawdown = drawdown < 0
        drawdown_changes = in_drawdown.astype(int).diff()

        drawdown_starts = drawdown_changes[drawdown_changes == 1].index
        drawdown_ends = drawdown_changes[drawdown_changes == -1].index

        drawdown_durations = []
        if len(drawdown_starts) > 0:
            for start in drawdown_starts:
                # Find corresponding end
                ends_after_start = drawdown_ends[drawdown_ends > start]
                if len(ends_after_start) > 0:
                    end = ends_after_start[0]
                    duration = (end - start).total_seconds() / 3600  # Hours
                    drawdown_durations.append(duration)

        return {
            'max_drawdown': float(abs(drawdown.min())),
            'max_drawdown_pct': float(abs(drawdown.min()) * 100),
            'avg_drawdown': float(abs(drawdown[drawdown < 0].mean())) if (drawdown < 0).any() else 0.0,
            'num_drawdown_periods': len(drawdown_starts),
            'avg_drawdown_duration_hours': float(np.mean(drawdown_durations)) if drawdown_durations else 0.0,
            'max_drawdown_duration_hours': float(max(drawdown_durations)) if drawdown_durations else 0.0
        }

    def _calculate_time_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate time-based metrics"""
        duration = equity_curve.index[-1] - equity_curve.index[0]
        duration_days = duration.total_seconds() / 86400

        returns = equity_curve.pct_change().fillna(0)

        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for ret in returns:
            if ret > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif ret < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        return {
            'start_date': str(equity_curve.index[0]),
            'end_date': str(equity_curve.index[-1]),
            'duration_days': float(duration_days),
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak
        }

    def _calculate_benchmark_metrics(
        self,
        equity_curve: pd.Series,
        benchmark: pd.Series
    ) -> Dict:
        """Calculate metrics relative to benchmark"""
        strategy_returns = equity_curve.pct_change().fillna(0)
        benchmark_returns = benchmark.pct_change().fillna(0)

        # Align series
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')

        # Calculate excess returns
        excess_returns = aligned_strategy - aligned_benchmark

        # Calculate alpha (simplified)
        alpha = excess_returns.mean()

        # Calculate beta
        covariance = np.cov(aligned_strategy, aligned_benchmark)[0][1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'excess_return': float(excess_returns.sum()),
            'tracking_error': float(excess_returns.std())
        }

    def print_summary(self, metrics: Dict):
        """Print formatted metrics summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        # Returns
        print("\n--- Returns ---")
        print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")

        # Risk-Adjusted
        print("\n--- Risk-Adjusted Metrics ---")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")

        # Drawdown
        print("\n--- Drawdown ---")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Avg Drawdown: {metrics.get('avg_drawdown', 0)*100:.2f}%")

        # Trades
        if 'total_trades' in metrics:
            print("\n--- Trade Statistics ---")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.3f}")
            print(f"Avg Win/Loss Ratio: {metrics.get('avg_win_loss_ratio', 0):.3f}")
            print(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")

        print("\n" + "="*60)
