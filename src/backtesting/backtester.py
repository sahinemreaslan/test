"""
Backtesting Engine

Simulates strategy execution with:
- Order execution (market orders)
- Position management
- Risk management (stop loss, take profit)
- Commission and slippage
- Performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class Position:
    """Trading position"""

    def __init__(
        self,
        side: OrderSide,
        entry_price: float,
        quantity: float,
        entry_time: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        margin_used: float = 0.0
    ):
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.margin_used = margin_used
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[pd.Timestamp] = None
        self.pnl: float = 0.0
        self.pnl_pct: float = 0.0
        self.liquidation_price: Optional[float] = None

        # Calculate liquidation price
        self._calculate_liquidation_price()

    def _calculate_liquidation_price(self):
        """Calculate liquidation price based on leverage and maintenance margin"""
        if self.leverage <= 1:
            self.liquidation_price = None  # No liquidation risk without leverage
            return

        # Simplified liquidation formula
        # Long: liquidation = entry * (1 - (1/leverage - maintenance_margin))
        # Short: liquidation = entry * (1 + (1/leverage - maintenance_margin))
        maintenance_margin = 0.05  # 5% default

        if self.side == OrderSide.BUY:
            # Long position
            self.liquidation_price = self.entry_price * (1 - (1/self.leverage - maintenance_margin))
        else:
            # Short position
            self.liquidation_price = self.entry_price * (1 + (1/self.leverage - maintenance_margin))

    def is_stop_loss_hit(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if self.stop_loss is None:
            return False

        if self.side == OrderSide.BUY:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

    def is_take_profit_hit(self, current_price: float) -> bool:
        """Check if take profit is hit"""
        if self.take_profit is None:
            return False

        if self.side == OrderSide.BUY:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def is_liquidated(self, current_price: float) -> bool:
        """Check if position is liquidated"""
        if self.liquidation_price is None:
            return False

        if self.side == OrderSide.BUY:
            return current_price <= self.liquidation_price
        else:
            return current_price >= self.liquidation_price

    def calculate_pnl(self, exit_price: float) -> Tuple[float, float]:
        """
        Calculate PnL

        Returns:
            Tuple of (absolute PnL, percentage PnL)
        """
        if self.side == OrderSide.BUY:
            pnl = (exit_price - self.entry_price) * self.quantity
            pnl_pct = (exit_price / self.entry_price) - 1
        else:
            pnl = (self.entry_price - exit_price) * self.quantity
            pnl_pct = (self.entry_price / exit_price) - 1

        return pnl, pnl_pct

    def close(self, exit_price: float, exit_time: pd.Timestamp):
        """Close position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl, self.pnl_pct = self.calculate_pnl(exit_price)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'side': self.side.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'duration': (self.exit_time - self.entry_time).total_seconds() / 3600 if self.exit_time else None,  # hours
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class Backtester:
    """Backtest trading strategy"""

    def __init__(self, config: Dict):
        """
        Initialize backtester

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})

        # Capital and position sizing
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.max_positions = self.backtest_config.get('max_positions', 1)

        # Costs
        self.commission = self.backtest_config.get('commission', 0.001)  # 0.1%
        self.slippage = self.backtest_config.get('slippage', 0.0005)    # 0.05%

        # Leverage settings
        self.leverage = self.backtest_config.get('leverage', 1)  # Default 1x (no leverage)
        self.max_leverage = self.backtest_config.get('max_leverage', 10)
        self.maintenance_margin = self.backtest_config.get('maintenance_margin', 0.05)
        self.initial_margin_ratio = self.backtest_config.get('initial_margin_ratio', 0.10)

        # Risk management
        self.max_drawdown_pct = self.backtest_config.get('max_drawdown_percent', 20) / 100

        # Tracking
        self.liquidations = 0  # Track liquidation events

        # State
        self.positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # Metrics calculator
        self.metrics_calculator = PerformanceMetrics()

    def reset(self):
        """Reset backtester state"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.liquidations = 0

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        params: Dict,
        verbose: bool = False
    ) -> Tuple[pd.Series, List[Dict]]:
        """
        Run backtest

        Args:
            data: OHLCV data with features
            signals: Trading signals (1 = buy, 0 = no action, -1 = sell)
            params: Strategy parameters (stop_loss_atr, take_profit_atr, position_size, etc.)
            verbose: Print progress

        Returns:
            Tuple of (equity curve Series, list of trade dicts)
        """
        logger.info("Running backtest...")
        self.reset()

        # Get ATR for stop loss/take profit calculation
        if 'atr' not in data.columns:
            logger.warning("ATR not found in data, using default values")
            data['atr'] = data['close'] * 0.02  # 2% of close as fallback

        position_size_pct = params.get('position_size', 0.05)  # Default 5%
        stop_loss_atr_mult = params.get('stop_loss_atr', 2.0)
        take_profit_atr_mult = params.get('take_profit_atr', 4.0)

        # Align signals with data
        signals = signals.reindex(data.index, fill_value=0)

        # Track peak equity for drawdown calculation
        peak_equity = self.initial_capital

        for timestamp, row in data.iterrows():
            if pd.isna(timestamp):
                continue

            current_price = row['close']
            current_atr = row.get('atr', current_price * 0.02)
            signal = signals.loc[timestamp] if timestamp in signals.index else 0

            # Check existing positions for stop loss / take profit
            self._check_exit_conditions(timestamp, row)

            # Calculate current equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append((timestamp, equity))

            # Update peak equity
            if equity > peak_equity:
                peak_equity = equity

            # Check drawdown limit
            current_drawdown = (peak_equity - equity) / peak_equity
            if current_drawdown >= self.max_drawdown_pct:
                if verbose:
                    logger.warning(f"Max drawdown reached at {timestamp}: {current_drawdown*100:.2f}%")
                # Close all positions
                self._close_all_positions(timestamp, current_price, reason="max_drawdown")
                continue

            # Process trading signal
            if signal > 0 and len(self.positions) < self.max_positions:
                # Buy signal
                self._open_position(
                    timestamp, row,
                    side=OrderSide.BUY,
                    position_size_pct=position_size_pct,
                    stop_loss_atr_mult=stop_loss_atr_mult,
                    take_profit_atr_mult=take_profit_atr_mult
                )

            elif signal < 0:
                # Sell signal - close all positions
                self._close_all_positions(timestamp, current_price, reason="sell_signal")

        # Close any remaining positions at the end
        if self.positions and len(data) > 0:
            last_timestamp = data.index[-1]
            last_price = data.iloc[-1]['close']
            self._close_all_positions(last_timestamp, last_price, reason="end_of_data")

        # Create equity curve series
        if self.equity_curve:
            equity_series = pd.Series(
                [eq for _, eq in self.equity_curve],
                index=[ts for ts, _ in self.equity_curve]
            )
        else:
            equity_series = pd.Series([self.initial_capital], index=[data.index[0]])

        # Convert trades to dicts
        trade_dicts = [trade.to_dict() for trade in self.closed_trades]

        logger.info(f"Backtest complete. Total trades: {len(trade_dicts)}")

        return equity_series, trade_dicts

    def _check_exit_conditions(self, timestamp: pd.Timestamp, row: pd.Series):
        """Check if any positions should be closed due to stop loss, take profit, or liquidation"""
        positions_to_close = []

        for position in self.positions:
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']

            # Check liquidation FIRST (highest priority)
            if position.is_liquidated(current_low if position.side == OrderSide.BUY else current_high):
                exit_price = position.liquidation_price
                positions_to_close.append((position, exit_price, 'liquidation'))
                self.liquidations += 1

            # Check stop loss
            elif position.is_stop_loss_hit(current_low if position.side == OrderSide.BUY else current_high):
                exit_price = position.stop_loss * (1 - self.slippage)  # Slippage
                positions_to_close.append((position, exit_price, 'stop_loss'))

            # Check take profit
            elif position.is_take_profit_hit(current_high if position.side == OrderSide.BUY else current_low):
                exit_price = position.take_profit * (1 - self.slippage)  # Slippage
                positions_to_close.append((position, exit_price, 'take_profit'))

        # Close positions
        for position, exit_price, reason in positions_to_close:
            self._close_position(position, timestamp, exit_price, reason)

    def _open_position(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        side: OrderSide,
        position_size_pct: float,
        stop_loss_atr_mult: float,
        take_profit_atr_mult: float
    ):
        """Open new position with leverage support"""
        entry_price = row['close'] * (1 + self.slippage)  # Apply slippage
        atr = row.get('atr', row['close'] * 0.02)

        # Calculate position size WITH LEVERAGE
        margin = self.current_capital * position_size_pct  # Actual capital to use as margin
        position_value = margin * self.leverage  # Leveraged position size

        # Commission is calculated on full position value (not just margin)
        commission_cost = position_value * self.commission

        # Quantity based on leveraged position value
        quantity = (position_value - commission_cost) / entry_price

        # Calculate stop loss and take profit
        if side == OrderSide.BUY:
            stop_loss = entry_price - (atr * stop_loss_atr_mult)
            take_profit = entry_price + (atr * take_profit_atr_mult)
        else:
            stop_loss = entry_price + (atr * stop_loss_atr_mult)
            take_profit = entry_price - (atr * take_profit_atr_mult)

        # Create position with leverage info
        position = Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.leverage,
            margin_used=margin
        )

        self.positions.append(position)

        # Deduct margin + commission from capital
        self.current_capital -= (margin + commission_cost)

    def _close_position(self, position: Position, timestamp: pd.Timestamp, exit_price: float, reason: str = 'signal'):
        """Close position and return margin"""
        # Calculate PnL
        position.close(exit_price, timestamp)

        # Commission on exit
        commission_cost = position.quantity * exit_price * self.commission

        # For liquidation, lose all margin
        if reason == 'liquidation':
            # Liquidation: lose all margin, no PnL return
            logger.warning(f"LIQUIDATION at {timestamp}: {position.side.value} position, "
                          f"entry={position.entry_price:.2f}, liq={position.liquidation_price:.2f}")
            # Capital already reduced by margin, don't add anything back
            pass
        else:
            # Normal exit: return margin + PnL - commission
            self.current_capital += position.margin_used + position.pnl - commission_cost

        # Move to closed trades
        self.closed_trades.append(position)
        self.positions.remove(position)

    def _close_all_positions(self, timestamp: pd.Timestamp, current_price: float, reason: str = "signal"):
        """Close all open positions"""
        positions_to_close = list(self.positions)  # Copy list

        for position in positions_to_close:
            exit_price = current_price * (1 - self.slippage)
            self._close_position(position, timestamp, exit_price, reason)

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        equity = self.current_capital

        # Add unrealized PnL from open positions
        for position in self.positions:
            unrealized_pnl, _ = position.calculate_pnl(current_price)
            equity += unrealized_pnl

        return equity

    def get_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        if not self.equity_curve:
            return {}

        equity_series = pd.Series(
            [eq for _, eq in self.equity_curve],
            index=[ts for ts, _ in self.equity_curve]
        )

        trade_dicts = [trade.to_dict() for trade in self.closed_trades]

        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            equity_series, trade_dicts
        )

        return metrics

    def print_results(self):
        """Print backtest results"""
        metrics = self.get_metrics()
        self.metrics_calculator.print_summary(metrics)

        # Print leverage and liquidation info
        if self.leverage > 1:
            logger.info(f"\n--- Leverage Info ---")
            logger.info(f"Leverage: {self.leverage}x")
            logger.info(f"Liquidations: {self.liquidations}")
