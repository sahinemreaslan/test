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
    """Trading position with advanced features"""

    def __init__(
        self,
        side: OrderSide,
        entry_price: float,
        quantity: float,
        entry_time: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        margin_used: float = 0.0,
        enable_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.02,
        enable_partial_exit: bool = False
    ):
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.original_quantity = quantity  # Track original quantity for partial exits
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.initial_stop_loss = stop_loss  # Keep original SL for reference
        self.take_profit = take_profit
        self.leverage = leverage
        self.margin_used = margin_used
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[pd.Timestamp] = None
        self.pnl: float = 0.0
        self.pnl_pct: float = 0.0
        self.liquidation_price: Optional[float] = None

        # Trailing stop
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.highest_price = entry_price if side == OrderSide.BUY else entry_price  # Track high for long, low for short
        self.lowest_price = entry_price

        # Partial exit tracking
        self.enable_partial_exit = enable_partial_exit
        self.partial_exits: List[Dict] = []  # Track partial exit levels
        self.is_partially_closed = False

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

    def update_trailing_stop(self, current_high: float, current_low: float):
        """
        Update trailing stop based on price movement

        Args:
            current_high: Current candle high
            current_low: Current candle low
        """
        if not self.enable_trailing_stop or self.stop_loss is None:
            return

        if self.side == OrderSide.BUY:
            # For long positions, track highest price
            if current_high > self.highest_price:
                self.highest_price = current_high
                # Update stop loss to trail below highest price
                new_stop = self.highest_price * (1 - self.trailing_stop_pct)
                # Only move stop loss up, never down
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop

        else:  # SHORT position
            # For short positions, track lowest price
            if current_low < self.lowest_price:
                self.lowest_price = current_low
                # Update stop loss to trail above lowest price
                new_stop = self.lowest_price * (1 + self.trailing_stop_pct)
                # Only move stop loss down, never up
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop

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

    def partial_close(self, exit_price: float, exit_time: pd.Timestamp, percentage: float) -> Tuple[float, float]:
        """
        Partially close position

        Args:
            exit_price: Exit price for this partial close
            exit_time: Exit timestamp
            percentage: Percentage of current quantity to close (0-1)

        Returns:
            Tuple of (quantity_closed, pnl_from_partial)
        """
        if percentage <= 0 or percentage > 1:
            raise ValueError(f"Invalid percentage: {percentage}")

        # Calculate quantity to close
        quantity_to_close = self.quantity * percentage

        # Calculate PnL for this partial exit
        if self.side == OrderSide.BUY:
            pnl = (exit_price - self.entry_price) * quantity_to_close
        else:
            pnl = (self.entry_price - exit_price) * quantity_to_close

        # Update position quantity
        self.quantity -= quantity_to_close

        # Track this partial exit
        self.partial_exits.append({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'quantity': quantity_to_close,
            'pnl': pnl
        })

        self.is_partially_closed = True

        return quantity_to_close, pnl

    def close(self, exit_price: float, exit_time: pd.Timestamp):
        """Close position (full or remaining quantity)"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl, self.pnl_pct = self.calculate_pnl(exit_price)

        # Add partial exit PnLs if any
        if self.partial_exits:
            partial_pnl = sum(pe['pnl'] for pe in self.partial_exits)
            self.pnl += partial_pnl

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

    def __init__(self, config: Dict, advanced_system=None):
        """
        Initialize backtester

        Args:
            config: Configuration dictionary
            advanced_system: Optional AdvancedTradingSystem for regime-based adjustments
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        self.advanced_system = advanced_system

        # Capital and position sizing
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.max_positions = self.backtest_config.get('max_positions', 1)

        # Costs
        self.commission = self.backtest_config.get('commission', 0.001)  # 0.1%
        self.slippage = self.backtest_config.get('slippage', 0.0005)    # 0.05%

        # Leverage settings
        self.base_leverage = self.backtest_config.get('leverage', 1)  # Default 1x (no leverage)
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

    def _get_volatility_multiplier(self, volatility: float) -> float:
        """
        Get leverage multiplier based on volatility

        Higher volatility = lower leverage for safety
        Lower volatility = can use higher leverage

        Args:
            volatility: Current volatility (daily std dev)

        Returns:
            Volatility multiplier for leverage (0.5 to 1.2)
        """
        if volatility < 0.01:  # Very low volatility
            return 1.2
        elif volatility < 0.02:  # Normal volatility
            return 1.0
        elif volatility < 0.03:  # Elevated volatility
            return 0.8
        elif volatility < 0.05:  # High volatility
            return 0.6
        else:  # Extreme volatility
            return 0.5

    def _calculate_trend_strength(self, data: pd.DataFrame, index: int, lookback: int = 50) -> float:
        """
        Calculate trend strength at given index

        Uses EMA slope and price position relative to EMA

        Args:
            data: DataFrame with OHLCV data
            index: Current index position
            lookback: Lookback period for EMA

        Returns:
            Trend strength: -1 to 1 (negative = downtrend, positive = uptrend)
        """
        if index < lookback + 10:
            return 0.0

        # Get price data
        prices = data['close'].iloc[max(0, index - lookback):index + 1]

        if len(prices) < 20:
            return 0.0

        # Calculate EMA
        ema_fast = prices.ewm(span=20).mean()
        ema_slow = prices.ewm(span=50).mean()

        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return 0.0

        # Current values
        current_price = prices.iloc[-1]
        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]

        # EMA alignment
        ema_alignment = (current_ema_fast - current_ema_slow) / current_ema_slow

        # Price position relative to EMAs
        price_position = (current_price - current_ema_fast) / current_ema_fast

        # EMA slope (rate of change)
        ema_slope = (ema_fast.iloc[-1] - ema_fast.iloc[-10]) / ema_fast.iloc[-10]

        # Combine factors
        trend_strength = (ema_alignment * 2 + price_position + ema_slope) / 4

        # Normalize to -1 to 1
        trend_strength = np.clip(trend_strength, -1.0, 1.0)

        return float(trend_strength)

    def _get_regime_adjusted_params(self, base_params: Dict, current_volatility: float = 0.02) -> Dict:
        """
        Get regime-adjusted parameters

        Args:
            base_params: Base strategy parameters
            current_volatility: Current market volatility

        Returns:
            Adjusted parameters based on current market regime and volatility
        """
        if self.advanced_system is None or not hasattr(self.advanced_system, 'get_regime_parameters'):
            return base_params

        # Get regime parameters
        regime_params = self.advanced_system.get_regime_parameters()

        # Apply multipliers
        adjusted_params = base_params.copy()
        adjusted_params['stop_loss_atr'] = base_params.get('stop_loss_atr', 2.0) * regime_params['stop_loss_multiplier']
        adjusted_params['take_profit_atr'] = base_params.get('take_profit_atr', 4.0) * regime_params['take_profit_multiplier']

        # Calculate leverage: base * regime_multiplier * volatility_multiplier
        regime_leverage = self.base_leverage * regime_params['leverage_multiplier']
        vol_multiplier = self._get_volatility_multiplier(current_volatility)
        adjusted_params['leverage'] = regime_leverage * vol_multiplier

        # Cap leverage at max
        adjusted_params['leverage'] = min(adjusted_params['leverage'], self.max_leverage)
        # Floor at 1x (no less than no leverage)
        adjusted_params['leverage'] = max(adjusted_params['leverage'], 1.0)

        return adjusted_params

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

        # Make a copy to avoid SettingWithCopyWarning
        data = data.copy()

        # Get ATR for stop loss/take profit calculation
        if 'atr' not in data.columns:
            logger.warning("ATR not found in data, using default values")
            data['atr'] = data['close'] * 0.02  # 2% of close as fallback

        # Calculate rolling volatility (20-period)
        data['volatility'] = data['close'].pct_change().rolling(20).std()

        # Get initial volatility for regime adjustment
        initial_volatility = data['volatility'].iloc[20] if len(data) > 20 else 0.02

        # Get regime-adjusted parameters
        adjusted_params = self._get_regime_adjusted_params(params, current_volatility=initial_volatility)

        position_size_pct = params.get('position_size', 0.05)  # Default 5%
        stop_loss_atr_mult = adjusted_params.get('stop_loss_atr', 2.0)
        take_profit_atr_mult = adjusted_params.get('take_profit_atr', 4.0)
        current_leverage = adjusted_params.get('leverage', self.base_leverage)

        # Align signals with data
        signals = signals.reindex(data.index, fill_value=0)

        # Track peak equity for drawdown calculation
        peak_equity = self.initial_capital

        # Track when to recalculate regime params (every N candles)
        recalc_interval = 100  # Recalculate every 100 candles
        candle_count = 0
        current_trend_strength = 0.0

        for i, (timestamp, row) in enumerate(data.iterrows()):
            if pd.isna(timestamp):
                continue

            current_price = row['close']
            current_atr = row.get('atr', current_price * 0.02)
            signal = signals.loc[timestamp] if timestamp in signals.index else 0

            # Periodically recalculate regime-adjusted params and trend strength
            candle_count += 1
            if candle_count % recalc_interval == 0:
                current_vol = row.get('volatility', 0.02)
                if not pd.isna(current_vol):
                    adjusted_params = self._get_regime_adjusted_params(params, current_volatility=current_vol)
                    stop_loss_atr_mult = adjusted_params.get('stop_loss_atr', 2.0)
                    take_profit_atr_mult = adjusted_params.get('take_profit_atr', 4.0)
                    current_leverage = adjusted_params.get('leverage', self.base_leverage)

                # Calculate trend strength for position holding filter
                current_trend_strength = self._calculate_trend_strength(data, i)

            # Check existing positions for stop loss / take profit
            # Pass trend strength to modify exit behavior
            self._check_exit_conditions_with_trend(timestamp, row, current_trend_strength)

            # Check for position scaling (pyramiding) if we have open positions
            if len(self.positions) > 0 and candle_count % 50 == 0:  # Check less frequently
                self._check_position_scaling(
                    timestamp,
                    row,
                    current_trend_strength,
                    stop_loss_atr_mult,
                    take_profit_atr_mult,
                    current_leverage
                )

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
            if signal > 0:  # BUY signal - want LONG position
                # Check if we have SHORT positions to close first
                short_positions = [p for p in self.positions if p.side == OrderSide.SELL]
                if short_positions:
                    # Close all SHORT positions before opening LONG
                    for position in short_positions:
                        exit_price = current_price * (1 - self.slippage)
                        self._close_position(position, timestamp, exit_price, reason="reverse_to_long")

                # Open LONG if we don't have one yet
                long_positions = [p for p in self.positions if p.side == OrderSide.BUY]
                if len(long_positions) < self.max_positions:
                    self._open_position(
                        timestamp, row,
                        side=OrderSide.BUY,
                        position_size_pct=position_size_pct,
                        stop_loss_atr_mult=stop_loss_atr_mult,
                        take_profit_atr_mult=take_profit_atr_mult,
                        leverage=current_leverage
                    )

            elif signal < 0:  # SELL signal - want SHORT position
                # Check if we have LONG positions to close first
                long_positions = [p for p in self.positions if p.side == OrderSide.BUY]
                if long_positions:
                    # Close all LONG positions before opening SHORT
                    for position in long_positions:
                        exit_price = current_price * (1 - self.slippage)
                        self._close_position(position, timestamp, exit_price, reason="reverse_to_short")

                # Open SHORT if we don't have one yet
                short_positions = [p for p in self.positions if p.side == OrderSide.SELL]
                if len(short_positions) < self.max_positions:
                    self._open_position(
                        timestamp, row,
                        side=OrderSide.SELL,
                        position_size_pct=position_size_pct,
                        stop_loss_atr_mult=stop_loss_atr_mult,
                        take_profit_atr_mult=take_profit_atr_mult,
                        leverage=current_leverage
                    )

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

    def _check_position_scaling(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        trend_strength: float = 0.0,
        stop_loss_atr_mult: float = 2.0,
        take_profit_atr_mult: float = 4.0,
        current_leverage: float = 1.0
    ) -> bool:
        """
        Check if we should add to existing position (pyramiding)

        Only add when:
        - Position is in profit
        - Strong trend in position direction
        - Haven't scaled too many times
        - NOT in extreme volatility (crash protection)
        - NOT in significant drawdown

        Returns:
            True if scaled, False otherwise
        """
        if not self.backtest_config.get('enable_position_scaling', False):
            return False

        if not self.positions or len(self.positions) == 0:
            return False

        position = self.positions[0]  # Assuming max_positions = 1

        # CRASH PROTECTION: Check extreme volatility
        current_vol = row.get('volatility', 0.02)
        extreme_vol_threshold = self.backtest_config.get('extreme_volatility_threshold', 0.05)
        if current_vol > extreme_vol_threshold:
            logger.debug(f"Position scaling disabled: Extreme volatility ({current_vol:.3f} > {extreme_vol_threshold})")
            return False

        # CRASH PROTECTION: Check current drawdown
        current_price = row['close']
        equity = self._calculate_equity(current_price)
        if hasattr(self, 'equity_curve') and self.equity_curve:
            peak_equity = max(eq for _, eq in self.equity_curve)
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_dd_for_scaling = self.backtest_config.get('max_drawdown_for_scaling', 0.10)

            if current_drawdown > max_dd_for_scaling:
                logger.debug(f"Position scaling disabled: In drawdown ({current_drawdown:.2%} > {max_dd_for_scaling:.2%})")
                return False

        # Check if already scaled maximum times
        max_scale_ins = self.backtest_config.get('max_scale_ins', 1)
        if hasattr(position, 'scale_in_count') and position.scale_in_count >= max_scale_ins:
            return False

        # Only scale in strong trends matching position direction
        if position.side == OrderSide.BUY and trend_strength < 0.5:
            return False
        if position.side == OrderSide.SELL and trend_strength > -0.5:
            return False

        # Check if position is in profit (at least 1 ATR)
        atr = row.get('atr', current_price * 0.02)
        profit = (current_price - position.entry_price) if position.side == OrderSide.BUY else (position.entry_price - current_price)

        if profit < atr:  # Not enough profit to scale
            return False

        # Add to position (smaller size than initial)
        scale_size_multiplier = self.backtest_config.get('scale_size_multiplier', 0.5)
        position_size_pct = 0.05 * scale_size_multiplier  # 50% of initial size

        # Open additional position (will be merged conceptually)
        self._open_position(
            timestamp,
            row,
            side=position.side,
            position_size_pct=position_size_pct,
            stop_loss_atr_mult=stop_loss_atr_mult,
            take_profit_atr_mult=take_profit_atr_mult,
            leverage=current_leverage
        )

        # Track scaling
        if not hasattr(position, 'scale_in_count'):
            position.scale_in_count = 0
        position.scale_in_count += 1

        logger.info(f"POSITION SCALING #{position.scale_in_count} at {timestamp}: "
                   f"{position.side.value}, price={current_price:.2f}, trend={trend_strength:.2f}")

        return True

    def _check_exit_conditions_with_trend(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        trend_strength: float = 0.0
    ):
        """
        Check exit conditions with trend filter, trailing stop, and partial exits

        In strong uptrends, hold positions longer by:
        - Ignoring early take profit signals
        - Using trailing stop to lock in profits

        Args:
            timestamp: Current timestamp
            row: Current OHLCV row
            trend_strength: Current trend strength (-1 to 1)
        """
        positions_to_close = []
        positions_to_partial_close = []

        for position in self.positions:
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']

            # Update trailing stop
            position.update_trailing_stop(current_high, current_low)

            # Check liquidation FIRST (always highest priority)
            if position.is_liquidated(current_low if position.side == OrderSide.BUY else current_high):
                exit_price = position.liquidation_price
                positions_to_close.append((position, exit_price, 'liquidation'))
                self.liquidations += 1
                continue

            # Partial exit logic (if enabled and in profit)
            if position.enable_partial_exit and not position.is_partially_closed:
                current_pnl_pct = (current_close - position.entry_price) / position.entry_price if position.side == OrderSide.BUY else (position.entry_price - current_close) / position.entry_price

                # Take partial profit at 50% of TP distance
                if position.take_profit is not None:
                    tp_distance = abs(position.take_profit - position.entry_price)
                    partial_target = position.entry_price + (tp_distance * 0.5) if position.side == OrderSide.BUY else position.entry_price - (tp_distance * 0.5)

                    if (position.side == OrderSide.BUY and current_high >= partial_target) or \
                       (position.side == OrderSide.SELL and current_low <= partial_target):
                        # Close 50% of position
                        positions_to_partial_close.append((position, partial_target, 0.5))

            # For LONG positions in strong UPTREND: hold longer
            if position.side == OrderSide.BUY and trend_strength > 0.3:
                # Only exit on stop loss in strong uptrends (let profits run)
                if position.is_stop_loss_hit(current_low):
                    exit_price = position.stop_loss * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'trailing_stop' if position.enable_trailing_stop else 'stop_loss'))

                # Ignore normal take profit in strong trends - let it run!
                # Take profit only if extreme profit (2x the normal TP)
                elif position.take_profit is not None:
                    extreme_tp = position.entry_price + (position.take_profit - position.entry_price) * 2
                    if current_high >= extreme_tp:
                        exit_price = extreme_tp * (1 - self.slippage)
                        positions_to_close.append((position, exit_price, 'take_profit_extreme'))

            # For LONG positions in DOWNTREND: exit quickly
            elif position.side == OrderSide.BUY and trend_strength < -0.3:
                # Exit on stop loss or take profit (normal behavior)
                if position.is_stop_loss_hit(current_low):
                    exit_price = position.stop_loss * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'trailing_stop' if position.enable_trailing_stop else 'stop_loss'))
                elif position.is_take_profit_hit(current_high):
                    # Take any profit in downtrends
                    exit_price = position.take_profit * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'take_profit'))

            # For SHORT positions in strong DOWNTREND: hold longer
            elif position.side == OrderSide.SELL and trend_strength < -0.3:
                # Only exit on stop loss in strong downtrends (let profits run)
                if position.is_stop_loss_hit(current_high):
                    exit_price = position.stop_loss * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'trailing_stop' if position.enable_trailing_stop else 'stop_loss'))

                # Ignore normal take profit in strong trends - let it run!
                # Take profit only if extreme profit (2x the normal TP)
                elif position.take_profit is not None:
                    extreme_tp = position.entry_price - (position.entry_price - position.take_profit) * 2
                    if current_low <= extreme_tp:
                        exit_price = extreme_tp * (1 + self.slippage)  # Positive slippage for SHORT
                        positions_to_close.append((position, exit_price, 'take_profit_extreme'))

            # For SHORT positions in UPTREND: exit quickly
            elif position.side == OrderSide.SELL and trend_strength > 0.3:
                # Exit on stop loss or take profit (normal behavior)
                if position.is_stop_loss_hit(current_high):
                    exit_price = position.stop_loss * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'trailing_stop' if position.enable_trailing_stop else 'stop_loss'))
                elif position.is_take_profit_hit(current_low):
                    # Take any profit in uptrends
                    exit_price = position.take_profit * (1 + self.slippage)  # Positive slippage for SHORT
                    positions_to_close.append((position, exit_price, 'take_profit'))

            # Normal conditions (weak trend or sideways)
            else:
                if position.is_stop_loss_hit(current_low if position.side == OrderSide.BUY else current_high):
                    exit_price = position.stop_loss * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'trailing_stop' if position.enable_trailing_stop else 'stop_loss'))
                elif position.is_take_profit_hit(current_high if position.side == OrderSide.BUY else current_low):
                    exit_price = position.take_profit * (1 - self.slippage)
                    positions_to_close.append((position, exit_price, 'take_profit'))

        # Execute partial closes first
        for position, exit_price, percentage in positions_to_partial_close:
            self._partial_close_position(position, timestamp, exit_price, percentage)

        # Then execute full closes
        for position, exit_price, reason in positions_to_close:
            self._close_position(position, timestamp, exit_price, reason)

    def _open_position(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        side: OrderSide,
        position_size_pct: float,
        stop_loss_atr_mult: float,
        take_profit_atr_mult: float,
        leverage: float = None
    ):
        """Open new position with leverage support"""
        # Use provided leverage or default to base leverage
        if leverage is None:
            leverage = self.base_leverage

        entry_price = row['close'] * (1 + self.slippage)  # Apply slippage
        atr = row.get('atr', row['close'] * 0.02)

        # Calculate position size WITH LEVERAGE
        margin = self.current_capital * position_size_pct  # Actual capital to use as margin
        position_value = margin * leverage  # Leveraged position size

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

        # Get advanced features settings from config
        enable_trailing_stop = self.backtest_config.get('enable_trailing_stop', True)
        trailing_stop_pct = self.backtest_config.get('trailing_stop_pct', 0.02)  # 2% default
        enable_partial_exit = self.backtest_config.get('enable_partial_exit', True)

        # Create position with leverage info and advanced features
        position = Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            margin_used=margin,
            enable_trailing_stop=enable_trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            enable_partial_exit=enable_partial_exit
        )

        self.positions.append(position)

        # Deduct margin + commission from capital
        self.current_capital -= (margin + commission_cost)

    def _partial_close_position(
        self,
        position: Position,
        timestamp: pd.Timestamp,
        exit_price: float,
        percentage: float
    ):
        """
        Partially close a position

        Args:
            position: Position to partially close
            timestamp: Current timestamp
            exit_price: Exit price
            percentage: Percentage to close (0-1)
        """
        # Execute partial close
        quantity_closed, pnl = position.partial_close(exit_price, timestamp, percentage)

        # Commission on partial exit
        commission_cost = quantity_closed * exit_price * self.commission

        # Return proportional margin + PnL - commission
        margin_returned = position.margin_used * percentage
        self.current_capital += margin_returned + pnl - commission_cost

        # Update position's margin used
        position.margin_used *= (1 - percentage)

        logger.debug(f"PARTIAL EXIT ({percentage*100:.0f}%) at {timestamp}: "
                    f"{position.side.value} position, price={exit_price:.2f}, PnL={pnl:.2f}")

    def _close_position(self, position: Position, timestamp: pd.Timestamp, exit_price: float, reason: str = 'signal'):
        """Close position and return margin"""
        # Calculate PnL
        position.close(exit_price, timestamp)

        # Commission on exit (on remaining quantity)
        commission_cost = position.quantity * exit_price * self.commission

        # For liquidation, lose all margin
        if reason == 'liquidation':
            # Liquidation: lose all margin, no PnL return
            logger.warning(f"LIQUIDATION at {timestamp}: {position.side.value} position, "
                          f"entry={position.entry_price:.2f}, liq={position.liquidation_price:.2f}")
            # Capital already reduced by margin, don't add anything back
            pass
        else:
            # Normal exit: return remaining margin + PnL - commission
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
