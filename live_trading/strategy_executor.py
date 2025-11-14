"""
Strategy Executor
Converts strategy signals into actual trades
"""

import logging
import sys
import os
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path to import our strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced.integrated_system import AdvancedTradingSystem
from src.features.feature_engineering import FeatureEngineer
from src.data.timeframe_converter import TimeframeConverter

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """Executes trading strategy and generates signals"""

    def __init__(self, config: Dict):
        """
        Initialize strategy executor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_eng = FeatureEngineer(config)

        # Advanced trading system
        self.advanced_system = AdvancedTradingSystem(config)
        self.trained = False

        # Trading state
        self.last_signal = 0
        self.last_signal_time = None

    def train_strategy(self, historical_data: pd.DataFrame):
        """
        Train the strategy on historical data

        Args:
            historical_data: DataFrame with OHLCV data (15m timeframe)
        """
        logger.info("🎓 Training strategy on historical data...")

        try:
            # Prepare data - ensure datetime index
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                if 'timestamp' in historical_data.columns:
                    historical_data = historical_data.set_index('timestamp')
                elif 'date' in historical_data.columns:
                    historical_data = historical_data.set_index('date')

            # Convert to multiple timeframes
            logger.info("Converting to multiple timeframes...")
            tf_converter = TimeframeConverter(historical_data)

            # Get timeframes from config
            timeframes = self.config.get('timeframes', {}).get('all', [
                '3M', '1M', '1W', '1D', '12h', '8h', '4h', '2h', '1h', '30m', '15m'
            ])
            all_timeframes_raw = tf_converter.convert_all_timeframes(timeframes)

            # Process each timeframe (calculate indicators)
            logger.info("Processing indicators for each timeframe...")
            all_timeframes = {}
            for tf, tf_df in all_timeframes_raw.items():
                all_timeframes[tf] = self.feature_eng.process_single_timeframe(tf_df, tf)

            # Create multi-timeframe feature matrix
            logger.info("Creating multi-timeframe features...")
            reference_tf = self.config.get('timeframes', {}).get('signal', '15m')
            df = self.feature_eng.create_multi_timeframe_features(all_timeframes, reference_tf)

            # Prepare ML dataset (features and target)
            logger.info("Preparing ML dataset...")
            features, target = self.feature_eng.prepare_ml_dataset(
                df,
                target_method='forward_returns',
                target_horizon=1,
                target_threshold=0.001
            )

            logger.info(f"Features created: {features.shape}")
            logger.info(f"Target distribution: {target.value_counts().to_dict()}")

            # Train advanced system (pass original OHLCV data for regime detection)
            logger.info("Training advanced system...")
            reference_ohlcv = all_timeframes[reference_tf]  # Original OHLCV with indicators
            self.advanced_system.train(reference_ohlcv, features, target, verbose=True)

            self.trained = True
            logger.info("✅ Strategy training complete!")

        except Exception as e:
            logger.error(f"❌ Error training strategy: {e}", exc_info=True)
            raise

    def generate_signal(self, current_data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Generate trading signal from current market data

        Args:
            current_data: Recent OHLCV data (at least 300 candles of 15m)

        Returns:
            Tuple of (signal, metadata)
            signal: 1 = BUY, 0 = HOLD, -1 = SELL
            metadata: Additional info (confidence, regime, etc.)
        """
        if not self.trained:
            logger.error("❌ Strategy not trained! Call train_strategy() first")
            return 0, {}

        try:
            # Prepare data - ensure datetime index
            if not isinstance(current_data.index, pd.DatetimeIndex):
                if 'timestamp' in current_data.columns:
                    current_data = current_data.set_index('timestamp')
                elif 'date' in current_data.columns:
                    current_data = current_data.set_index('date')

            # Convert to multiple timeframes
            tf_converter = TimeframeConverter(current_data)

            # Get timeframes from config
            timeframes = self.config.get('timeframes', {}).get('all', [
                '3M', '1M', '1W', '1D', '12h', '8h', '4h', '2h', '1h', '30m', '15m'
            ])
            all_timeframes_raw = tf_converter.convert_all_timeframes(timeframes)

            # Process each timeframe (calculate indicators)
            all_timeframes = {}
            for tf, tf_df in all_timeframes_raw.items():
                all_timeframes[tf] = self.feature_eng.process_single_timeframe(tf_df, tf)

            # Create multi-timeframe feature matrix
            reference_tf = self.config.get('timeframes', {}).get('signal', '15m')
            df = self.feature_eng.create_multi_timeframe_features(all_timeframes, reference_tf)

            # Prepare ML dataset (features and target)
            features, target = self.feature_eng.prepare_ml_dataset(
                df,
                target_method='forward_returns',
                target_horizon=1,
                target_threshold=0.001
            )

            # Generate signals
            signals = self.advanced_system.generate_signals(df, features)

            # Get last signal
            signal = int(signals.iloc[-1])

            # Get regime info
            regime_params = self.advanced_system.get_regime_parameters()

            # Get ensemble probability (confidence)
            ensemble_proba = self.advanced_system.ensemble_model.predict_proba(features)
            confidence = float(ensemble_proba.iloc[-1])

            metadata = {
                'signal': signal,
                'confidence': confidence,
                'regime': self.advanced_system.regime_detector.get_regime_name(
                    self.advanced_system.current_regime
                ),
                'regime_params': regime_params,
                'timestamp': datetime.now().isoformat(),
                'close_price': float(current_data['close'].iloc[-1])
            }

            # Update last signal
            self.last_signal = signal
            self.last_signal_time = datetime.now()

            logger.info(f"📊 Signal: {signal} | Confidence: {confidence:.2f} | "
                       f"Regime: {metadata['regime']}")

            return signal, metadata

        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}", exc_info=True)
            return 0, {}

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        leverage: int,
        regime_params: Dict
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size

        Args:
            balance: Available balance in USDT
            price: Current BTC price
            leverage: Leverage to use
            regime_params: Regime-specific parameters

        Returns:
            Tuple of (quantity_btc, position_value_usdt)
        """
        # Base position size from config
        base_position_pct = self.config.get('trading', {}).get('position_size_pct', 0.05)

        # Apply regime multiplier
        regime_multiplier = regime_params.get('position_size_multiplier', 1.0)
        position_pct = base_position_pct * regime_multiplier

        # Cap at maximum
        max_position_pct = self.config.get('trading', {}).get('max_position_pct', 0.25)
        position_pct = min(position_pct, max_position_pct)

        # Calculate position value
        position_value = balance * position_pct * leverage

        # Calculate quantity in BTC
        quantity_btc = position_value / price

        logger.info(f"💰 Position size: {position_pct*100:.1f}% = {position_value:.2f} USDT "
                   f"= {quantity_btc:.6f} BTC (Leverage: {leverage}x)")

        return quantity_btc, position_value

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        side: str,
        atr: float,
        regime_params: Dict
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices

        Args:
            entry_price: Position entry price
            side: LONG or SHORT
            atr: Current ATR value
            regime_params: Regime parameters

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Get multipliers from config and regime
        base_sl_mult = self.config.get('trading', {}).get('stop_loss_atr_mult', 2.0)
        base_tp_mult = self.config.get('trading', {}).get('take_profit_atr_mult', 4.0)

        # Apply regime multipliers
        sl_mult = base_sl_mult * regime_params.get('stop_loss_multiplier', 1.0)
        tp_mult = base_tp_mult * regime_params.get('take_profit_multiplier', 1.0)

        if side == 'LONG':
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
        else:  # SHORT
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)

        logger.info(f"🎯 SL: {stop_loss:.2f} | TP: {take_profit:.2f} "
                   f"(SL: {sl_mult:.1f}x ATR, TP: {tp_mult:.1f}x ATR)")

        return stop_loss, take_profit

    def get_regime_info(self) -> Dict:
        """Get current market regime information"""
        if not self.trained:
            return {}

        regime_params = self.advanced_system.get_regime_parameters()

        return {
            'regime_name': self.advanced_system.regime_detector.get_regime_name(
                self.advanced_system.current_regime
            ),
            'parameters': regime_params
        }
