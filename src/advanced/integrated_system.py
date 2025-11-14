"""
Integrated Advanced Trading System

Combines all research-grade components:
1. Market Regime Detection (HMM)
2. Ensemble Models (XGB + LGB + CAT)
3. Attention Mechanisms
4. LSTM/Transformer Models
5. Reinforcement Learning (PPO)
6. Kelly Criterion & Advanced Risk Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .market_regime import MarketRegimeDetector
from .ensemble_models import EnsemblePredictor
from .risk_management import KellyCriterion, AdvancedRiskMetrics, DynamicPositionSizer

logger = logging.getLogger(__name__)


class AdvancedTradingSystem:
    """
    Research-grade trading system integrating all advanced components
    """

    def __init__(self, config: Dict):
        """
        Initialize advanced trading system

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Components
        self.regime_detector = MarketRegimeDetector(n_regimes=4)
        self.ensemble_model = EnsemblePredictor(config)
        self.kelly_criterion = KellyCriterion(kelly_fraction=0.5)
        self.risk_metrics = AdvancedRiskMetrics()
        self.position_sizer = DynamicPositionSizer()

        # Optional deep learning components (require PyTorch)
        self.use_deep_learning = config.get('advanced', {}).get('use_deep_learning', False)
        self.lstm_model = None
        self.transformer_model = None
        self.attention_model = None

        # RL component
        self.use_rl = config.get('advanced', {}).get('use_rl', False)
        self.rl_agent = None

        # State
        self.current_regime = None
        self.trained = False

    def train(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series,
        verbose: bool = True
    ):
        """
        Train all components

        Args:
            df: OHLCV DataFrame
            features: Feature DataFrame
            target: Target Series
            verbose: Print progress
        """
        logger.info("="*70)
        logger.info("TRAINING ADVANCED TRADING SYSTEM")
        logger.info("="*70)

        # 1. Train Market Regime Detector
        logger.info("\n[1/4] Training Market Regime Detector (HMM)...")
        self.regime_detector.fit(df, verbose=verbose)

        # 2. Train Ensemble Model
        logger.info("\n[2/4] Training Ensemble Models (XGB + LGB + CAT)...")
        ensemble_metrics = self.ensemble_model.train(
            features,
            target,
            validation_split=0.2,
            verbose=verbose
        )

        # 3. Train Deep Learning Models (if enabled)
        if self.use_deep_learning:
            logger.info("\n[3/4] Training Deep Learning Models...")
            self._train_deep_learning_models(features, target, verbose)
        else:
            logger.info("\n[3/4] Skipping Deep Learning (disabled)")

        # 4. Train RL Agent (if enabled)
        if self.use_rl:
            logger.info("\n[4/4] Training Reinforcement Learning Agent...")
            self._train_rl_agent(df, features, verbose)
        else:
            logger.info("\n[4/4] Skipping RL (disabled)")

        self.trained = True

        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)

        return ensemble_metrics

    def _train_deep_learning_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        verbose: bool
    ):
        """Train deep learning models"""
        try:
            from .attention_models import TimeframeAttentionModel, FeatureAttentionModel
            from .sequence_models import LSTMSequenceModel, TransformerSequenceModel

            # TODO: Implement training pipeline for DL models
            # This requires sequence preparation and PyTorch training loop
            logger.info("Deep learning models training not yet implemented in this version")

        except ImportError as e:
            logger.warning(f"Could not import deep learning modules: {e}")

    def _train_rl_agent(self, df: pd.DataFrame, features: pd.DataFrame, verbose: bool):
        """Train RL agent"""
        try:
            from .rl_agent import TradingRLAgent

            self.rl_agent = TradingRLAgent(self.config)
            self.rl_agent.train(
                df=df,
                features=features,
                total_timesteps=100000,
                verbose=verbose
            )

        except ImportError as e:
            logger.warning(f"Could not import RL modules: {e}")

    def generate_signals(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.Series:
        """
        Generate trading signals using all components

        Args:
            df: OHLCV DataFrame
            features: Feature DataFrame

        Returns:
            Series with signals (1 = buy, 0 = hold, -1 = sell)
        """
        if not self.trained:
            raise ValueError("System not trained. Call train() first.")

        logger.info("Generating signals with advanced system...")

        # 1. Detect current market regime
        regimes = self.regime_detector.predict(df)
        self.current_regime = regimes.iloc[-1]

        logger.info(f"Current regime: {self.regime_detector.get_regime_name(self.current_regime)}")

        # 2. Get ensemble predictions
        ensemble_proba = self.ensemble_model.predict_proba(features)

        # 3. Adjust confidence threshold based on regime
        regime_params = self.regime_detector.get_regime_strategy_params(self.current_regime)
        confidence_threshold = regime_params['signal_threshold']

        # 4. Generate signals
        signals = pd.Series(0, index=features.index)

        # Buy signals
        buy_mask = ensemble_proba >= confidence_threshold
        signals[buy_mask] = 1

        # Sell signals (low confidence or regime change)
        sell_mask = ensemble_proba < (confidence_threshold * 0.7)
        signals[sell_mask] = -1

        logger.info(f"Generated {(signals == 1).sum()} buy signals, "
                   f"{(signals == -1).sum()} sell signals")

        return signals

    def calculate_position_size(
        self,
        df: pd.DataFrame,
        trades: List[Dict]
    ) -> float:
        """
        Calculate optimal position size using Kelly + regime + risk

        Args:
            df: OHLCV DataFrame
            trades: Recent trades

        Returns:
            Position size (fraction of capital)
        """
        # Current volatility
        if len(df) > 20:
            current_volatility = df['close'].pct_change().tail(20).std()
        else:
            current_volatility = 0.02

        # Current regime name
        regime_name = self.regime_detector.get_regime_name(self.current_regime)

        # Calculate dynamic position size
        base_position_size = self.position_sizer.calculate_position_size(
            trades=trades,
            current_volatility=current_volatility,
            market_regime=regime_name
        )

        # Apply regime-specific position size multiplier
        regime_params = self.regime_detector.get_regime_strategy_params(self.current_regime)
        position_size = base_position_size * regime_params['position_size_multiplier']

        # Cap at reasonable maximum
        position_size = min(position_size, 0.25)  # Max 25% of capital

        return position_size

    def get_regime_parameters(self) -> Dict[str, float]:
        """
        Get current regime-based strategy parameters

        Returns:
            Dictionary with regime-specific multipliers:
            - position_size_multiplier
            - stop_loss_multiplier
            - take_profit_multiplier
            - signal_threshold
            - leverage_multiplier
        """
        if self.current_regime is None:
            # Default parameters if no regime detected yet
            return {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'signal_threshold': 0.6,
                'leverage_multiplier': 1.0
            }

        return self.regime_detector.get_regime_strategy_params(self.current_regime)

    def get_advanced_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate all advanced risk metrics

        Args:
            equity_curve: Equity curve
            trades: List of trades

        Returns:
            Dictionary with all metrics
        """
        return self.risk_metrics.calculate_all_metrics(equity_curve, trades)

    def get_system_summary(self) -> Dict:
        """Get summary of system components and status"""
        # Get regime confidence (0-1) based on how long we've been in this regime
        regime_confidence = 1.0  # Default to high confidence

        return {
            'trained': self.trained,
            'components': {
                'Regime Detector': self.regime_detector is not None,
                'Ensemble Models': self.ensemble_model.trained if hasattr(self.ensemble_model, 'trained') else False,
                'Deep Learning': self.lstm_model is not None or self.transformer_model is not None,
                'Reinforcement Learning': self.rl_agent is not None,
                'Kelly Criterion': True,
                'Advanced Risk Metrics': True
            },
            'current_regime': (
                self.regime_detector.get_regime_name(self.current_regime)
                if self.current_regime is not None else 'Not detected'
            ),
            'regime_confidence': regime_confidence
        }
