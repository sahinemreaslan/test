"""
Fractal Multi-Timeframe Trading Strategy

Combines:
- Fractal pattern analysis across multiple timeframes
- Technical indicators
- XGBoost ML predictions
- Genetic algorithm optimized parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FractalMultiTimeframeStrategy:
    """Main trading strategy implementation"""

    def __init__(
        self,
        params: Dict,
        ml_model: Optional[any] = None
    ):
        """
        Initialize strategy

        Args:
            params: Strategy parameters (from genetic algorithm or config)
            ml_model: Optional trained XGBoost model
        """
        self.params = params
        self.ml_model = ml_model

        # Extract key parameters
        self.ml_confidence_threshold = params.get('ml_confidence_threshold', 0.6)
        self.fractal_score_threshold = params.get('fractal_score_threshold', 0.5)

        # Timeframe weights
        self.timeframe_weights = {
            k.replace('weight_', ''): v
            for k, v in params.items()
            if k.startswith('weight_')
        }

        # Indicator weights
        self.indicator_weights = {
            k.replace('ind_weight_', ''): v
            for k, v in params.items()
            if k.startswith('ind_weight_')
        }

        # RSI thresholds
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)

    def generate_signals(
        self,
        features: pd.DataFrame,
        use_ml: bool = True
    ) -> pd.Series:
        """
        Generate trading signals

        Args:
            features: Multi-timeframe feature DataFrame
            use_ml: Whether to use ML model predictions

        Returns:
            Series with signals (1 = buy, 0 = hold, -1 = sell)
        """
        logger.info("Generating trading signals...")

        signals = pd.Series(0, index=features.index)

        # Get ML predictions if available
        if use_ml and self.ml_model is not None:
            try:
                ml_probabilities = self.ml_model.predict_proba(features)
                ml_predictions = (ml_probabilities >= self.ml_confidence_threshold).astype(int)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Falling back to rule-based signals.")
                ml_predictions = None
                ml_probabilities = None
        else:
            ml_predictions = None
            ml_probabilities = None

        # Generate signals for each timestamp
        for i, (timestamp, row) in enumerate(features.iterrows()):
            # Rule-based signal
            rule_signal = self._calculate_rule_based_signal(row)

            # ML signal (if available)
            if ml_predictions is not None:
                ml_signal = ml_predictions[i]
                ml_confidence = ml_probabilities[i]

                # Combine rule-based and ML signals
                # ML must agree with rules and have sufficient confidence
                if ml_signal == 1 and ml_confidence >= self.ml_confidence_threshold:
                    if rule_signal == 1:
                        signals.iloc[i] = 1  # Strong buy
                    elif rule_signal == 0:
                        signals.iloc[i] = 1  # ML override (medium confidence)
                elif ml_signal == 0:
                    # ML says no trade
                    if rule_signal == -1:
                        signals.iloc[i] = -1  # Exit signal
                    else:
                        signals.iloc[i] = 0  # Hold
            else:
                # Use only rule-based signal
                signals.iloc[i] = rule_signal

        logger.info(f"Generated {(signals == 1).sum()} buy signals, "
                   f"{(signals == -1).sum()} sell signals")

        return signals

    def _calculate_rule_based_signal(self, features: pd.Series) -> int:
        """
        Calculate signal using rule-based logic

        Args:
            features: Feature row

        Returns:
            Signal: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate weighted fractal score across timeframes
        fractal_score = self._calculate_weighted_fractal_score(features)

        # Calculate indicator score
        indicator_score = self._calculate_indicator_score(features)

        # Calculate timeframe alignment
        alignment_score = self._calculate_alignment_score(features)

        # Combine scores
        total_score = (
            fractal_score * 0.4 +
            indicator_score * 0.3 +
            alignment_score * 0.3
        )

        # Generate signal based on total score
        if total_score > self.fractal_score_threshold:
            return 1  # Buy
        elif total_score < -self.fractal_score_threshold:
            return -1  # Sell
        else:
            return 0  # Hold

    def _calculate_weighted_fractal_score(self, features: pd.Series) -> float:
        """Calculate weighted fractal score from all timeframes"""
        total_score = 0.0
        total_weight = 0.0

        for tf, weight in self.timeframe_weights.items():
            score_col = f'{tf}_fractal_score'
            if score_col in features.index:
                score = features[score_col]
                if not pd.isna(score):
                    total_score += score * weight
                    total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0

    def _calculate_indicator_score(self, features: pd.Series) -> float:
        """Calculate weighted indicator score"""
        score = 0.0
        total_weight = sum(self.indicator_weights.values())

        if total_weight == 0:
            return 0.0

        # Reference timeframe (usually 15m)
        ref_tf = '15m'  # Could be made configurable

        # RSI score
        rsi_col = f'{ref_tf}_rsi_14'
        if rsi_col in features.index and not pd.isna(features[rsi_col]):
            rsi = features[rsi_col]
            if rsi < self.rsi_oversold:
                rsi_score = 1.0  # Bullish
            elif rsi > self.rsi_overbought:
                rsi_score = -1.0  # Bearish
            else:
                # Normalize RSI to [-1, 1]
                rsi_score = (rsi - 50) / 50
            score += rsi_score * self.indicator_weights.get('rsi', 1.0)

        # MACD score
        macd_col = f'{ref_tf}_macd_histogram'
        if macd_col in features.index and not pd.isna(features[macd_col]):
            macd_hist = features[macd_col]
            macd_score = 1.0 if macd_hist > 0 else -1.0
            score += macd_score * self.indicator_weights.get('macd', 1.0)

        # Bollinger Bands score
        bb_pos_col = f'{ref_tf}_bb_position'
        if bb_pos_col in features.index and not pd.isna(features[bb_pos_col]):
            bb_pos = features[bb_pos_col]
            # Normalize BB position to [-1, 1]
            bb_score = (bb_pos - 0.5) * 2
            score += bb_score * self.indicator_weights.get('bollinger', 1.0)

        # EMA alignment score
        ema_align_col = f'{ref_tf}_bullish_ema_alignment'
        if ema_align_col in features.index and not pd.isna(features[ema_align_col]):
            ema_score = 1.0 if features[ema_align_col] == 1 else -1.0
            score += ema_score * self.indicator_weights.get('ema', 1.0)

        # Volume score
        vol_ratio_col = f'{ref_tf}_volume_ratio'
        if vol_ratio_col in features.index and not pd.isna(features[vol_ratio_col]):
            vol_ratio = features[vol_ratio_col]
            vol_score = 1.0 if vol_ratio > 1.2 else 0.0  # High volume is bullish
            score += vol_score * self.indicator_weights.get('volume', 1.0)

        # Heiken Ashi score
        ha_col = f'{ref_tf}_ha_color'
        if ha_col in features.index and not pd.isna(features[ha_col]):
            ha_score = 1.0 if features[ha_col] == 1 else -1.0
            score += ha_score * self.indicator_weights.get('heiken_ashi', 1.0)

        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_alignment_score(self, features: pd.Series) -> float:
        """Calculate timeframe alignment score"""
        # Check for cross-timeframe features
        if 'fractal_consensus' in features.index:
            return features['fractal_consensus']
        elif 'fractal_score_mean_all_tf' in features.index:
            return features['fractal_score_mean_all_tf']
        else:
            return 0.0

    def get_position_size(self, capital: float) -> float:
        """
        Calculate position size

        Args:
            capital: Available capital

        Returns:
            Position size in capital units
        """
        position_size_pct = self.params.get('position_size', 0.05)
        return capital * position_size_pct

    def get_stop_loss_take_profit(
        self,
        entry_price: float,
        atr: float,
        side: str = 'long'
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Entry price
            atr: Average True Range
            side: 'long' or 'short'

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        stop_loss_mult = self.params.get('stop_loss_atr', 2.0)
        take_profit_mult = self.params.get('take_profit_atr', 4.0)

        if side == 'long':
            stop_loss = entry_price - (atr * stop_loss_mult)
            take_profit = entry_price + (atr * take_profit_mult)
        else:
            stop_loss = entry_price + (atr * stop_loss_mult)
            take_profit = entry_price - (atr * take_profit_mult)

        return stop_loss, take_profit

    def get_strategy_summary(self) -> Dict:
        """Get strategy parameter summary"""
        return {
            'ml_confidence_threshold': self.ml_confidence_threshold,
            'fractal_score_threshold': self.fractal_score_threshold,
            'top_timeframe_weights': sorted(
                self.timeframe_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'top_indicator_weights': sorted(
                self.indicator_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'rsi_thresholds': (self.rsi_oversold, self.rsi_overbought),
            'risk_params': {
                'stop_loss_atr': self.params.get('stop_loss_atr'),
                'take_profit_atr': self.params.get('take_profit_atr'),
                'position_size': self.params.get('position_size')
            }
        }
