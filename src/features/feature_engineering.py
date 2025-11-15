"""
Feature Engineering Pipeline

Combines fractal analysis and technical indicators across multiple timeframes
to create a comprehensive feature set for ML models and genetic algorithm optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .fractal_analysis import FractalAnalyzer
from .indicators import IndicatorCalculator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from multi-timeframe data"""

    def __init__(self, config: Dict):
        """
        Initialize feature engineer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fractal_analyzer = FractalAnalyzer()
        self.indicator_calculator = IndicatorCalculator(config.get('indicators', {}))
        self.timeframes = config.get('timeframes', {}).get('all', [])
        self.trading_tf = config.get('timeframes', {}).get('trading', '5m')

    def process_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Process a single timeframe: add fractal analysis and indicators

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe identifier

        Returns:
            DataFrame with all features
        """
        logger.info(f"Processing timeframe: {timeframe}")

        # Add fractal analysis
        df = self.fractal_analyzer.analyze_dataframe(df)
        df = self.fractal_analyzer.get_fractal_features(df)

        # Add technical indicators
        df = self.indicator_calculator.calculate_all_indicators(df)
        df = self.indicator_calculator.get_indicator_signals(df)

        return df

    def create_multi_timeframe_features(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        reference_tf: str = '15m'
    ) -> pd.DataFrame:
        """
        Create multi-timeframe feature matrix

        For each timestamp in the reference timeframe, combine features from all timeframes.
        This allows the model to see the "big picture" from higher timeframes while trading
        on the reference timeframe.

        Args:
            timeframe_data: Dict mapping timeframe -> processed DataFrame
            reference_tf: Reference timeframe (usually trading timeframe)

        Returns:
            DataFrame with multi-timeframe features aligned to reference timeframe
        """
        logger.info(f"Creating multi-timeframe features (reference: {reference_tf})")

        if reference_tf not in timeframe_data:
            raise ValueError(f"Reference timeframe {reference_tf} not in timeframe_data")

        # Start with reference timeframe data
        ref_df = timeframe_data[reference_tf].copy()
        feature_df = pd.DataFrame(index=ref_df.index)

        # Add reference timeframe features (with prefix) - use dict to avoid fragmentation
        ref_cols = {}
        for col in ref_df.columns:
            ref_cols[f'{reference_tf}_{col}'] = ref_df[col]
        feature_df = pd.DataFrame(ref_cols, index=ref_df.index)

        # Add features from other timeframes
        for tf, df in timeframe_data.items():
            if tf == reference_tf:
                continue

            logger.info(f"  Adding features from {tf}")

            # Forward fill higher timeframe data to match reference timeframe
            # CRITICAL: Shift by 1 to prevent look-ahead bias!
            # Higher timeframe candle close is only known AFTER the candle completes
            # Example: Daily candle at 2024-01-15 23:59 is available from 2024-01-16 00:00
            aligned_df = df.reindex(ref_df.index, method='ffill').shift(1)

            # Select important features to include
            important_features = self._get_important_features(aligned_df)

            # Collect columns to add in a temporary dict (avoid fragmentation)
            temp_cols = {}
            for col in important_features:
                temp_cols[f'{tf}_{col}'] = aligned_df[col]

            # Add all columns at once using pd.concat (much faster)
            if temp_cols:
                temp_df = pd.DataFrame(temp_cols, index=ref_df.index)
                feature_df = pd.concat([feature_df, temp_df], axis=1)

        # Add cross-timeframe features
        feature_df = self._add_cross_timeframe_features(feature_df, timeframe_data.keys())

        # Fill any remaining NaN values
        feature_df = feature_df.ffill().bfill().fillna(0)

        logger.info(f"Created feature matrix with {len(feature_df.columns)} features")

        return feature_df

    def _get_important_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select important features to include from each timeframe

        Args:
            df: DataFrame with all features

        Returns:
            List of important feature column names
        """
        # Core OHLCV features
        core_features = ['open', 'high', 'low', 'close', 'volume']

        # Fractal features (exclude 'fractal_pattern' as it's a string and gets dropped during ML prep)
        fractal_features = [
            'fractal_score', 'trend_strength', 'pattern_momentum',
            'is_hhhl', 'is_hllh', 'is_inside', 'is_outside', 'is_undefined'
        ]

        # Indicator features
        indicator_features = [
            'rsi_14', 'macd', 'macd_histogram', 'bb_position', 'bb_width',
            'stoch_k', 'atr_pct', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'volume_ratio', 'ha_color', 'volatility'
        ]

        # Signal features
        signal_features = [
            'rsi_oversold', 'rsi_overbought', 'macd_bullish',
            'bb_squeeze', 'bullish_ema_alignment', 'bearish_ema_alignment',
            'high_volume'
        ]

        # Combine all
        all_important = (
            core_features + fractal_features + indicator_features + signal_features
        )

        # Filter to only existing columns
        return [col for col in all_important if col in df.columns]

    def _add_cross_timeframe_features(
        self,
        feature_df: pd.DataFrame,
        timeframes: List[str]
    ) -> pd.DataFrame:
        """
        Add features that compare across timeframes

        Args:
            feature_df: DataFrame with multi-timeframe features
            timeframes: List of timeframes

        Returns:
            DataFrame with cross-timeframe features added
        """
        logger.info("Adding cross-timeframe features...")

        # Fractal alignment across timeframes
        fractal_scores = []
        for tf in timeframes:
            score_col = f'{tf}_fractal_score'
            if score_col in feature_df.columns:
                fractal_scores.append(score_col)

        if fractal_scores:
            # Average fractal score across all timeframes
            feature_df['fractal_score_mean_all_tf'] = feature_df[fractal_scores].mean(axis=1)

            # Fractal consensus (how many timeframes agree on direction)
            feature_df['fractal_bullish_count'] = (feature_df[fractal_scores] > 0).sum(axis=1)
            feature_df['fractal_bearish_count'] = (feature_df[fractal_scores] < 0).sum(axis=1)
            feature_df['fractal_consensus'] = (
                feature_df['fractal_bullish_count'] - feature_df['fractal_bearish_count']
            ) / len(fractal_scores)

        # RSI divergence across timeframes
        rsi_cols = []
        for tf in timeframes:
            rsi_col = f'{tf}_rsi_14'
            if rsi_col in feature_df.columns:
                rsi_cols.append(rsi_col)

        if len(rsi_cols) >= 2:
            feature_df['rsi_mean_all_tf'] = feature_df[rsi_cols].mean(axis=1)
            feature_df['rsi_std_all_tf'] = feature_df[rsi_cols].std(axis=1)

        # Trend alignment across timeframes (using EMAs)
        bullish_ema_cols = []
        for tf in timeframes:
            col = f'{tf}_bullish_ema_alignment'
            if col in feature_df.columns:
                bullish_ema_cols.append(col)

        if bullish_ema_cols:
            feature_df['bullish_ema_count'] = feature_df[bullish_ema_cols].sum(axis=1)
            feature_df['ema_alignment_strength'] = (
                feature_df['bullish_ema_count'] / len(bullish_ema_cols)
            )

        # Volume confirmation across timeframes
        volume_ratio_cols = []
        for tf in timeframes:
            col = f'{tf}_volume_ratio'
            if col in feature_df.columns:
                volume_ratio_cols.append(col)

        if volume_ratio_cols:
            feature_df['volume_ratio_mean_all_tf'] = feature_df[volume_ratio_cols].mean(axis=1)

        return feature_df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        method: str = 'forward_returns',
        horizon: int = 1,
        threshold: float = 0.001
    ) -> pd.Series:
        """
        Create target variable for ML model

        Args:
            df: DataFrame with price data
            method: Method to create target ('forward_returns', 'trend', 'breakout')
            horizon: Forward-looking periods
            threshold: Threshold for classification

        Returns:
            Series with target variable (1 = buy, 0 = hold/sell)
        """
        logger.info(f"Creating target variable (method={method}, horizon={horizon})")

        if method == 'forward_returns':
            # Future returns
            future_returns = df['close'].pct_change(periods=horizon).shift(-horizon)

            # Binary classification: 1 if return > threshold, 0 otherwise
            target = (future_returns > threshold).astype(int)

        elif method == 'trend':
            # Trend direction: 1 if price is higher in N periods
            future_price = df['close'].shift(-horizon)
            target = (future_price > df['close']).astype(int)

        elif method == 'breakout':
            # Breakout: 1 if price breaks above recent high
            rolling_high = df['high'].rolling(window=20, min_periods=1).max()
            future_high = df['high'].shift(-horizon)
            target = (future_high > rolling_high).astype(int)

        else:
            raise ValueError(f"Unknown target method: {method}")

        return target

    def prepare_ml_dataset(
        self,
        feature_df: pd.DataFrame,
        target_method: str = 'forward_returns',
        target_horizon: int = 1,
        target_threshold: float = 0.001,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final dataset for ML model

        Args:
            feature_df: DataFrame with all features
            target_method: Method to create target variable
            target_horizon: Forward-looking periods for target
            target_threshold: Threshold for classification
            drop_na: Whether to drop NaN values

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing ML dataset...")

        # Get reference timeframe close price for target
        ref_tf = self.config.get('timeframes', {}).get('signal', '15m')
        close_col = f'{ref_tf}_close'

        if close_col not in feature_df.columns:
            raise ValueError(f"Close price column {close_col} not found in features")

        # Create temporary df with close price
        temp_df = pd.DataFrame({'close': feature_df[close_col]}, index=feature_df.index)

        # Create target
        target = self.create_target_variable(
            temp_df,
            method=target_method,
            horizon=target_horizon,
            threshold=target_threshold
        )

        # Remove non-numeric columns (like 'fractal_pattern' strings)
        numeric_features = feature_df.select_dtypes(include=[np.number])

        # Drop highly correlated features (optional, for dimensionality reduction)
        # numeric_features = self._remove_highly_correlated_features(numeric_features)

        # Align features and target
        features = numeric_features.loc[target.index]

        if drop_na:
            # Drop rows with NaN in target
            valid_idx = target.notna()
            features = features[valid_idx]
            target = target[valid_idx]

        logger.info(f"Final dataset: {len(features)} samples, {len(features.columns)} features")
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")

        return features, target

    def _remove_highly_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce dimensionality

        Args:
            df: DataFrame with features
            threshold: Correlation threshold

        Returns:
            DataFrame with reduced features
        """
        logger.info(f"Removing features with correlation > {threshold}")

        # Calculate correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        logger.info(f"Dropping {len(to_drop)} highly correlated features")

        return df.drop(columns=to_drop)

    def get_feature_importance_names(self, feature_df: pd.DataFrame) -> List[str]:
        """
        Get ordered list of feature names for importance analysis

        Args:
            feature_df: DataFrame with features

        Returns:
            List of feature names
        """
        return list(feature_df.select_dtypes(include=[np.number]).columns)
