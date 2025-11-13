"""
Timeframe conversion module - Convert 15m data to multiple timeframes
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TimeframeConverter:
    """Convert base timeframe to multiple higher timeframes"""

    # Mapping of timeframe strings to pandas resample rules
    # NOTE: Can only convert to LARGER timeframes than base data
    # If base data is 15m, you cannot create 5m or 10m
    TIMEFRAME_MAP = {
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '8h': '8h',
        '12h': '12h',
        '1D': '1D',
        '1W': '1W',
        '1M': '1MS',  # Month start
        '3M': '3MS'   # Quarterly
    }

    def __init__(self, base_df: pd.DataFrame):
        """
        Initialize with base timeframe data

        Args:
            base_df: DataFrame with OHLCV data (must have datetime index)
        """
        self.base_df = base_df.copy()
        self.converted_data: Dict[str, pd.DataFrame] = {}

    def convert_to_timeframe(self, timeframe: str) -> pd.DataFrame:
        """
        Convert base data to specified timeframe

        Args:
            timeframe: Target timeframe (e.g., '1h', '4h', '1D')

        Returns:
            DataFrame with converted OHLCV data
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. "
                           f"Supported: {list(self.TIMEFRAME_MAP.keys())}")

        rule = self.TIMEFRAME_MAP[timeframe]

        # Resample OHLCV data
        resampled = pd.DataFrame()

        # OHLC aggregation
        resampled['open'] = self.base_df['open'].resample(rule).first()
        resampled['high'] = self.base_df['high'].resample(rule).max()
        resampled['low'] = self.base_df['low'].resample(rule).min()
        resampled['close'] = self.base_df['close'].resample(rule).last()

        # Volume aggregation (sum)
        resampled['volume'] = self.base_df['volume'].resample(rule).sum()

        # Forward fill any missing values (for gaps in data)
        resampled = resampled.dropna()

        # Add derived features
        resampled['returns'] = resampled['close'].pct_change()
        resampled['log_returns'] = np.log(resampled['close'] / resampled['close'].shift(1))
        resampled['range'] = resampled['high'] - resampled['low']
        resampled['body'] = abs(resampled['close'] - resampled['open'])
        resampled['body_pct'] = resampled['body'] / resampled['range']
        resampled['is_bullish'] = (resampled['close'] > resampled['open']).astype(int)

        logger.info(f"Converted to {timeframe}: {len(resampled)} candles")

        return resampled

    def convert_all_timeframes(self, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Convert to all specified timeframes

        Args:
            timeframes: List of timeframes to convert

        Returns:
            Dictionary mapping timeframe -> DataFrame
        """
        logger.info(f"Converting to {len(timeframes)} timeframes: {timeframes}")

        for tf in timeframes:
            self.converted_data[tf] = self.convert_to_timeframe(tf)

        return self.converted_data

    def get_aligned_data(self, timeframes: List[str], reference_tf: str = '15m') -> Dict[str, pd.DataFrame]:
        """
        Get timeframe data aligned to reference timeframe timestamps

        This ensures that for each candle in the reference timeframe,
        we can get the corresponding higher timeframe context.

        Args:
            timeframes: List of timeframes to align
            reference_tf: Reference timeframe (usually the trading timeframe)

        Returns:
            Dictionary of aligned DataFrames
        """
        if not self.converted_data:
            self.convert_all_timeframes(timeframes)

        # Get reference data
        if reference_tf not in self.converted_data:
            self.converted_data[reference_tf] = self.convert_to_timeframe(reference_tf)

        ref_data = self.converted_data[reference_tf]
        aligned_data = {}

        for tf in timeframes:
            if tf == reference_tf:
                aligned_data[tf] = ref_data.copy()
            else:
                # Forward fill higher timeframe data to match reference timeframe
                tf_data = self.converted_data[tf]
                aligned = tf_data.reindex(ref_data.index, method='ffill')
                aligned_data[tf] = aligned

        return aligned_data

    @staticmethod
    def get_timeframe_multiplier(tf1: str, tf2: str) -> float:
        """
        Get the multiplier between two timeframes

        Args:
            tf1: First timeframe
            tf2: Second timeframe

        Returns:
            Multiplier (how many times tf1 fits into tf2)
        """
        # Convert to minutes
        minutes_map = {
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '8h': 480,
            '12h': 720,
            '1D': 1440,
            '1W': 10080,
            '1M': 43200,  # Approximate
            '3M': 129600  # Approximate
        }

        return minutes_map.get(tf2, 1) / minutes_map.get(tf1, 1)
