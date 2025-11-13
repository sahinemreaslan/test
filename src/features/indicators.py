"""
Technical Indicators Calculator Module

Calculates various technical indicators for multiple timeframes:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- EMAs (Exponential Moving Averages)
- Volume indicators
- Heiken Ashi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Calculate technical indicators for trading analysis"""

    def __init__(self, config: Dict = None):
        """
        Initialize with configuration

        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """Default indicator configuration"""
        return {
            'rsi': {'periods': [14, 21, 28]},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'stochastic': {'k_period': 14, 'd_period': 3, 'smooth_k': 3},
            'atr': {'period': 14},
            'ema': {'periods': [9, 21, 50, 100, 200]},
            'volume': {'ma_period': 20}
        }

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            df: DataFrame with price data
            period: RSI period
            column: Column to calculate RSI on

        Returns:
            Series with RSI values
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False, min_periods=1).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with price data
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column to calculate bands on

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = df[column].rolling(window=period, min_periods=1).mean()
        std = df[column].rolling(window=period, min_periods=1).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator

        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D period
            smooth_k: Smoothing period for %K

        Returns:
            Tuple of (%K, %D)
        """
        # Calculate %K
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()

        k_fast = 100 * (df['close'] - low_min) / (high_max - low_min)

        # Smooth %K
        k = k_fast.rolling(window=smooth_k, min_periods=1).mean()

        # Calculate %D
        d = k.rolling(window=d_period, min_periods=1).mean()

        return k, d

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    def calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average

        Args:
            df: DataFrame with price data
            period: EMA period
            column: Column to calculate EMA on

        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=period, adjust=False, min_periods=1).mean()

    def calculate_volume_indicators(self, df: pd.DataFrame, ma_period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators

        Args:
            df: DataFrame with volume data
            ma_period: Moving average period for volume

        Returns:
            Dictionary with volume indicators
        """
        indicators = {}

        # Volume moving average
        indicators['volume_ma'] = df['volume'].rolling(window=ma_period, min_periods=1).mean()

        # Volume ratio (current volume vs average)
        indicators['volume_ratio'] = df['volume'] / indicators['volume_ma']

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        indicators['obv'] = obv

        # Volume-weighted average price (VWAP) - daily
        if 'high' in df.columns and 'low' in df.columns:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            indicators['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return indicators

    def calculate_heiken_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heiken Ashi candles

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with Heiken Ashi OHLC
        """
        ha_df = pd.DataFrame(index=df.index)

        # HA Close
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # HA Open
        ha_df['ha_open'] = 0.0
        for i in range(len(df)):
            if i == 0:
                ha_df.loc[ha_df.index[i], 'ha_open'] = (df['open'].iloc[i] + df['close'].iloc[i]) / 2
            else:
                ha_df.loc[ha_df.index[i], 'ha_open'] = (
                    ha_df['ha_open'].iloc[i-1] + ha_df['ha_close'].iloc[i-1]
                ) / 2

        # HA High and Low
        ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
        ha_df['ha_low'] = pd.concat([df['low'], ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)

        # HA color (1 = green, -1 = red)
        ha_df['ha_color'] = np.where(ha_df['ha_close'] >= ha_df['ha_open'], 1, -1)

        # HA body size
        ha_df['ha_body'] = abs(ha_df['ha_close'] - ha_df['ha_open'])
        ha_df['ha_body_pct'] = ha_df['ha_body'] / (ha_df['ha_high'] - ha_df['ha_low'])

        return ha_df

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()

        logger.info("Calculating all technical indicators...")

        # RSI for multiple periods
        for period in self.config['rsi']['periods']:
            df[f'rsi_{period}'] = self.calculate_rsi(df, period=period)

        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(
            df,
            fast=self.config['macd']['fast'],
            slow=self.config['macd']['slow'],
            signal=self.config['macd']['signal']
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            df,
            period=self.config['bollinger']['period'],
            std_dev=self.config['bollinger']['std_dev']
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(
            df,
            k_period=self.config['stochastic']['k_period'],
            d_period=self.config['stochastic']['d_period'],
            smooth_k=self.config['stochastic']['smooth_k']
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # ATR
        df['atr'] = self.calculate_atr(df, period=self.config['atr']['period'])
        df['atr_pct'] = df['atr'] / df['close']

        # EMAs
        for period in self.config['ema']['periods']:
            df[f'ema_{period}'] = self.calculate_ema(df, period=period)

        # EMA crossovers
        df['ema_9_21_cross'] = df['ema_9'] - df['ema_21']
        df['ema_21_50_cross'] = df['ema_21'] - df['ema_50']
        df['ema_50_200_cross'] = df['ema_50'] - df['ema_200']

        # Price vs EMAs
        df['price_above_ema_9'] = (df['close'] > df['ema_9']).astype(int)
        df['price_above_ema_21'] = (df['close'] > df['ema_21']).astype(int)
        df['price_above_ema_50'] = (df['close'] > df['ema_50']).astype(int)
        df['price_above_ema_200'] = (df['close'] > df['ema_200']).astype(int)

        # Volume indicators
        vol_indicators = self.calculate_volume_indicators(
            df,
            ma_period=self.config['volume']['ma_period']
        )
        for name, values in vol_indicators.items():
            df[name] = values

        # Heiken Ashi
        ha_df = self.calculate_heiken_ashi(df)
        for col in ha_df.columns:
            df[col] = ha_df[col]

        # Additional derived features
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['volume_momentum'] = df['volume'].pct_change(periods=5)

        # Volatility (rolling standard deviation of returns)
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()

        logger.info(f"Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} indicator features")

        return df

    def get_indicator_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from indicators

        Args:
            df: DataFrame with calculated indicators

        Returns:
            DataFrame with signal columns
        """
        df = df.copy()

        # RSI signals (oversold/overbought)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)

        # MACD signals
        df['macd_bullish'] = (df['macd_histogram'] > 0).astype(int)
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        # Bollinger Bands signals
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(window=20, min_periods=1).quantile(0.2)).astype(int)
        df['price_below_bb_lower'] = (df['close'] < df['bb_lower']).astype(int)
        df['price_above_bb_upper'] = (df['close'] > df['bb_upper']).astype(int)

        # Stochastic signals
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

        # Trend signals (EMA alignment)
        df['bullish_ema_alignment'] = (
            (df['ema_9'] > df['ema_21']) &
            (df['ema_21'] > df['ema_50']) &
            (df['ema_50'] > df['ema_200'])
        ).astype(int)

        df['bearish_ema_alignment'] = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['ema_50'] < df['ema_200'])
        ).astype(int)

        # Volume signals
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        df['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)

        # Heiken Ashi signals
        df['ha_bullish_streak'] = (df['ha_color'] == 1).astype(int).rolling(window=3, min_periods=1).sum()
        df['ha_bearish_streak'] = (df['ha_color'] == -1).astype(int).rolling(window=3, min_periods=1).sum()

        return df
