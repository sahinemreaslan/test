"""
Fractal Candle Pattern Analysis Module

This module implements the core fractal philosophy:
Every candle has a relationship with the previous candle, creating 4 basic states:
1. HHHL (Higher High Higher Low) - Bullish Power
2. HLLH (Lower High Lower Low) - Bearish Power
3. INSIDE (Inside Bar) - Consolidation/Uncertainty
4. OUTSIDE (Outside Bar) - Volatility/Expansion
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FractalPattern(Enum):
    """Fractal candle relationship patterns"""
    HHHL = "Higher High Higher Low (Bullish Power)"
    HLLH = "Lower High Lower Low (Bearish Power)"
    INSIDE = "Inside Bar (Consolidation)"
    OUTSIDE = "Outside Bar (Volatility Expansion)"
    UNDEFINED = "Undefined"


class FractalAnalyzer:
    """Analyze fractal relationships between candles"""

    def __init__(self):
        self.pattern_scores = {
            FractalPattern.HHHL: 1.0,      # Strong bullish
            FractalPattern.HLLH: -1.0,     # Strong bearish
            FractalPattern.INSIDE: 0.0,    # Neutral (consolidation)
            FractalPattern.OUTSIDE: 0.5,   # Weak bullish (expansion often leads to continuation)
            FractalPattern.UNDEFINED: 0.0
        }

    def identify_pattern(self, current: pd.Series, previous: pd.Series) -> FractalPattern:
        """
        Identify the fractal pattern between current and previous candle

        Args:
            current: Current candle (must have 'high', 'low', 'close', 'open')
            previous: Previous candle

        Returns:
            FractalPattern enum
        """
        high_higher = current['high'] > previous['high']
        high_lower = current['high'] < previous['high']
        low_higher = current['low'] > previous['low']
        low_lower = current['low'] < previous['low']

        # HHHL - Both high and low are higher (Bullish Power)
        if high_higher and low_higher:
            return FractalPattern.HHHL

        # HLLH - Both high and low are lower (Bearish Power)
        elif high_lower and low_lower:
            return FractalPattern.HLLH

        # INSIDE - High is lower AND low is higher (Consolidation)
        elif high_lower and low_higher:
            return FractalPattern.INSIDE

        # OUTSIDE - High is higher AND low is lower (Expansion)
        elif high_higher and low_lower:
            return FractalPattern.OUTSIDE

        else:
            return FractalPattern.UNDEFINED

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entire dataframe and add fractal pattern columns

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with added fractal analysis columns
        """
        df = df.copy()

        # Initialize pattern column
        patterns = []

        for i in range(len(df)):
            if i == 0:
                patterns.append(FractalPattern.UNDEFINED.name)
            else:
                pattern = self.identify_pattern(df.iloc[i], df.iloc[i-1])
                patterns.append(pattern.name)

        df['fractal_pattern'] = patterns

        # Add pattern scores
        df['fractal_score'] = df['fractal_pattern'].apply(
            lambda x: self.pattern_scores.get(FractalPattern[x], 0.0)
        )

        # Add pattern binary flags for ML
        for pattern in FractalPattern:
            df[f'is_{pattern.name.lower()}'] = (df['fractal_pattern'] == pattern.name).astype(int)

        # Calculate pattern streaks (consecutive same patterns)
        df['pattern_streak'] = self._calculate_pattern_streaks(df['fractal_pattern'])

        # Calculate trend strength (cumulative pattern scores)
        df['trend_strength'] = df['fractal_score'].rolling(window=10, min_periods=1).sum()

        # Pattern momentum (rate of change in fractal scores)
        df['pattern_momentum'] = df['fractal_score'].rolling(window=5, min_periods=1).mean()

        return df

    def _calculate_pattern_streaks(self, pattern_series: pd.Series) -> pd.Series:
        """Calculate consecutive occurrences of the same pattern"""
        streaks = []
        current_streak = 0
        prev_pattern = None

        for pattern in pattern_series:
            if pattern == prev_pattern:
                current_streak += 1
            else:
                current_streak = 1
            streaks.append(current_streak)
            prev_pattern = pattern

        return pd.Series(streaks, index=pattern_series.index)

    def get_pattern_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistical summary of fractal patterns

        Args:
            df: DataFrame with fractal analysis

        Returns:
            Dictionary with pattern statistics
        """
        if 'fractal_pattern' not in df.columns:
            df = self.analyze_dataframe(df)

        total = len(df)
        stats = {
            'total_candles': total,
            'pattern_distribution': {},
            'average_streak_length': {},
            'max_streak_length': {}
        }

        for pattern in FractalPattern:
            pattern_name = pattern.name
            count = (df['fractal_pattern'] == pattern_name).sum()
            stats['pattern_distribution'][pattern_name] = {
                'count': int(count),
                'percentage': float(count / total * 100) if total > 0 else 0.0
            }

            # Streak statistics
            pattern_df = df[df['fractal_pattern'] == pattern_name]
            if not pattern_df.empty:
                stats['average_streak_length'][pattern_name] = float(
                    pattern_df['pattern_streak'].mean()
                )
                stats['max_streak_length'][pattern_name] = int(
                    pattern_df['pattern_streak'].max()
                )

        return stats

    def calculate_multi_timeframe_alignment(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate fractal alignment across multiple timeframes

        This is crucial for determining if all timeframes are aligned
        (e.g., all showing HHHL = strong bullish signal)

        Args:
            timeframe_data: Dict mapping timeframe -> DataFrame (with fractal analysis)
            timestamp: Current timestamp to analyze

        Returns:
            Dictionary with alignment metrics
        """
        alignment_scores = {}
        pattern_counts = {pattern.name: 0 for pattern in FractalPattern}

        for tf, df in timeframe_data.items():
            if timestamp not in df.index:
                continue

            pattern = df.loc[timestamp, 'fractal_pattern']
            score = df.loc[timestamp, 'fractal_score']

            alignment_scores[tf] = score
            pattern_counts[pattern] += 1

        # Calculate alignment strength
        scores = list(alignment_scores.values())
        if scores:
            # All positive = bullish alignment, all negative = bearish alignment
            bullish_alignment = sum(1 for s in scores if s > 0) / len(scores)
            bearish_alignment = sum(1 for s in scores if s < 0) / len(scores)
            consensus_strength = abs(np.mean(scores))  # How strong is the consensus
        else:
            bullish_alignment = 0.0
            bearish_alignment = 0.0
            consensus_strength = 0.0

        return {
            'timeframe_scores': alignment_scores,
            'pattern_distribution': pattern_counts,
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'consensus_strength': consensus_strength,
            'net_score': np.sum(scores) if scores else 0.0
        }

    def get_fractal_features(
        self,
        df: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Generate additional fractal-based features for ML

        Args:
            df: DataFrame with fractal analysis
            lookback_periods: Periods to calculate rolling features

        Returns:
            DataFrame with additional fractal features
        """
        df = df.copy()

        if 'fractal_score' not in df.columns:
            df = self.analyze_dataframe(df)

        for period in lookback_periods:
            # Rolling pattern distribution
            df[f'hhhl_ratio_{period}'] = (
                df['is_hhhl'].rolling(window=period, min_periods=1).mean()
            )
            df[f'hllh_ratio_{period}'] = (
                df['is_hllh'].rolling(window=period, min_periods=1).mean()
            )
            df[f'inside_ratio_{period}'] = (
                df['is_inside'].rolling(window=period, min_periods=1).mean()
            )
            df[f'outside_ratio_{period}'] = (
                df['is_outside'].rolling(window=period, min_periods=1).mean()
            )

            # Rolling fractal score statistics
            df[f'fractal_score_mean_{period}'] = (
                df['fractal_score'].rolling(window=period, min_periods=1).mean()
            )
            df[f'fractal_score_std_{period}'] = (
                df['fractal_score'].rolling(window=period, min_periods=1).std()
            )

            # Trend consistency (how consistent is the trend)
            df[f'trend_consistency_{period}'] = (
                df['fractal_score'].rolling(window=period, min_periods=1).apply(
                    lambda x: len(x[x > 0]) / len(x) if len(x[x > 0]) > len(x[x < 0])
                    else len(x[x < 0]) / len(x),
                    raw=True
                )
            )

        return df
