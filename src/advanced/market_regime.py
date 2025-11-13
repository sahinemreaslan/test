"""
Market Regime Detection using Hidden Markov Models

Identifies different market states:
- Bull Market (uptrend)
- Bear Market (downtrend)
- Sideways/Consolidation
- High Volatility

References:
- Ang & Bekaert (2002) - Regime switching models
- Kritzman et al. (2012) - Regime shifts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market regimes using Hidden Markov Model"""

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initialize regime detector

        Args:
            n_regimes: Number of market regimes (default: 4)
                      1 = Bull, 2 = Bear, 3 = Sideways, 4 = High Vol
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "Bull Market",
            1: "Bear Market",
            2: "Sideways",
            3: "High Volatility"
        }

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Scaled feature matrix
        """
        features = pd.DataFrame(index=df.index)

        # Returns (momentum)
        features['returns'] = df['close'].pct_change()
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_20'] = df['close'].pct_change(20)

        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = (
            features['returns'].rolling(5).std() /
            features['returns'].rolling(20).std()
        )

        # Volume
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma_ratio'] = (
                df['volume'] / df['volume'].rolling(20).mean()
            )

        # Trend strength
        features['sma_20'] = df['close'].rolling(20).mean()
        features['price_sma_ratio'] = df['close'] / features['sma_20']

        # Range
        if 'high' in df.columns and 'low' in df.columns:
            features['range'] = (df['high'] - df['low']) / df['close']

        # Drop NaN
        features = features.dropna()

        return features

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """
        Fit HMM to market data

        Args:
            df: DataFrame with OHLCV data
            verbose: Print progress
        """
        logger.info(f"Training {self.n_regimes}-regime HMM...")

        # Prepare features
        features = self.prepare_features(df)

        # Scale features
        X = self.scaler.fit_transform(features.values)

        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state
        )

        self.model.fit(X)

        # Analyze regimes
        hidden_states = self.model.predict(X)
        self._analyze_regimes(features, hidden_states, verbose)

        logger.info("HMM training complete")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict market regime

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with regime labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare features
        features = self.prepare_features(df)

        # Scale
        X = self.scaler.transform(features.values)

        # Predict
        regimes = self.model.predict(X)

        # Create series
        regime_series = pd.Series(regimes, index=features.index)

        return regime_series

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Array of regime probabilities [n_samples, n_regimes]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        features = self.prepare_features(df)
        X = self.scaler.transform(features.values)

        return self.model.predict_proba(X)

    def _analyze_regimes(
        self,
        features: pd.DataFrame,
        regimes: np.ndarray,
        verbose: bool = True
    ):
        """Analyze and characterize each regime"""
        regime_stats = {}

        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_features = features[mask]

            stats = {
                'count': mask.sum(),
                'pct': mask.sum() / len(regimes) * 100,
                'mean_return': regime_features['returns'].mean(),
                'volatility': regime_features['volatility'].mean() if 'volatility' in regime_features else 0,
                'trend': regime_features['returns_20'].mean() if 'returns_20' in regime_features else 0
            }

            regime_stats[regime] = stats

        # Assign regime names based on characteristics
        self._assign_regime_names(regime_stats)

        if verbose:
            logger.info("\nRegime Statistics:")
            for regime, stats in regime_stats.items():
                logger.info(f"\n{self.regime_names[regime]} (Regime {regime}):")
                logger.info(f"  Frequency: {stats['pct']:.1f}%")
                logger.info(f"  Avg Return: {stats['mean_return']*100:.3f}%")
                logger.info(f"  Volatility: {stats['volatility']:.4f}")
                logger.info(f"  Trend: {stats['trend']*100:.3f}%")

    def _assign_regime_names(self, regime_stats: Dict):
        """
        Auto-assign regime names based on statistics

        Logic:
        - Bull: Positive returns, moderate vol
        - Bear: Negative returns, moderate vol
        - Sideways: Low returns, low vol
        - High Vol: High volatility
        """
        regimes_sorted = []

        for regime, stats in regime_stats.items():
            regimes_sorted.append({
                'id': regime,
                'return': stats['mean_return'],
                'vol': stats['volatility']
            })

        # Sort by characteristics
        # High vol regime
        high_vol = max(regimes_sorted, key=lambda x: x['vol'])

        # Remove high vol from list
        regimes_sorted = [r for r in regimes_sorted if r['id'] != high_vol['id']]

        # Bull (highest return)
        bull = max(regimes_sorted, key=lambda x: x['return'])

        # Bear (lowest return)
        bear = min(regimes_sorted, key=lambda x: x['return'])

        # Sideways (remaining)
        sideways = [r for r in regimes_sorted if r['id'] not in [bull['id'], bear['id']]][0]

        # Update regime names
        self.regime_names = {
            bull['id']: "Bull Market",
            bear['id']: "Bear Market",
            sideways['id']: "Sideways",
            high_vol['id']: "High Volatility"
        }

    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable regime name"""
        return self.regime_names.get(regime_id, f"Regime {regime_id}")

    def get_regime_strategy_params(self, regime_id: int) -> Dict:
        """
        Get recommended strategy parameters for each regime

        Args:
            regime_id: Regime identifier

        Returns:
            Dictionary with strategy adjustments
        """
        regime_name = self.get_regime_name(regime_id)

        if "Bull" in regime_name:
            return {
                'position_size_multiplier': 1.2,  # More aggressive
                'stop_loss_multiplier': 0.9,      # Tighter stops
                'take_profit_multiplier': 1.2,    # Higher targets
                'signal_threshold': 0.5           # Lower threshold
            }
        elif "Bear" in regime_name:
            return {
                'position_size_multiplier': 0.5,  # Conservative
                'stop_loss_multiplier': 1.2,      # Wider stops
                'take_profit_multiplier': 0.8,    # Lower targets
                'signal_threshold': 0.7           # Higher threshold
            }
        elif "High Volatility" in regime_name:
            return {
                'position_size_multiplier': 0.3,  # Very small
                'stop_loss_multiplier': 1.5,      # Very wide stops
                'take_profit_multiplier': 1.5,    # Quick exits
                'signal_threshold': 0.8           # Very selective
            }
        else:  # Sideways
            return {
                'position_size_multiplier': 0.7,  # Moderate
                'stop_loss_multiplier': 1.0,      # Normal stops
                'take_profit_multiplier': 1.0,    # Normal targets
                'signal_threshold': 0.6           # Moderate threshold
            }
