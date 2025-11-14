"""
Walk-Forward Analysis & Market Regime Testing

Tests the strategy across different time periods and market conditions:
1. Train/Test Split (80/20)
2. Annual Walk-Forward (each year separately)
3. Market Regime Split (Bull/Bear/Sideways)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.helpers import ConfigLoader, setup_logging, create_output_dirs
from src.data.data_loader import DataLoader
from src.data.timeframe_converter import TimeframeConverter
from src.features.fractal_analysis import FractalAnalyzer
from src.features.indicators import IndicatorCalculator
from src.features.feature_engineering import FeatureEngineer
from src.backtesting.backtester import Backtester

# Advanced system (optional)
try:
    from src.advanced.integrated_system import AdvancedTradingSystem
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    """Perform walk-forward and regime-based analysis"""

    def __init__(self, config_path: str, use_advanced: bool = False):
        self.config = ConfigLoader.load(config_path)
        self.use_advanced = use_advanced and ADVANCED_AVAILABLE

        # Data storage
        self.base_data = None
        self.timeframe_data = {}
        self.features = None
        self.target = None

    def load_and_prepare_data(self):
        """Load and prepare all data"""
        logger.info("Loading and preparing data...")

        # Load data
        data_config = self.config.get('data', {})
        file_path = data_config.get('file_path', 'btc_15m_data_2018_to_2025.csv')

        data_loader = DataLoader(file_path)
        self.base_data = data_loader.load_data()

        # Convert timeframes
        timeframes = self.config.get('timeframes', {}).get('all', [])
        converter = TimeframeConverter(self.base_data)
        self.timeframe_data = converter.convert_all_timeframes(timeframes)

        # Process all timeframes
        fractal_analyzer = FractalAnalyzer()
        indicator_calculator = IndicatorCalculator(self.config.get('indicators', {}))

        for tf, df in self.timeframe_data.items():
            df = fractal_analyzer.analyze_dataframe(df)
            df = fractal_analyzer.get_fractal_features(df)
            df = indicator_calculator.calculate_all_indicators(df)
            df = indicator_calculator.get_indicator_signals(df)
            self.timeframe_data[tf] = df

        # Engineer features
        feature_engineer = FeatureEngineer(self.config)
        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')

        self.features = feature_engineer.create_multi_timeframe_features(
            self.timeframe_data,
            reference_tf=signal_tf
        )

        self.features, self.target = feature_engineer.prepare_ml_dataset(
            self.features,
            target_method='forward_returns',
            target_horizon=1,
            target_threshold=0.001
        )

        logger.info(f"Data prepared: {len(self.base_data)} candles")

    def train_test_split_analysis(self, train_ratio: float = 0.8):
        """
        Train on first X%, test on remaining

        Args:
            train_ratio: Ratio of data to use for training
        """
        logger.info("\n" + "="*70)
        logger.info(f"TRAIN/TEST SPLIT ANALYSIS ({int(train_ratio*100)}/{int((1-train_ratio)*100)})")
        logger.info("="*70)

        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')
        df = self.timeframe_data[signal_tf]

        # Split point
        split_idx = int(len(df) * train_ratio)
        split_date = df.index[split_idx]

        logger.info(f"Train period: {df.index[0]} to {split_date}")
        logger.info(f"Test period: {split_date} to {df.index[-1]}")

        # Train on first portion
        train_features = self.features.iloc[:split_idx]
        train_target = self.target.iloc[:split_idx]

        logger.info(f"\nTraining on {len(train_features)} samples...")
        advanced_system = self._train_system(df.iloc[:split_idx], train_features, train_target)

        # Test on remaining portion
        test_df = df.iloc[split_idx:]
        test_features = self.features.iloc[split_idx:]

        logger.info(f"\nTesting on {len(test_features)} samples...")
        results = self._run_backtest(advanced_system, test_df, test_features, "test")

        return results

    def annual_walk_forward(self):
        """Test each year separately"""
        logger.info("\n" + "="*70)
        logger.info("ANNUAL WALK-FORWARD ANALYSIS")
        logger.info("="*70)

        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')
        df = self.timeframe_data[signal_tf]

        # Get unique years
        years = df.index.year.unique()
        logger.info(f"Found {len(years)} years: {list(years)}")

        annual_results = {}

        for i, year in enumerate(years):
            if i == 0:
                continue  # Skip first year (need previous data for training)

            logger.info(f"\n{'='*70}")
            logger.info(f"YEAR {year}")
            logger.info("="*70)

            # Train on all data up to this year
            train_mask = df.index.year < year
            test_mask = df.index.year == year

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(test_df) == 0:
                logger.warning(f"No data for year {year}, skipping")
                continue

            logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} samples)")
            logger.info(f"Test: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} samples)")

            # Get corresponding features
            train_features = self.features[train_mask]
            train_target = self.target[train_mask]
            test_features = self.features[test_mask]

            # Train
            logger.info(f"\nTraining on data up to {year}...")
            advanced_system = self._train_system(train_df, train_features, train_target)

            # Test
            logger.info(f"\nTesting on year {year}...")
            results = self._run_backtest(advanced_system, test_df, test_features, f"year_{year}")
            annual_results[year] = results

        # Summary
        logger.info("\n" + "="*70)
        logger.info("ANNUAL SUMMARY")
        logger.info("="*70)

        for year, results in annual_results.items():
            logger.info(f"\n{year}:")
            logger.info(f"  Return: {results['total_return']:.2%}")
            logger.info(f"  Sharpe: {results['sharpe_ratio']:.3f}")
            logger.info(f"  Max DD: {results['max_drawdown']:.2%}")
            logger.info(f"  Win Rate: {results['win_rate']:.2f}%")
            logger.info(f"  Trades: {results['total_trades']}")

        return annual_results

    def market_regime_analysis(self):
        """Test on different market regimes (Bull/Bear/Sideways)"""
        logger.info("\n" + "="*70)
        logger.info("MARKET REGIME ANALYSIS")
        logger.info("="*70)

        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')
        df = self.timeframe_data[signal_tf]

        # Define regime periods based on BTC history
        regimes = {
            'Bull 2020-2021': ('2020-01-01', '2021-11-30'),
            'Bear 2022': ('2022-01-01', '2022-12-31'),
            'Recovery 2023': ('2023-01-01', '2023-12-31'),
            'Bull 2024': ('2024-01-01', '2024-12-31'),
        }

        regime_results = {}

        for regime_name, (start_date, end_date) in regimes.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"REGIME: {regime_name}")
            logger.info("="*70)

            # Filter data for this regime
            regime_mask = (df.index >= start_date) & (df.index <= end_date)
            regime_df = df[regime_mask]

            if len(regime_df) == 0:
                logger.warning(f"No data for regime {regime_name}, skipping")
                continue

            # Train on all data before this regime
            train_mask = df.index < start_date
            train_df = df[train_mask]

            if len(train_df) < 1000:
                logger.warning(f"Insufficient training data for {regime_name}, skipping")
                continue

            logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} samples)")
            logger.info(f"Test: {regime_df.index[0]} to {regime_df.index[-1]} ({len(regime_df)} samples)")

            # Calculate regime characteristics
            returns = regime_df['close'].pct_change()
            total_return = (regime_df['close'].iloc[-1] / regime_df['close'].iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(365 * 24 * 4)  # Annualized (15m data)

            logger.info(f"\nRegime Characteristics:")
            logger.info(f"  Total Return: {total_return:.2f}%")
            logger.info(f"  Volatility: {volatility:.2f}%")

            # Get features
            train_features = self.features[train_mask]
            train_target = self.target[train_mask]
            test_features = self.features[regime_mask]

            # Train
            logger.info(f"\nTraining on pre-regime data...")
            advanced_system = self._train_system(train_df, train_features, train_target)

            # Test
            logger.info(f"\nTesting on {regime_name}...")
            results = self._run_backtest(advanced_system, regime_df, test_features,
                                        regime_name.replace(' ', '_').replace('-', '_'))
            results['market_return'] = total_return
            results['market_volatility'] = volatility
            regime_results[regime_name] = results

        # Summary
        logger.info("\n" + "="*70)
        logger.info("REGIME SUMMARY")
        logger.info("="*70)

        for regime_name, results in regime_results.items():
            logger.info(f"\n{regime_name}:")
            logger.info(f"  Market Return: {results['market_return']:.2f}%")
            logger.info(f"  Strategy Return: {results['total_return']:.2%}")
            logger.info(f"  Sharpe: {results['sharpe_ratio']:.3f}")
            logger.info(f"  Max DD: {results['max_drawdown']:.2%}")
            logger.info(f"  Win Rate: {results['win_rate']:.2f}%")

        return regime_results

    def _train_system(self, df, features, target):
        """Train the trading system"""
        if self.use_advanced:
            # Set advanced config
            if 'advanced' not in self.config:
                self.config['advanced'] = {}
            self.config['advanced']['use_deep_learning'] = False  # Faster training
            self.config['advanced']['use_rl'] = False

            advanced_system = AdvancedTradingSystem(self.config)
            advanced_system.train(df, features, target, verbose=False)
            return advanced_system
        else:
            # Basic system would go here
            return None

    def _run_backtest(self, system, df, features, label):
        """Run backtest and return metrics"""
        # Generate signals
        signals = system.generate_signals(df, features)

        # Default params
        params = self._get_default_params()

        # Run backtest
        backtester = Backtester(self.config)
        equity_curve, trades = backtester.run(df, signals, params, verbose=False)

        # Get metrics
        metrics = backtester.get_metrics()

        # Print summary
        logger.info(f"\n{label.upper()} RESULTS:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"  Total Trades: {metrics['total_trades']}")

        # Save results
        results_dir = self.config.get('output', {}).get('results_dir', 'results')
        equity_curve.to_csv(os.path.join(results_dir, f'equity_{label}.csv'))

        return metrics

    def _get_default_params(self):
        """Get default parameters"""
        timeframes = self.config.get('timeframes', {}).get('all', [])
        params = {}

        for i, tf in enumerate(timeframes):
            params[f'weight_{tf}'] = 10.0 - i * 0.5

        indicators = ['rsi', 'macd', 'bollinger', 'stochastic', 'ema', 'volume', 'heiken_ashi']
        for ind in indicators:
            params[f'ind_weight_{ind}'] = 1.0

        params['rsi_oversold'] = 30
        params['rsi_overbought'] = 70
        params['stop_loss_atr'] = 2.0
        params['take_profit_atr'] = 4.0
        params['position_size'] = 0.05
        params['ml_confidence_threshold'] = 0.6
        params['fractal_score_threshold'] = 0.5

        return params


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward and Regime Analysis')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--use-advanced', action='store_true',
                       help='Use Advanced Trading System')
    parser.add_argument('--train-test', action='store_true',
                       help='Run train/test split analysis')
    parser.add_argument('--annual', action='store_true',
                       help='Run annual walk-forward analysis')
    parser.add_argument('--regime', action='store_true',
                       help='Run market regime analysis')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    if not ADVANCED_AVAILABLE and args.use_advanced:
        logger.error("Advanced system not available. Install dependencies:")
        logger.error("pip install hmmlearn lightgbm catboost")
        return

    # Create analyzer
    analyzer = WalkForwardAnalyzer(args.config, use_advanced=args.use_advanced)

    # Load data once
    analyzer.load_and_prepare_data()

    # Run requested analyses
    if args.all or args.train_test:
        analyzer.train_test_split_analysis(train_ratio=0.8)

    if args.all or args.annual:
        analyzer.annual_walk_forward()

    if args.all or args.regime:
        analyzer.market_regime_analysis()

    if not (args.all or args.train_test or args.annual or args.regime):
        logger.warning("No analysis selected. Use --all, --train-test, --annual, or --regime")
        logger.info("Example: python walk_forward_analysis.py --use-advanced --all")


if __name__ == '__main__':
    main()
