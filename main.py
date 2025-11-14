"""
Fractal Multi-Timeframe Trading Strategy - Main Execution Script

This script orchestrates the entire trading system:
1. Load and prepare data
2. Convert to multiple timeframes
3. Calculate fractal patterns and indicators
4. Engineer features
5. Train XGBoost model
6. Optimize parameters with genetic algorithm
7. Backtest strategy
8. Generate reports
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.helpers import ConfigLoader, setup_logging, create_output_dirs
from src.data.data_loader import DataLoader
from src.data.timeframe_converter import TimeframeConverter
from src.features.fractal_analysis import FractalAnalyzer
from src.features.indicators import IndicatorCalculator
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostPredictor
from src.optimization.genetic_algorithm import GeneticOptimizer
from src.backtesting.backtester import Backtester
from src.strategy.fractal_strategy import FractalMultiTimeframeStrategy

logger = logging.getLogger(__name__)

# Try to import advanced system (optional dependency)
try:
    from src.advanced.integrated_system import AdvancedTradingSystem
    ADVANCED_AVAILABLE = True
except ImportError as e:
    ADVANCED_AVAILABLE = False
    logger.debug(f"Advanced system not available: {e}")


class TradingSystemOrchestrator:
    """Main orchestrator for the trading system"""

    def __init__(self, config_path: str, use_advanced: bool = False):
        """
        Initialize orchestrator

        Args:
            config_path: Path to configuration file
            use_advanced: Whether to use advanced trading system (Level 3)
        """
        self.config = ConfigLoader.load(config_path)
        create_output_dirs(self.config)
        self.use_advanced = use_advanced

        # Initialize components
        self.data_loader = None
        self.timeframe_converter = None
        self.feature_engineer = None
        self.ml_model = None
        self.ga_optimizer = None
        self.advanced_system = None

        # Data storage
        self.base_data = None
        self.timeframe_data = {}
        self.features = None
        self.target = None

    def run_full_pipeline(self, train_ml: bool = True, optimize_ga: bool = True):
        """
        Run the complete trading system pipeline

        Args:
            train_ml: Whether to train ML model
            optimize_ga: Whether to run genetic algorithm optimization
        """
        logger.info("="*70)
        if self.use_advanced:
            logger.info("FRACTAL MULTI-TIMEFRAME TRADING SYSTEM - LEVEL 3 (ADVANCED)")
        else:
            logger.info("FRACTAL MULTI-TIMEFRAME TRADING SYSTEM")
        logger.info("="*70)

        # If using advanced system, run advanced pipeline
        if self.use_advanced:
            if not ADVANCED_AVAILABLE:
                logger.error("Advanced system requested but dependencies not installed!")
                logger.error("Please install: pip install hmmlearn lightgbm catboost torch stable-baselines3 gymnasium")
                return
            self.run_advanced_pipeline(train_ml, optimize_ga)
            return

        # Step 1: Load data
        logger.info("\n[1/8] Loading data...")
        self.load_data()

        # Step 2: Convert timeframes
        logger.info("\n[2/8] Converting to multiple timeframes...")
        self.convert_timeframes()

        # Step 3: Calculate indicators and fractal patterns
        logger.info("\n[3/8] Calculating indicators and fractal patterns...")
        self.process_timeframes()

        # Step 4: Engineer features
        logger.info("\n[4/8] Engineering features...")
        self.engineer_features()

        # Step 5: Train ML model
        if train_ml:
            logger.info("\n[5/8] Training XGBoost model...")
            self.train_ml_model()
        else:
            logger.info("\n[5/8] Skipping ML model training")
            self.ml_model = None

        # Step 6: Optimize with genetic algorithm
        if optimize_ga:
            logger.info("\n[6/8] Optimizing with genetic algorithm...")
            best_params, best_metrics = self.optimize_with_ga()
        else:
            logger.info("\n[6/8] Using default parameters (no GA optimization)")
            best_params = self.get_default_params()
            best_metrics = None

        # Step 7: Backtest with best parameters
        logger.info("\n[7/8] Running final backtest...")
        self.run_final_backtest(best_params)

        # Step 8: Generate report
        logger.info("\n[8/8] Generating report...")
        self.generate_report(best_params, best_metrics)

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*70)

    def load_data(self):
        """Load base data"""
        data_config = self.config.get('data', {})
        file_path = data_config.get('file_path', 'btc_15m_data_2018_to_2025.csv')

        self.data_loader = DataLoader(file_path)
        self.base_data = self.data_loader.load_data()

        info = self.data_loader.get_data_info()
        logger.info(f"Loaded {info['total_candles']} candles")
        logger.info(f"Date range: {info['start_date']} to {info['end_date']}")
        logger.info(f"Price range: ${info['price_range']['min']:.2f} - ${info['price_range']['max']:.2f}")

    def convert_timeframes(self):
        """Convert to multiple timeframes"""
        timeframes = self.config.get('timeframes', {}).get('all', [])

        self.timeframe_converter = TimeframeConverter(self.base_data)
        self.timeframe_data = self.timeframe_converter.convert_all_timeframes(timeframes)

        logger.info(f"Converted to {len(self.timeframe_data)} timeframes")

    def process_timeframes(self):
        """Process all timeframes with indicators and fractal analysis"""
        fractal_analyzer = FractalAnalyzer()
        indicator_calculator = IndicatorCalculator(self.config.get('indicators', {}))

        for tf, df in self.timeframe_data.items():
            logger.info(f"  Processing {tf}...")

            # Fractal analysis
            df = fractal_analyzer.analyze_dataframe(df)
            df = fractal_analyzer.get_fractal_features(df)

            # Technical indicators
            df = indicator_calculator.calculate_all_indicators(df)
            df = indicator_calculator.get_indicator_signals(df)

            self.timeframe_data[tf] = df

        logger.info("All timeframes processed")

    def engineer_features(self):
        """Engineer features from multi-timeframe data"""
        self.feature_engineer = FeatureEngineer(self.config)

        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')

        # Create multi-timeframe feature matrix
        self.features = self.feature_engineer.create_multi_timeframe_features(
            self.timeframe_data,
            reference_tf=signal_tf
        )

        # Create target variable
        self.target = self.feature_engineer.create_target_variable(
            self.timeframe_data[signal_tf],
            method='forward_returns',
            horizon=1,
            threshold=0.001
        )

        # Align features and target
        self.features, self.target = self.feature_engineer.prepare_ml_dataset(
            self.features,
            target_method='forward_returns',
            target_horizon=1,
            target_threshold=0.001
        )

        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Target shape: {self.target.shape}")
        logger.info(f"Target distribution: {self.target.value_counts().to_dict()}")

    def train_ml_model(self):
        """Train XGBoost model"""
        self.ml_model = XGBoostPredictor(self.config)

        metrics = self.ml_model.train(
            self.features,
            self.target,
            validation_split=0.2,
            use_time_series_split=True,
            verbose=True
        )

        logger.info(f"Model training complete. Metrics: {metrics}")

        # Show feature importance
        importance = self.ml_model.get_feature_importance(top_n=10)
        logger.info("\nTop 10 Most Important Features:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save model
        model_path = os.path.join(
            self.config.get('output', {}).get('models_dir', 'models'),
            'xgboost_model.pkl'
        )
        self.ml_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    def optimize_with_ga(self):
        """Optimize strategy parameters using genetic algorithm"""
        self.ga_optimizer = GeneticOptimizer(self.config)

        # Define backtest function for GA
        def backtest_func(data, params):
            strategy = FractalMultiTimeframeStrategy(params, self.ml_model)
            signals = strategy.generate_signals(self.features, use_ml=(self.ml_model is not None))

            backtester = Backtester(self.config)
            signal_tf = self.config.get('timeframes', {}).get('signal', '15m')
            equity_curve, trades = backtester.run(
                self.timeframe_data[signal_tf],
                signals,
                params,
                verbose=False
            )

            return equity_curve, trades

        # Run optimization
        best_params, best_metrics = self.ga_optimizer.optimize(
            backtest_func,
            data=None,  # Data is captured in closure
            verbose=True
        )

        logger.info("\nBest Parameters Found:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value:.4f}")

        logger.info("\nBest Metrics:")
        for key, value in best_metrics.items():
            logger.info(f"  {key}: {value}")

        return best_params, best_metrics

    def run_final_backtest(self, params):
        """Run final backtest with optimized parameters"""
        strategy = FractalMultiTimeframeStrategy(params, self.ml_model)
        signals = strategy.generate_signals(self.features, use_ml=(self.ml_model is not None))

        backtester = Backtester(self.config)
        signal_tf = self.config.get('timeframes', {}).get('signal', '15m')

        equity_curve, trades = backtester.run(
            self.timeframe_data[signal_tf],
            signals,
            params,
            verbose=True
        )

        # Print results
        backtester.print_results()

        # Save results
        results_dir = self.config.get('output', {}).get('results_dir', 'results')
        equity_curve.to_csv(os.path.join(results_dir, 'equity_curve.csv'))
        logger.info(f"Results saved to {results_dir}")

    def get_default_params(self):
        """Get default parameters (no GA optimization)"""
        # Create default parameters based on config
        timeframes = self.config.get('timeframes', {}).get('all', [])

        params = {}

        # Default timeframe weights (decreasing with smaller timeframes)
        for i, tf in enumerate(timeframes):
            params[f'weight_{tf}'] = 10.0 - i * 0.5

        # Default indicator weights
        indicators = ['rsi', 'macd', 'bollinger', 'stochastic', 'ema', 'volume', 'heiken_ashi']
        for ind in indicators:
            params[f'ind_weight_{ind}'] = 1.0

        # Default thresholds
        params['rsi_oversold'] = 30
        params['rsi_overbought'] = 70
        params['stop_loss_atr'] = 2.0
        params['take_profit_atr'] = 4.0
        params['position_size'] = 0.05
        params['ml_confidence_threshold'] = 0.6
        params['fractal_score_threshold'] = 0.5

        return params

    def generate_report(self, best_params, best_metrics):
        """Generate final report"""
        logger.info("\n" + "="*70)
        logger.info("FINAL REPORT")
        logger.info("="*70)

        logger.info("\nStrategy Configuration:")
        strategy = FractalMultiTimeframeStrategy(best_params, self.ml_model)
        summary = strategy.get_strategy_summary()
        logger.info(f"  ML Confidence Threshold: {summary['ml_confidence_threshold']:.2f}")
        logger.info(f"  Fractal Score Threshold: {summary['fractal_score_threshold']:.2f}")

        logger.info("\n  Top 5 Timeframe Weights:")
        for tf, weight in summary['top_timeframe_weights']:
            logger.info(f"    {tf}: {weight:.2f}")

        logger.info("\n  Top 5 Indicator Weights:")
        for ind, weight in summary['top_indicator_weights']:
            logger.info(f"    {ind}: {weight:.2f}")

        if best_metrics:
            logger.info("\nOptimization Metrics:")
            logger.info(f"  Total Return: {best_metrics.get('total_return', 0)*100:.2f}%")
            logger.info(f"  Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {best_metrics.get('max_drawdown', 0)*100:.2f}%")
            logger.info(f"  Win Rate: {best_metrics.get('win_rate', 0):.2f}%")
            logger.info(f"  Total Trades: {best_metrics.get('num_trades', 0)}")

        logger.info("\n" + "="*70)

    def run_advanced_pipeline(self, train_ml: bool = True, optimize_ga: bool = True):
        """
        Run the advanced trading system pipeline (Level 3)

        Args:
            train_ml: Whether to train models
            optimize_ga: Whether to run GA optimization (not used in advanced mode)
        """
        # Steps 1-4: Same as basic pipeline
        logger.info("\n[1/6] Loading data...")
        self.load_data()

        logger.info("\n[2/6] Converting to multiple timeframes...")
        self.convert_timeframes()

        logger.info("\n[3/6] Calculating indicators and fractal patterns...")
        self.process_timeframes()

        logger.info("\n[4/6] Engineering features...")
        self.engineer_features()

        # Step 5: Train advanced system
        if train_ml:
            logger.info("\n[5/6] Training Advanced System (Level 3)...")
            logger.info("  - Market Regime Detection (HMM)")
            logger.info("  - Ensemble Models (XGBoost + LightGBM + CatBoost)")
            logger.info("  - Deep Learning Models (LSTM + Transformers)")
            logger.info("  - Reinforcement Learning (PPO)")
            logger.info("  - Advanced Risk Management (Kelly, CVaR, etc.)")

            # Set advanced options in config
            if 'advanced' not in self.config:
                self.config['advanced'] = {}
            self.config['advanced']['use_deep_learning'] = True
            self.config['advanced']['use_rl'] = True

            self.advanced_system = AdvancedTradingSystem(self.config)

            # Get reference timeframe data for regime detection
            signal_tf = self.config.get('timeframes', {}).get('signal', '15m')

            # Train the system
            metrics = self.advanced_system.train(
                self.timeframe_data[signal_tf],
                self.features,
                self.target
            )

            logger.info("\nAdvanced System Training Metrics:")
            logger.info(f"  Ensemble Accuracy: {metrics.get('ensemble_accuracy', 0):.2%}")
            logger.info(f"  Ensemble AUC: {metrics.get('ensemble_auc', 0):.4f}")
            if 'dl_metrics' in metrics:
                logger.info(f"  Deep Learning Accuracy: {metrics['dl_metrics'].get('lstm_accuracy', 0):.2%}")
            if 'rl_metrics' in metrics:
                logger.info(f"  RL Training Complete: {metrics['rl_metrics'].get('trained', False)}")
        else:
            logger.info("\n[5/6] Skipping Advanced System training")

        # Step 6: Run backtest with advanced system
        logger.info("\n[6/6] Running backtest with Advanced System...")
        if self.advanced_system:
            signal_tf = self.config.get('timeframes', {}).get('signal', '15m')

            # Generate signals
            signals = self.advanced_system.generate_signals(
                self.timeframe_data[signal_tf],
                self.features
            )

            # Get default parameters for backtester
            params = self.get_default_params()

            # Run backtest
            backtester = Backtester(self.config)
            equity_curve, trades = backtester.run(
                self.timeframe_data[signal_tf],
                signals,
                params,
                verbose=True
            )

            # Print results
            backtester.print_results()

            # Save results
            results_dir = self.config.get('output', {}).get('results_dir', 'results')
            equity_curve.to_csv(os.path.join(results_dir, 'advanced_equity_curve.csv'))

            # Print system summary
            logger.info("\n" + "="*70)
            logger.info("ADVANCED SYSTEM SUMMARY")
            logger.info("="*70)
            summary = self.advanced_system.get_system_summary()
            logger.info(f"\nCurrent Market Regime: {summary['current_regime']}")
            logger.info(f"Regime Confidence: {summary['regime_confidence']:.2%}")
            logger.info(f"\nActive Components:")
            for component, status in summary['components'].items():
                logger.info(f"  {component}: {'✓' if status else '✗'}")

            logger.info(f"\nResults saved to {results_dir}")

        logger.info("\n" + "="*70)
        logger.info("ADVANCED PIPELINE COMPLETE!")
        logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fractal Multi-Timeframe Trading Strategy')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-ml', action='store_true',
                       help='Skip ML model training')
    parser.add_argument('--no-ga', action='store_true',
                       help='Skip genetic algorithm optimization')
    parser.add_argument('--use-advanced', action='store_true',
                       help='Use Advanced Trading System (Level 3) with ensemble models, deep learning, and RL')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Run pipeline
    orchestrator = TradingSystemOrchestrator(args.config, use_advanced=args.use_advanced)
    orchestrator.run_full_pipeline(
        train_ml=not args.no_ml,
        optimize_ga=not args.no_ga
    )


if __name__ == '__main__':
    main()
