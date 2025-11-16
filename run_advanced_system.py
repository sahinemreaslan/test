"""
Advanced Trading System - Main Runner

Demonstrates the research-grade Level 3 system with all components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import argparse
import logging
from src.utils.helpers import ConfigLoader, setup_logging
from src.data.data_loader import DataLoader
from src.data.timeframe_converter import TimeframeConverter
from src.features.feature_engineering import FeatureEngineer
from src.advanced.integrated_system import AdvancedTradingSystem

logger = logging.getLogger(__name__)


def main():
    """Run advanced trading system"""
    parser = argparse.ArgumentParser(description='Advanced Trading System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--use-rl', action='store_true',
                       help='Enable Reinforcement Learning')
    parser.add_argument('--use-dl', action='store_true',
                       help='Enable Deep Learning models')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    logger.info("="*70)
    logger.info("ADVANCED TRADING SYSTEM - LEVEL 3 (RESEARCH-GRADE)")
    logger.info("="*70)

    # Load config
    config = ConfigLoader.load(args.config)

    # Enable advanced features
    if not 'advanced' in config:
        config['advanced'] = {}

    config['advanced']['use_rl'] = args.use_rl
    config['advanced']['use_deep_learning'] = args.use_dl

    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data...")
    data_config = config.get('data', {})
    file_path = data_config.get('file_path', 'btc_15m_data_2018_to_2025.csv')

    data_loader = DataLoader(file_path)
    base_data = data_loader.load_data()

    logger.info(f"Loaded {len(base_data)} candles")

    # Step 2: Convert timeframes
    logger.info("\n[Step 2/6] Converting to multiple timeframes...")
    timeframes = config.get('timeframes', {}).get('all', [])
    base_timeframe = config.get('data', {}).get('base_timeframe', '15m')

    converter = TimeframeConverter(base_data, base_timeframe)
    timeframe_data = converter.convert_all_timeframes(timeframes)

    logger.info(f"Converted to {len(timeframe_data)} timeframes")

    # Step 3: Engineer features
    logger.info("\n[Step 3/6] Engineering features...")
    feature_engineer = FeatureEngineer(config)

    # Process each timeframe
    for tf, df in timeframe_data.items():
        logger.info(f"  Processing {tf}...")
        df = feature_engineer.process_single_timeframe(df, tf)
        timeframe_data[tf] = df

    # Create multi-timeframe features
    signal_tf = config.get('timeframes', {}).get('signal', '15m')
    features = feature_engineer.create_multi_timeframe_features(
        timeframe_data,
        reference_tf=signal_tf
    )

    # Create target
    target = feature_engineer.create_target_variable(
        timeframe_data[signal_tf],
        method='forward_returns',
        horizon=1,
        threshold=0.001
    )

    # Prepare dataset
    features, target = feature_engineer.prepare_ml_dataset(features)

    logger.info(f"Features: {features.shape}, Target: {target.shape}")

    # Step 4: Train advanced system
    logger.info("\n[Step 4/6] Training Advanced System...")

    advanced_system = AdvancedTradingSystem(config)

    metrics = advanced_system.train(
        df=timeframe_data[signal_tf],
        features=features,
        target=target,
        verbose=True
    )

    # Step 5: Generate signals
    logger.info("\n[Step 5/6] Generating signals...")

    signals = advanced_system.generate_signals(
        df=timeframe_data[signal_tf],
        features=features
    )

    logger.info(f"Signals generated: Buy={( signals==1).sum()}, Sell={(signals==-1).sum()}")

    # Step 6: Display summary
    logger.info("\n[Step 6/6] System Summary...")

    summary = advanced_system.get_system_summary()

    logger.info("\n" + "="*70)
    logger.info("SYSTEM SUMMARY")
    logger.info("="*70)
    logger.info(f"Status: {'Trained' if summary['trained'] else 'Not Trained'}")
    logger.info(f"Current Market Regime: {summary['current_regime']}")

    logger.info("\nComponents:")
    for comp, desc in summary['components'].items():
        logger.info(f"  {comp}: {desc}")

    logger.info("\n" + "="*70)
    logger.info("READY FOR TRADING!")
    logger.info("="*70)

    logger.info("\nTo run backtest with advanced system:")
    logger.info("  python main.py --use-advanced")


if __name__ == '__main__':
    main()
