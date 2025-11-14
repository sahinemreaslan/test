#!/usr/bin/env python3
"""
Offline Model Training
Train the strategy on full historical CSV data (2018-2025) and save model
"""

import os
import sys
import pandas as pd
import pickle
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.timeframe_converter import TimeframeConverter
from src.features.feature_engineering import FeatureEngineer
from src.advanced.integrated_system import AdvancedTradingSystem

def train_and_save_model(csv_path: str, config_path: str, output_dir: str = "models"):
    """
    Train on full historical data and save model

    Args:
        csv_path: Path to historical CSV data
        config_path: Path to config file
        output_dir: Directory to save models
    """
    print("="*70)
    print("🎓 OFFLINE MODEL TRAINING")
    print("="*70)
    print(f"CSV Data: {csv_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}/")
    print("="*70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    print("\n📊 Loading historical data...")
    df = pd.read_csv(csv_path)

    # Normalize column names (lowercase)
    df.columns = df.columns.str.lower().str.strip()

    # Prepare data
    if 'open time' in df.columns:
        df['open time'] = pd.to_datetime(df['open time'])
        df = df.set_index('open time')
        df.index.name = 'timestamp'
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.index.name = 'timestamp'

    # Keep only OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]

    print(f"✅ Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    # Initialize components
    print("\n🔧 Initializing components...")
    feature_eng = FeatureEngineer(config)
    advanced_system = AdvancedTradingSystem(config)

    # Convert to multiple timeframes
    print("\n⏱️ Converting to multiple timeframes...")
    tf_converter = TimeframeConverter(df)
    timeframes = config.get('timeframes', {}).get('all', [
        '3M', '1M', '1W', '1D', '12h', '8h', '4h', '2h', '1h', '30m', '15m'
    ])
    all_timeframes_raw = tf_converter.convert_all_timeframes(timeframes)

    # Process each timeframe
    print("\n🔬 Processing indicators for each timeframe...")
    all_timeframes = {}
    for tf, tf_df in all_timeframes_raw.items():
        print(f"  Processing {tf}...")
        all_timeframes[tf] = feature_eng.process_single_timeframe(tf_df, tf)

    # Create features
    print("\n🧬 Creating multi-timeframe features...")
    reference_tf = config.get('timeframes', {}).get('signal', '15m')
    feature_df = feature_eng.create_multi_timeframe_features(all_timeframes, reference_tf)

    # Prepare ML dataset
    print("\n📚 Preparing ML dataset...")
    features, target = feature_eng.prepare_ml_dataset(
        feature_df,
        target_method='forward_returns',
        target_horizon=1,
        target_threshold=0.001
    )

    print(f"✅ Features: {features.shape}")
    print(f"✅ Target distribution: {target.value_counts().to_dict()}")

    # Train model
    print("\n🎓 Training advanced system...")
    reference_ohlcv = all_timeframes[reference_tf]
    advanced_system.train(reference_ohlcv, features, target, verbose=True)

    # Save models
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n💾 Saving models...")

    # Save advanced system (contains all models)
    model_path = f"{output_dir}/advanced_system_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(advanced_system, f)
    print(f"✅ Saved: {model_path}")

    # Save latest symlink
    latest_path = f"{output_dir}/advanced_system_latest.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(advanced_system, f)
    print(f"✅ Saved: {latest_path}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'data_path': csv_path,
        'data_range': f"{df.index[0]} to {df.index[-1]}",
        'num_samples': len(features),
        'num_features': len(features.columns),
        'target_distribution': target.value_counts().to_dict(),
        'config': config
    }

    metadata_path = f"{output_dir}/model_metadata_{timestamp}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved: {metadata_path}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {model_path}")
    print(f"Use in live trading: python live_trader.py --model {latest_path}")
    print("="*70)

    return model_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model on historical data')
    parser.add_argument('--csv', type=str,
                       default='../btc_15m_data_2018_to_2025.csv',
                       help='Path to historical CSV data')
    parser.add_argument('--config', type=str,
                       default='config_live.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str,
                       default='../models',
                       help='Output directory for models')

    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    # Train and save
    train_and_save_model(args.csv, args.config, args.output)
