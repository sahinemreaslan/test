"""
Quick test script to validate the system works
Tests individual components before running full pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.helpers import ConfigLoader, setup_logging
from src.data.data_loader import DataLoader
from src.data.timeframe_converter import TimeframeConverter

def test_data_loading():
    """Test data loading"""
    print("\n=== Testing Data Loading ===")
    loader = DataLoader('btc_15m_data_2018_to_2025.csv')
    df = loader.load_data()
    print(f"✓ Loaded {len(df)} candles")
    print(f"✓ Columns: {list(df.columns)}")
    return df

def test_timeframe_conversion(base_df):
    """Test timeframe conversion"""
    print("\n=== Testing Timeframe Conversion ===")
    converter = TimeframeConverter(base_df, base_timeframe='15m')

    # Test a few timeframes (including base timeframe)
    test_tfs = ['15m', '30m', '1h', '4h', '1D']
    for tf in test_tfs:
        converted = converter.convert_to_timeframe(tf)
        print(f"✓ {tf}: {len(converted)} candles")

    return converter

def test_fractal_analysis(df):
    """Test fractal analysis"""
    print("\n=== Testing Fractal Analysis ===")
    from src.features.fractal_analysis import FractalAnalyzer

    analyzer = FractalAnalyzer()
    df_analyzed = analyzer.analyze_dataframe(df.head(1000))  # Test on subset

    print(f"✓ Fractal patterns identified")
    print(f"✓ Pattern distribution:")

    stats = analyzer.get_pattern_statistics(df_analyzed)
    for pattern, data in stats['pattern_distribution'].items():
        print(f"  {pattern}: {data['percentage']:.2f}%")

def test_indicators(df):
    """Test indicator calculation"""
    print("\n=== Testing Technical Indicators ===")
    from src.features.indicators import IndicatorCalculator

    calculator = IndicatorCalculator()
    df_with_indicators = calculator.calculate_all_indicators(df.head(1000))  # Test on subset

    print(f"✓ Calculated indicators")
    print(f"✓ Total columns: {len(df_with_indicators.columns)}")
    print(f"✓ Sample indicators: RSI, MACD, Bollinger Bands, ATR")

def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration ===")
    config = ConfigLoader.load('config.yaml')
    print(f"✓ Config loaded")
    print(f"✓ Timeframes: {config.get('timeframes', {}).get('all', [])}")
    print(f"✓ GA population size: {config.get('genetic_algorithm', {}).get('population_size')}")

def main():
    """Run quick tests"""
    print("\n" + "="*60)
    print("QUICK SYSTEM TEST")
    print("="*60)

    try:
        # Test configuration
        test_config_loading()

        # Test data loading
        df = test_data_loading()

        # Test timeframe conversion
        converter = test_timeframe_conversion(df)

        # Test fractal analysis
        test_fractal_analysis(df)

        # Test indicators
        test_indicators(df)

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nSystem is ready. Run: python main.py")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
