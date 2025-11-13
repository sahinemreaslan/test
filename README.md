# Fractal Multi-Timeframe Trading Strategy

Advanced algorithmic trading system combining fractal pattern analysis, technical indicators, XGBoost machine learning, and genetic algorithm optimization.

## Overview

This system implements a sophisticated trading strategy based on:

1. **Fraktal Mum İlişkisi (Fractal Candle Relationships)**
   - HHHL (Higher High Higher Low) - Bullish Power
   - HLLH (Lower High Lower Low) - Bearish Power
   - Inside Bar - Consolidation
   - Outside Bar - Volatility Expansion

2. **Multi-Timeframe Analysis**
   - Analyzes 13 timeframes: 3M, 1M, 1W, 1D, 12h, 8h, 4h, 2h, 1h, 30m, 15m, 10m, 5m
   - Each timeframe provides context for smaller timeframes
   - Weighted according to importance

3. **Technical Indicators**
   - RSI (multiple periods)
   - MACD
   - Bollinger Bands
   - Stochastic Oscillator
   - ATR (for volatility)
   - EMAs (9, 21, 50, 100, 200)
   - Volume indicators
   - Heiken Ashi

4. **Machine Learning**
   - XGBoost classifier for signal prediction
   - Time-series cross-validation
   - Feature importance analysis

5. **Genetic Algorithm Optimization**
   - Optimizes 50+ parameters
   - Maximizes risk-adjusted returns (Sharpe, Calmar, Sortino ratios)
   - Tournament selection with elitism

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
├── test_quick.py              # Quick validation test
├── btc_15m_data_2018_to_2025.csv  # BTC 15m OHLCV data
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Load and preprocess data
│   │   └── timeframe_converter.py  # Multi-timeframe conversion
│   │
│   ├── features/
│   │   ├── fractal_analysis.py     # Fractal pattern detection
│   │   ├── indicators.py           # Technical indicators
│   │   └── feature_engineering.py  # Feature pipeline
│   │
│   ├── models/
│   │   └── xgboost_model.py       # XGBoost ML model
│   │
│   ├── optimization/
│   │   ├── genetic_algorithm.py   # GA optimizer
│   │   └── fitness_functions.py   # Fitness metrics
│   │
│   ├── backtesting/
│   │   ├── backtester.py          # Backtest engine
│   │   └── metrics.py             # Performance metrics
│   │
│   ├── strategy/
│   │   └── fractal_strategy.py    # Main strategy logic
│   │
│   └── utils/
│       └── helpers.py             # Utility functions
│
├── results/                    # Backtest results (created)
├── models/                     # Saved models (created)
└── plots/                      # Visualization plots (created)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Test

Validate the system works:

```bash
python test_quick.py
```

### Full Pipeline

Run the complete system (ML + GA optimization):

```bash
python main.py
```

### Options

```bash
# Skip ML training (use rule-based signals only)
python main.py --no-ml

# Skip GA optimization (use default parameters)
python main.py --no-ga

# Both
python main.py --no-ml --no-ga

# Custom config
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **Data**: Input file path, base timeframe
- **Timeframes**: Which timeframes to analyze
- **Indicators**: Indicator parameters (RSI periods, MACD settings, etc.)
- **Genetic Algorithm**: Population size, generations, mutation rate, parameter bounds
- **XGBoost**: Model hyperparameters, cross-validation settings
- **Backtesting**: Initial capital, commission, slippage, risk limits
- **Output**: Where to save results, models, plots

## Strategy Philosophy

The strategy is built on the principle that **every candle has a relationship with the previous candle**, creating a fractal pattern. By analyzing these patterns across multiple timeframes simultaneously, we can identify high-probability trading opportunities.

### Signal Generation

1. **Fractal Score**: Weighted average of fractal patterns across all timeframes
2. **Indicator Score**: Weighted combination of technical indicators
3. **Alignment Score**: How well all timeframes agree on direction
4. **ML Confidence**: XGBoost probability for signal confirmation

A buy signal is generated when:
- Fractal consensus is bullish (HHHL patterns dominant)
- Indicators confirm (RSI, MACD, EMAs aligned)
- ML model confidence > threshold
- Multiple timeframes in agreement

### Risk Management

- **Position Sizing**: Configurable % of capital per trade
- **Stop Loss**: ATR-based dynamic stops
- **Take Profit**: ATR-based dynamic targets
- **Max Drawdown**: Automatic shutdown if drawdown exceeds limit

## Performance Metrics

The system tracks:

- **Return Metrics**: Total return, mean return, CAGR
- **Risk Metrics**: Volatility, max drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, expectancy, avg win/loss
- **Drawdown Analysis**: Max DD duration, recovery time

## Output

After running, you'll find:

- `results/equity_curve.csv`: Equity curve over time
- `models/xgboost_model.pkl`: Trained ML model
- Console output with comprehensive performance report

## Advanced Features

### Feature Engineering

The system generates 200+ features including:
- Fractal patterns for each timeframe
- Pattern streaks and momentum
- Trend strength and consistency
- Cross-timeframe correlations
- Indicator signals and divergences

### Genetic Algorithm

Optimizes:
- Timeframe weights (importance of each timeframe)
- Indicator weights
- Entry/exit thresholds
- Risk management parameters

Uses:
- Tournament selection
- Two-point crossover
- Gaussian mutation with bounds
- Elitism (preserves best individuals)

### XGBoost Model

- Handles class imbalance with scale_pos_weight
- Time-series cross-validation
- Early stopping to prevent overfitting
- Feature importance for interpretability

## Development

The codebase is modular and extensible:

- Add new indicators in `src/features/indicators.py`
- Add new fractal patterns in `src/features/fractal_analysis.py`
- Modify strategy logic in `src/strategy/fractal_strategy.py`
- Add new fitness functions in `src/optimization/fitness_functions.py`

## Notes

- The system is designed for Bitcoin 15m data but can work with any cryptocurrency or timeframe
- Backtesting includes realistic commission and slippage
- All timestamps are preserved for accurate time-series analysis
- The system uses forward-filling for higher timeframe data alignment

## License

This is a research and educational project.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without understanding the risks. Past performance does not guarantee future results.
