"""
Advanced Trading System Components

Research-grade implementations:
- Market Regime Detection (HMM)
- Ensemble Learning (LightGBM, CatBoost)
- Attention Mechanisms
- LSTM/Transformer Models
- Reinforcement Learning (PPO)
- Advanced Risk Management
"""

from .market_regime import MarketRegimeDetector
from .ensemble_models import EnsemblePredictor
from .risk_management import KellyCriterion, AdvancedRiskMetrics, DynamicPositionSizer
from .integrated_system import AdvancedTradingSystem

__all__ = [
    'MarketRegimeDetector',
    'EnsemblePredictor',
    'KellyCriterion',
    'AdvancedRiskMetrics',
    'DynamicPositionSizer',
    'AdvancedTradingSystem'
]

__version__ = "1.0.0"
