"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess OHLCV data"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load CSV data and prepare for analysis"""
        logger.info(f"Loading data from {self.file_path}")

        # Read CSV
        df = pd.read_csv(self.file_path)

        # Rename columns for consistency
        column_mapping = {
            'Open time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Close time': 'close_time',
            'Quote asset volume': 'quote_volume',
            'Number of trades': 'num_trades',
            'Taker buy base asset volume': 'taker_buy_volume',
            'Taker buy quote asset volume': 'taker_buy_quote_volume'
        }

        df = df.rename(columns=column_mapping)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any NaN values
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # Sort by timestamp
        df = df.sort_index()

        # Add basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['range']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Upper and lower wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        self.df = df

        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        return df

    def get_ohlcv(self) -> pd.DataFrame:
        """Get OHLCV data"""
        if self.df is None:
            self.load_data()

        return self.df[['open', 'high', 'low', 'close', 'volume']].copy()

    def get_data_info(self) -> Dict:
        """Get information about the loaded data"""
        if self.df is None:
            self.load_data()

        return {
            'total_candles': len(self.df),
            'start_date': str(self.df.index[0]),
            'end_date': str(self.df.index[-1]),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'price_range': {
                'min': float(self.df['low'].min()),
                'max': float(self.df['high'].max()),
                'mean': float(self.df['close'].mean())
            }
        }
