"""
Dashboard Data Manager
Manages trading data for real-time visualization
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path


class DashboardDataManager:
    """Manages data persistence for dashboard"""

    def __init__(self, data_dir: str = "data"):
        """Initialize data manager

        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # File paths
        self.trades_file = self.data_dir / "trades.json"
        self.signals_file = self.data_dir / "signals.json"
        self.performance_file = self.data_dir / "performance.json"
        self.status_file = self.data_dir / "bot_status.json"

        # Initialize files if they don't exist
        self._init_files()

    def _init_files(self):
        """Initialize data files if they don't exist"""
        if not self.trades_file.exists():
            self._save_json(self.trades_file, [])

        if not self.signals_file.exists():
            self._save_json(self.signals_file, [])

        if not self.performance_file.exists():
            self._save_json(self.performance_file, {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'start_balance': 0.0,
                'current_balance': 0.0,
                'last_updated': datetime.now().isoformat()
            })

        if not self.status_file.exists():
            self._save_json(self.status_file, {
                'running': False,
                'last_check': None,
                'current_price': 0.0,
                'current_regime': 'Unknown',
                'open_position': None,
                'last_signal': 'HOLD',
                'last_confidence': 0.0
            })

    def _save_json(self, filepath: Path, data):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_json(self, filepath: Path):
        """Load data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    # ==================== Trades ====================

    def add_trade(self, trade: Dict):
        """Add a new trade

        Args:
            trade: Trade data dictionary
                {
                    'timestamp': ISO timestamp,
                    'type': 'OPEN' or 'CLOSE',
                    'side': 'LONG' or 'SHORT',
                    'entry_price': float,
                    'exit_price': float (if CLOSE),
                    'quantity': float,
                    'pnl': float (if CLOSE),
                    'pnl_pct': float (if CLOSE),
                    'regime': str,
                    'confidence': float
                }
        """
        trades = self._load_json(self.trades_file) or []
        trade['timestamp'] = datetime.now().isoformat()
        trades.append(trade)

        # Keep last 1000 trades
        if len(trades) > 1000:
            trades = trades[-1000:]

        self._save_json(self.trades_file, trades)

    def get_trades(self) -> List[Dict]:
        """Get all trades"""
        return self._load_json(self.trades_file) or []

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        trades = self.get_trades()
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    # ==================== Signals ====================

    def add_signal(self, signal: Dict):
        """Add a new signal

        Args:
            signal: Signal data dictionary
                {
                    'timestamp': ISO timestamp,
                    'signal': 1, 0, or -1,
                    'signal_name': 'BUY', 'HOLD', or 'SELL',
                    'price': float,
                    'confidence': float,
                    'regime': str
                }
        """
        signals = self._load_json(self.signals_file) or []
        signal['timestamp'] = datetime.now().isoformat()
        signals.append(signal)

        # Keep last 5000 signals
        if len(signals) > 5000:
            signals = signals[-5000:]

        self._save_json(self.signals_file, signals)

    def get_signals(self) -> List[Dict]:
        """Get all signals"""
        return self._load_json(self.signals_file) or []

    def get_signals_df(self) -> pd.DataFrame:
        """Get signals as DataFrame"""
        signals = self.get_signals()
        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    # ==================== Performance ====================

    def update_performance(self, metrics: Dict):
        """Update performance metrics

        Args:
            metrics: Performance metrics dictionary
        """
        perf = self._load_json(self.performance_file) or {}
        perf.update(metrics)
        perf['last_updated'] = datetime.now().isoformat()
        self._save_json(self.performance_file, perf)

    def get_performance(self) -> Dict:
        """Get performance metrics"""
        return self._load_json(self.performance_file) or {}

    def calculate_performance(self):
        """Calculate performance metrics from trades"""
        trades_df = self.get_trades_df()

        if trades_df.empty:
            return

        # Filter closed trades
        closed_trades = trades_df[trades_df['type'] == 'CLOSE']

        if closed_trades.empty:
            return

        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] < 0])

        total_pnl = closed_trades['pnl'].sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Sharpe ratio (simplified)
        if len(closed_trades) > 1:
            returns = closed_trades['pnl_pct'] / 100
            sharpe = (returns.mean() / returns.std()) * (365 ** 0.5) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative_pnl = closed_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max)
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Update performance
        self.update_performance({
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'total_pnl': float(total_pnl),
            'win_rate': float(win_rate),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown)
        })

    # ==================== Status ====================

    def update_status(self, status: Dict):
        """Update bot status

        Args:
            status: Status dictionary
        """
        current_status = self._load_json(self.status_file) or {}
        current_status.update(status)
        current_status['last_updated'] = datetime.now().isoformat()
        self._save_json(self.status_file, current_status)

    def get_status(self) -> Dict:
        """Get bot status"""
        return self._load_json(self.status_file) or {}

    # ==================== Utilities ====================

    def clear_all_data(self):
        """Clear all data (useful for testing)"""
        self._init_files()

    def export_data(self, output_dir: str):
        """Export all data to a directory

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Export to CSV
        trades_df = self.get_trades_df()
        if not trades_df.empty:
            trades_df.to_csv(output_path / "trades.csv", index=False)

        signals_df = self.get_signals_df()
        if not signals_df.empty:
            signals_df.to_csv(output_path / "signals.csv", index=False)

        # Export performance
        perf = self.get_performance()
        with open(output_path / "performance.json", 'w') as f:
            json.dump(perf, f, indent=2)
