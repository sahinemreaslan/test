"""
Helper utilities for the trading system
"""

import yaml
import os
import logging
from typing import Dict
from pathlib import Path


class ConfigLoader:
    """Load and validate configuration"""

    @staticmethod
    def load(config_path: str) -> Dict:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def save(config: Dict, config_path: str):
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary
            config_path: Path to save config
        """
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def setup_logging(verbose: bool = True, log_file: str = None):
    """
    Setup logging configuration

    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.INFO if verbose else logging.WARNING

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_output_dirs(config: Dict):
    """
    Create output directories

    Args:
        config: Configuration dictionary
    """
    output_config = config.get('output', {})

    dirs = [
        output_config.get('results_dir', 'results'),
        output_config.get('models_dir', 'models'),
        output_config.get('plots_dir', 'plots')
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_number(num: float, precision: int = 2) -> str:
    """Format number for display"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_periods_per_year(timeframe: str) -> int:
    """
    Calculate number of periods per year for a given timeframe

    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '1D')

    Returns:
        Number of periods per year
    """
    # Map timeframe to minutes
    minutes_map = {
        '5m': 5,
        '10m': 10,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '8h': 480,
        '12h': 720,
        '1D': 1440,
        '1W': 10080,
        '1M': 43200,
        '3M': 129600
    }

    minutes = minutes_map.get(timeframe, 5)
    periods_per_year = int(365 * 24 * 60 / minutes)

    return periods_per_year
