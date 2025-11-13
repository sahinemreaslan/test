"""
Reinforcement Learning Trading Agent using PPO

Learns to trade directly by maximizing risk-adjusted returns.

References:
- Schulman et al. (2017) - Proximal Policy Optimization
- Théate & Ernst (2021) - Deep RL for trading
- Zhang et al. (2022) - Financial trading with RL
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agent

    State: Market features + current position + PnL
    Action: [position_size, hold/buy/sell]
    Reward: Sharpe ratio + transaction costs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001,
        max_position_size: float = 0.1,
        reward_metric: str = 'sharpe'
    ):
        """
        Initialize trading environment

        Args:
            df: DataFrame with OHLCV data
            features: DataFrame with engineered features
            initial_capital: Starting capital
            commission: Commission rate
            max_position_size: Max position as fraction of capital
            reward_metric: 'sharpe', 'sortino', or 'profit'
        """
        super().__init__()

        self.df = df
        self.features = features
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        self.reward_metric = reward_metric

        # Align indices
        common_idx = df.index.intersection(features.index)
        self.df = df.loc[common_idx]
        self.features = features.loc[common_idx]

        # Environment state
        self.current_step = 0
        self.done = False
        self.capital = initial_capital
        self.position = 0  # BTC amount
        self.position_value = 0
        self.trades = []
        self.equity_history = []

        # Define action and observation spaces
        # Actions: [position_size (-1 to 1), action_type (0=hold, 1=buy, 2=sell)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0]),
            high=np.array([1.0, 2]),
            dtype=np.float32
        )

        # Observation: features + position info
        num_features = len(features.columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features + 3,),  # +3 for position, capital, pnl
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.equity_history = [self.initial_capital]

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute action

        Args:
            action: [position_size, action_type]

        Returns:
            observation, reward, done, truncated, info
        """
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Parse action
        position_size = np.clip(action[0], -1, 1)  # -1=full short, 1=full long
        action_type = int(np.clip(action[1], 0, 2))  # 0=hold, 1=buy, 2=sell

        # Get current price
        current_price = self.df.iloc[self.current_step]['close']

        # Execute trade
        reward = 0
        if action_type == 1:  # Buy
            reward = self._execute_buy(position_size, current_price)
        elif action_type == 2:  # Sell
            reward = self._execute_sell(current_price)

        # Update position value
        if self.position != 0:
            self.position_value = self.position * current_price

        # Calculate total equity
        total_equity = self.capital + self.position_value
        self.equity_history.append(total_equity)

        # Move to next step
        self.current_step += 1

        # Check if done
        if self.current_step >= len(self.df) - 1:
            self.done = True
            # Close any open position
            if self.position != 0:
                self._execute_sell(current_price)

        # Calculate reward based on metric
        if self.reward_metric == 'sharpe':
            reward = self._calculate_sharpe_reward()
        elif self.reward_metric == 'sortino':
            reward = self._calculate_sortino_reward()
        else:  # profit
            reward = (total_equity - self.initial_capital) / self.initial_capital

        return self._get_observation(), reward, self.done, False, {}

    def _execute_buy(self, position_size: float, price: float) -> float:
        """Execute buy order"""
        # Calculate position
        max_investment = self.capital * self.max_position_size
        investment = max_investment * abs(position_size)

        # Commission
        commission_cost = investment * self.commission

        # Buy BTC
        btc_bought = (investment - commission_cost) / price

        # Update state
        self.position += btc_bought
        self.capital -= investment

        # Record trade
        self.trades.append({
            'step': self.current_step,
            'type': 'buy',
            'price': price,
            'amount': btc_bought,
            'cost': investment
        })

        return -commission_cost  # Penalty for transaction cost

    def _execute_sell(self, price: float) -> float:
        """Execute sell order"""
        if self.position == 0:
            return 0

        # Sell all position
        proceeds = self.position * price

        # Commission
        commission_cost = proceeds * self.commission
        proceeds -= commission_cost

        # Calculate PnL
        pnl = proceeds - (self.position * self.trades[-1]['price'] if self.trades else 0)

        # Update state
        self.capital += proceeds
        self.position = 0
        self.position_value = 0

        # Record trade
        self.trades.append({
            'step': self.current_step,
            'type': 'sell',
            'price': price,
            'amount': -self.position,
            'proceeds': proceeds,
            'pnl': pnl
        })

        return pnl / self.initial_capital  # Normalized PnL

    def _get_observation(self):
        """Get current state observation"""
        # Market features
        feature_vector = self.features.iloc[self.current_step].values

        # Position info
        total_equity = self.capital + self.position_value
        position_info = np.array([
            self.position / 10,  # Normalize
            self.capital / self.initial_capital,
            total_equity / self.initial_capital - 1  # PnL
        ])

        # Combine
        observation = np.concatenate([feature_vector, position_info])

        return observation.astype(np.float32)

    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe ratio as reward"""
        if len(self.equity_history) < 2:
            return 0

        returns = np.diff(self.equity_history) / self.equity_history[:-1]

        if len(returns) < 2 or returns.std() == 0:
            return 0

        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        return sharpe

    def _calculate_sortino_reward(self) -> float:
        """Calculate Sortino ratio as reward"""
        if len(self.equity_history) < 2:
            return 0

        returns = np.diff(self.equity_history) / self.equity_history[:-1]
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino


class TradingRLAgent:
    """Reinforcement Learning trading agent using PPO"""

    def __init__(self, config: Dict):
        """
        Initialize RL agent

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.env = None

    def train(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        total_timesteps: int = 100000,
        verbose: bool = True
    ):
        """
        Train RL agent

        Args:
            df: OHLCV DataFrame
            features: Feature DataFrame
            total_timesteps: Training steps
            verbose: Print progress
        """
        logger.info("Training PPO agent...")

        # Create environment
        self.env = DummyVecEnv([
            lambda: TradingEnvironment(
                df=df,
                features=features,
                initial_capital=self.config.get('backtesting', {}).get('initial_capital', 10000),
                commission=self.config.get('backtesting', {}).get('commission', 0.001)
            )
        ])

        # Create PPO model
        self.model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1 if verbose else 0
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=verbose
        )

        logger.info("PPO training complete")

    def predict(self, observation):
        """Predict action given observation"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, filepath: str):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.save(filepath)
        logger.info(f"RL model saved to {filepath}")

    def load(self, filepath: str):
        """Load model"""
        self.model = PPO.load(filepath)
        logger.info(f"RL model loaded from {filepath}")
