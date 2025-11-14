"""
Bitcoin Live Trading Bot
Runs the fractal multi-timeframe strategy on Binance Futures
"""

import os
import sys
import time
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, Optional
from dotenv import load_dotenv
import colorlog

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_connector import BinanceConnector
from strategy_executor import StrategyExecutor


class LiveTradingBot:
    """Live trading bot for Binance Futures"""

    def __init__(self, config_path: str = "config_live.yaml", model_path: Optional[str] = None):
        """Initialize the trading bot

        Args:
            config_path: Path to config file
            model_path: Path to pre-trained model (optional)
        """
        # Setup logging
        self._setup_logging()

        # Load configuration
        self.config = self._load_config(config_path)
        self.model_path = model_path

        # Load API credentials
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("❌ BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file!")

        # Initialize components
        testnet = self.config.get('trading', {}).get('testnet', True)
        self.binance = BinanceConnector(api_key, api_secret, testnet=testnet)
        self.strategy = StrategyExecutor(self.config)

        # Trading parameters
        self.symbol = self.config.get('trading', {}).get('symbol', 'BTCUSDT')
        self.leverage = self.config.get('trading', {}).get('leverage', 3)
        self.check_interval = self.config.get('trading', {}).get('check_interval_seconds', 60)
        self.paper_trading = self.config.get('trading', {}).get('paper_trading', True)

        # State
        self.running = False
        self.last_check_time = None
        self.position_opened_at = None
        self.trades_count = 0

        logger.info("="*70)
        logger.info("🤖 BITCOIN LIVE TRADING BOT INITIALIZED")
        logger.info("="*70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"Testnet: {'✅ Yes (Fake money)' if testnet else '⚠️ NO (REAL MONEY!)'}")
        logger.info(f"Paper Trading: {'✅ Yes (No actual trades)' if self.paper_trading else '❌ No (Real trades!)'}")
        logger.info("="*70)

    def _setup_logging(self):
        """Setup colored logging"""
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))

        global logger
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        # Try live_trading directory first
        if not os.path.exists(config_path):
            config_path = os.path.join('live_trading', config_path)

        if not os.path.exists(config_path):
            # Load default config from parent directory
            parent_config = os.path.join('..', 'config.yaml')
            if os.path.exists(parent_config):
                logger.warning(f"Live config not found, using default: {parent_config}")
                config_path = parent_config

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def initialize(self):
        """Initialize trading (setup leverage, train strategy, etc.)"""
        logger.info("\n" + "="*70)
        logger.info("🚀 INITIALIZING TRADING BOT")
        logger.info("="*70)

        try:
            # Check balance
            balance = self.binance.get_account_balance()
            if balance['available_balance'] < 10:
                raise ValueError(f"❌ Insufficient balance: {balance['available_balance']} USDT")

            # Set leverage and margin type
            logger.info(f"\n⚙️ Setting up {self.symbol}...")
            self.binance.set_margin_type(self.symbol, "CROSSED")
            self.binance.set_leverage(self.symbol, self.leverage)

            # Check if using pre-trained model
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"\n📦 Loading pre-trained model from: {self.model_path}")
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.strategy.advanced_system = pickle.load(f)
                self.strategy.trained = True
                logger.info("✅ Pre-trained model loaded successfully!")
            else:
                if self.model_path:
                    logger.warning(f"⚠️ Model not found: {self.model_path}")
                    logger.info("🎓 Falling back to fresh training...")

                # Download historical data for training
                logger.info(f"\n📊 Downloading historical data...")
                historical_data = self.binance.get_historical_klines(
                    symbol=self.symbol,
                    interval="15m",
                    limit=1500  # ~15 days of 15m candles
                )

                if len(historical_data) < 300:
                    raise ValueError("❌ Not enough historical data for training")

                # Train strategy
                logger.info(f"\n🎓 Training strategy...")
                self.strategy.train_strategy(historical_data)

            logger.info("\n" + "="*70)
            logger.info("✅ INITIALIZATION COMPLETE!")
            logger.info("="*70)

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False

    def check_and_trade(self):
        """Main trading logic - check signal and execute if needed"""
        try:
            logger.info("\n" + "-"*70)
            logger.info(f"🔍 Checking market at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("-"*70)

            # Get current data
            current_data = self.binance.get_historical_klines(
                symbol=self.symbol,
                interval="15m",
                limit=500
            )

            if len(current_data) < 300:
                logger.warning("⚠️ Not enough data to generate signal")
                return

            # Get current price and position
            current_price = self.binance.get_current_price(self.symbol)
            current_position = self.binance.get_position_info(self.symbol)

            logger.info(f"💵 Current price: {current_price:.2f} USDT")

            if current_position:
                logger.info(f"📍 Current position: {current_position['side']} "
                          f"{current_position['quantity']:.6f} BTC @ {current_position['entry_price']:.2f}")
                logger.info(f"   Unrealized PnL: {current_position['unrealized_pnl']:.2f} USDT")

            # Generate signal
            signal, metadata = self.strategy.generate_signal(current_data)

            # Execute based on signal
            if signal == 1 and not current_position:
                # BUY signal and no position
                self._open_long_position(current_price, metadata, current_data)

            elif signal == -1 and current_position:
                # SELL signal and have position
                self._close_position(current_price, current_position)

            elif current_position:
                # Have position, check for trailing stop or other exits
                self._manage_position(current_price, current_position, current_data)

            self.last_check_time = datetime.now()

        except Exception as e:
            logger.error(f"❌ Error in check_and_trade: {e}", exc_info=True)

    def _open_long_position(self, current_price: float, metadata: Dict, current_data):
        """Open a long position"""
        logger.info("\n" + "="*50)
        logger.info("🟢 BUY SIGNAL - Opening LONG position")
        logger.info("="*50)

        try:
            # Get balance
            balance = self.binance.get_account_balance()
            available = balance['available_balance']

            # Calculate position size
            regime_params = metadata.get('regime_params', {})
            quantity, position_value = self.strategy.calculate_position_size(
                balance=available,
                price=current_price,
                leverage=self.leverage,
                regime_params=regime_params
            )

            # Calculate SL/TP
            atr = float(current_data['close'].pct_change().rolling(14).std().iloc[-1] * current_price)
            stop_loss, take_profit = self.strategy.calculate_stop_loss_take_profit(
                entry_price=current_price,
                side='LONG',
                atr=atr,
                regime_params=regime_params
            )

            logger.info(f"\n📋 Order Details:")
            logger.info(f"   Quantity: {quantity:.6f} BTC")
            logger.info(f"   Value: {position_value:.2f} USDT")
            logger.info(f"   Entry: ~{current_price:.2f} USDT")
            logger.info(f"   Stop Loss: {stop_loss:.2f} USDT")
            logger.info(f"   Take Profit: {take_profit:.2f} USDT")
            logger.info(f"   Leverage: {self.leverage}x")
            logger.info(f"   Regime: {metadata.get('regime', 'Unknown')}")
            logger.info(f"   Confidence: {metadata.get('confidence', 0):.2%}")

            if self.paper_trading:
                logger.info("\n📝 PAPER TRADING - No actual order placed")
                return

            # Place market buy order
            order = self.binance.place_market_order(
                symbol=self.symbol,
                side='BUY',
                quantity=quantity
            )

            if order:
                # Place stop loss
                self.binance.place_stop_loss_order(
                    symbol=self.symbol,
                    side='SELL',
                    quantity=quantity,
                    stop_price=stop_loss
                )

                # Place take profit
                self.binance.place_take_profit_order(
                    symbol=self.symbol,
                    side='SELL',
                    quantity=quantity,
                    take_profit_price=take_profit
                )

                self.position_opened_at = datetime.now()
                self.trades_count += 1

                logger.info("\n✅ Position opened successfully!")

        except Exception as e:
            logger.error(f"❌ Error opening position: {e}", exc_info=True)

    def _close_position(self, current_price: float, position: Dict):
        """Close current position"""
        logger.info("\n" + "="*50)
        logger.info("🔴 SELL SIGNAL - Closing position")
        logger.info("="*50)

        try:
            logger.info(f"Position: {position['side']} {position['quantity']:.6f} BTC")
            logger.info(f"Entry: {position['entry_price']:.2f}")
            logger.info(f"Current: {current_price:.2f}")
            logger.info(f"PnL: {position['unrealized_pnl']:.2f} USDT")

            if self.paper_trading:
                logger.info("\n📝 PAPER TRADING - No actual close")
                return

            success = self.binance.close_position(self.symbol)

            if success:
                duration = datetime.now() - self.position_opened_at if self.position_opened_at else None
                logger.info(f"\n✅ Position closed!")
                if duration:
                    logger.info(f"   Duration: {duration}")

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}", exc_info=True)

    def _manage_position(self, current_price: float, position: Dict, current_data):
        """Manage existing position (trailing stop, etc.)"""
        # This can be enhanced with trailing stop logic
        pass

    def run(self):
        """Main run loop"""
        if not self.initialize():
            logger.error("❌ Initialization failed, exiting...")
            return

        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING LIVE TRADING BOT")
        logger.info("="*70)
        logger.info(f"Check interval: Every {self.check_interval} seconds")
        logger.info(f"Press Ctrl+C to stop")
        logger.info("="*70)

        self.running = True

        try:
            while self.running:
                self.check_and_trade()

                logger.info(f"\n💤 Sleeping for {self.check_interval} seconds...")
                logger.info(f"Next check at: {(datetime.now() + timedelta(seconds=self.check_interval)).strftime('%H:%M:%S')}")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n\n⏹️ Stopping bot (Ctrl+C pressed)...")
            self.stop()

        except Exception as e:
            logger.error(f"\n❌ Unexpected error: {e}", exc_info=True)
            self.stop()

    def stop(self):
        """Stop the bot gracefully"""
        self.running = False

        logger.info("\n" + "="*70)
        logger.info("🛑 SHUTTING DOWN BOT")
        logger.info("="*70)

        # Check final position
        position = self.binance.get_position_info(self.symbol)
        if position:
            logger.warning(f"⚠️ You still have an open position!")
            logger.info(f"   {position['side']} {position['quantity']:.6f} BTC")
            logger.info(f"   PnL: {position['unrealized_pnl']:.2f} USDT")

        # Final balance
        balance = self.binance.get_account_balance()
        logger.info(f"\n💰 Final balance: {balance['total_balance']:.2f} USDT")
        logger.info(f"   Total trades executed: {self.trades_count}")

        logger.info("\n✅ Bot stopped successfully")
        logger.info("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bitcoin Live Trading Bot')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model (optional)')
    parser.add_argument('--config', type=str, default='config_live.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    bot = LiveTradingBot(config_path=args.config, model_path=args.model)
    bot.run()
