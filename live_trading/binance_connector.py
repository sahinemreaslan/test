"""
Binance API Connector
Handles all interactions with Binance Futures API
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceConnector:
    """Binance Futures API Connector"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ):
        """
        Initialize Binance connector

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (True) or real trading (False)
        """
        self.testnet = testnet

        if testnet:
            # Testnet URLs
            self.client = Client(
                api_key,
                api_secret,
                testnet=True
            )
            logger.info("🧪 TESTNET MODE - Using fake money!")
        else:
            # REAL TRADING - BE CAREFUL!
            self.client = Client(api_key, api_secret)
            logger.warning("⚠️ LIVE MODE - Using REAL money! Be careful!")

        # Symbol info cache
        self.symbol_info = {}

    def get_account_balance(self) -> Dict[str, float]:
        """Get USDT balance from Futures account"""
        try:
            account = self.client.futures_account()

            balance = {
                'total_balance': 0.0,
                'available_balance': 0.0,
                'unrealized_pnl': 0.0
            }

            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    balance['total_balance'] = float(asset['walletBalance'])
                    balance['available_balance'] = float(asset['availableBalance'])
                    balance['unrealized_pnl'] = float(asset['unrealizedProfit'])
                    break

            logger.info(f"💰 Balance: {balance['total_balance']:.2f} USDT "
                       f"(Available: {balance['available_balance']:.2f}, "
                       f"PnL: {balance['unrealized_pnl']:.2f})")

            return balance

        except BinanceAPIException as e:
            logger.error(f"❌ Error getting balance: {e}")
            return {'total_balance': 0.0, 'available_balance': 0.0, 'unrealized_pnl': 0.0}

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price for symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            return price
        except BinanceAPIException as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return 0.0

    def get_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical klines (candlestick data)

        Args:
            symbol: Trading pair (default: BTCUSDT)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles (max 1500)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"📊 Loaded {len(df)} candles for {symbol} ({interval})")

            return df

        except BinanceAPIException as e:
            logger.error(f"❌ Error getting klines: {e}")
            return pd.DataFrame()

    def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for symbol"""
        try:
            result = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"⚙️ Leverage set to {leverage}x for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"❌ Error setting leverage: {e}")
            return None

    def set_margin_type(self, symbol: str, margin_type: str = "CROSSED"):
        """
        Set margin type (ISOLATED or CROSSED)

        CROSSED: All balance as collateral (recommended)
        ISOLATED: Only position margin at risk
        """
        try:
            result = self.client.futures_change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
            logger.info(f"⚙️ Margin type set to {margin_type} for {symbol}")
            return result
        except BinanceAPIException as e:
            # Error -4046 means margin type is already set
            if e.code == -4046:
                logger.info(f"⚙️ Margin type already {margin_type}")
                return None
            logger.error(f"❌ Error setting margin type: {e}")
            return None

    def get_position_info(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get current position information"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)

            for pos in positions:
                if pos['symbol'] == symbol:
                    position_amt = float(pos['positionAmt'])

                    if position_amt != 0:
                        return {
                            'symbol': symbol,
                            'side': 'LONG' if position_amt > 0 else 'SHORT',
                            'quantity': abs(position_amt),
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'leverage': int(pos['leverage']),
                            'margin_type': pos['marginType']
                        }

            return None  # No position

        except BinanceAPIException as e:
            logger.error(f"❌ Error getting position: {e}")
            return None

    def _get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading rules (precision, min notional, etc.)"""
        if symbol in self.symbol_info:
            return self.symbol_info[symbol]

        try:
            info = self.client.futures_exchange_info()

            for s in info['symbols']:
                if s['symbol'] == symbol:
                    filters = {f['filterType']: f for f in s['filters']}

                    self.symbol_info[symbol] = {
                        'price_precision': s['pricePrecision'],
                        'quantity_precision': s['quantityPrecision'],
                        'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('notional', 10)),
                        'min_qty': float(filters.get('LOT_SIZE', {}).get('minQty', 0.001)),
                        'step_size': float(filters.get('LOT_SIZE', {}).get('stepSize', 0.001))
                    }

                    return self.symbol_info[symbol]

            return {}

        except BinanceAPIException as e:
            logger.error(f"❌ Error getting symbol info: {e}")
            return {}

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity according to symbol rules"""
        info = self._get_symbol_info(symbol)
        step_size = info.get('step_size', 0.001)
        precision = info.get('quantity_precision', 3)

        # Round down to step size
        quantity_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(step_size))

        rounded = (quantity_decimal // step_decimal) * step_decimal
        return float(rounded.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Place market order

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Amount to trade
            reduce_only: Only reduce existing position (for closing)

        Returns:
            Order info if successful, None otherwise
        """
        try:
            # Round quantity
            quantity = self._round_quantity(symbol, quantity)

            if quantity <= 0:
                logger.error(f"❌ Invalid quantity: {quantity}")
                return None

            # Validate min notional
            price = self.get_current_price(symbol)
            notional = quantity * price
            info = self._get_symbol_info(symbol)

            if notional < info.get('min_notional', 10):
                logger.error(f"❌ Order value ({notional:.2f}) below minimum ({info.get('min_notional', 10):.2f})")
                return None

            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                reduceOnly=reduce_only
            )

            logger.info(f"✅ {side} order placed: {quantity} {symbol} @ Market")
            logger.info(f"   Order ID: {order['orderId']}, Status: {order['status']}")

            return order

        except BinanceAPIException as e:
            logger.error(f"❌ Error placing market order: {e}")
            return None

    def place_stop_loss_order(
        self,
        symbol: str,
        side: str,  # SELL for long, BUY for short
        quantity: float,
        stop_price: float
    ) -> Optional[Dict]:
        """Place stop loss order"""
        try:
            quantity = self._round_quantity(symbol, quantity)

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                stopPrice=stop_price,
                quantity=quantity,
                reduceOnly=True
            )

            logger.info(f"🛡️ Stop loss placed: {quantity} {symbol} @ {stop_price}")

            return order

        except BinanceAPIException as e:
            logger.error(f"❌ Error placing stop loss: {e}")
            return None

    def place_take_profit_order(
        self,
        symbol: str,
        side: str,  # SELL for long, BUY for short
        quantity: float,
        take_profit_price: float
    ) -> Optional[Dict]:
        """Place take profit order"""
        try:
            quantity = self._round_quantity(symbol, quantity)

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit_price,
                quantity=quantity,
                reduceOnly=True
            )

            logger.info(f"🎯 Take profit placed: {quantity} {symbol} @ {take_profit_price}")

            return order

        except BinanceAPIException as e:
            logger.error(f"❌ Error placing take profit: {e}")
            return None

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for symbol"""
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"🗑️ All orders cancelled for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"❌ Error cancelling orders: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Close current position"""
        try:
            position = self.get_position_info(symbol)

            if not position:
                logger.info("No position to close")
                return True

            # Cancel all pending orders first
            self.cancel_all_orders(symbol)

            # Close position with market order
            close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'

            order = self.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=position['quantity'],
                reduce_only=True
            )

            if order:
                logger.info(f"✅ Position closed: {position['side']} {position['quantity']} @ Market")
                logger.info(f"   PnL: {position['unrealized_pnl']:.2f} USDT")
                return True

            return False

        except BinanceAPIException as e:
            logger.error(f"❌ Error closing position: {e}")
            return False
