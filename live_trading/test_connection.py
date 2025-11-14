#!/usr/bin/env python3
"""
Test Binance API Connection
Quick script to verify your API keys and connection work
"""

import os
import sys
from dotenv import load_dotenv
from binance_connector import BinanceConnector

def test_connection():
    """Test Binance API connection"""
    print("="*60)
    print("🧪 Testing Binance API Connection")
    print("="*60)
    print()

    # Load credentials
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("❌ ERROR: API keys not found in .env file!")
        print()
        print("Please create .env file with:")
        print("BINANCE_API_KEY=your_key_here")
        print("BINANCE_API_SECRET=your_secret_here")
        return False

    if "your_api_key_here" in api_key:
        print("❌ ERROR: Please replace placeholder API keys in .env file!")
        return False

    print("✅ API keys found in .env")
    print()

    # Test connection
    print("Testing connection to Binance Testnet...")
    try:
        binance = BinanceConnector(api_key, api_secret, testnet=True)
        print("✅ Connected to Binance API")
        print()

        # Test balance
        print("Fetching account balance...")
        balance = binance.get_account_balance()
        print(f"✅ Balance: {balance['total_balance']:.2f} USDT")
        print(f"   Available: {balance['available_balance']:.2f} USDT")
        print()

        if balance['available_balance'] < 10:
            print("⚠️  WARNING: Low balance! You need at least $10 USDT to trade")
            print("   Go to https://testnet.binancefuture.com/ to get testnet funds")
            print()

        # Test price fetch
        print("Fetching BTC price...")
        price = binance.get_current_price("BTCUSDT")
        print(f"✅ Current BTC price: ${price:,.2f}")
        print()

        # Test historical data
        print("Fetching historical data...")
        data = binance.get_historical_klines("BTCUSDT", "15m", limit=100)
        print(f"✅ Loaded {len(data)} candles")
        print()

        # Test leverage setting
        print("Testing leverage setting...")
        binance.set_leverage("BTCUSDT", 5)
        print("✅ Leverage set successfully")
        print()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print()
        print("You can now run the live trading bot:")
        print("  python live_trader.py")
        print()
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Possible issues:")
        print("1. Wrong API keys")
        print("2. API keys don't have Futures permission")
        print("3. Network connection problem")
        print("4. Binance API is down")
        print()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
