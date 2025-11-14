#!/bin/bash
# Quick run script for live trading bot

echo "🤖 Starting Bitcoin Live Trading Bot..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please run: ./setup.sh first"
    exit 1
fi

# Check if API keys are configured
if grep -q "your_api_key_here" .env; then
    echo "❌ API keys not configured!"
    echo "Please edit .env file and add your Binance API keys"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p live_results

# Run the bot
python3 live_trader.py
