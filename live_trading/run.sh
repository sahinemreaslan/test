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

# Check if pre-trained model exists
MODEL_PATH="../models/advanced_system_latest.pkl"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ Found pre-trained model: $MODEL_PATH"
    echo "   Using model trained on 2018-2025 data (recommended)"
    echo ""

    # Run with pre-trained model
    python3 live_trader.py --model "$MODEL_PATH"
else
    echo "⚠️  No pre-trained model found"
    echo "   Bot will train on last 15 days of data from API"
    echo ""
    echo "💡 TIP: For better performance, train offline first:"
    echo "   ./train_model.sh"
    echo ""

    # Run with live training
    python3 live_trader.py
fi
