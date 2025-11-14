#!/bin/bash
# Quick script to train model on historical data

echo "🎓 Starting Offline Model Training"
echo "=================================="
echo ""

# Check if CSV exists
if [ ! -f "../btc_15m_data_2018_to_2025.csv" ]; then
    echo "❌ Error: CSV file not found: ../btc_15m_data_2018_to_2025.csv"
    echo ""
    echo "Please make sure the historical data file exists in the parent directory."
    exit 1
fi

# Check if config exists
if [ ! -f "config_live.yaml" ]; then
    echo "⚠️  Warning: config_live.yaml not found, using default config.yaml"
    CONFIG="../config.yaml"
else
    CONFIG="config_live.yaml"
fi

echo "📊 Training data: ../btc_15m_data_2018_to_2025.csv"
echo "⚙️  Config file: $CONFIG"
echo ""
echo "This will take 10-20 minutes..."
echo ""

# Run training
python train_offline.py --csv ../btc_15m_data_2018_to_2025.csv --config $CONFIG --output ../models

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📦 Model saved to: ../models/advanced_system_latest.pkl"
    echo ""
    echo "🚀 Start live trading with:"
    echo "   python live_trader.py --model ../models/advanced_system_latest.pkl"
    echo ""
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi
