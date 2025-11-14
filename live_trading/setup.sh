#!/bin/bash
# Live Trading Bot Setup Script

echo "================================================"
echo "🤖 Bitcoin Live Trading Bot - Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi
echo "✅ Python OK"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your Binance API keys!"
    echo "   nano .env"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Create logs directory
mkdir -p logs
mkdir -p live_results
echo "✅ Created logs and results directories"
echo ""

# Check if API keys are configured
if grep -q "your_api_key_here" .env 2>/dev/null; then
    echo "⚠️  WARNING: API keys not configured!"
    echo "   Please edit .env and add your keys:"
    echo "   nano .env"
    echo ""
else
    echo "✅ API keys configured"
    echo ""
fi

echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Get testnet API keys: https://testnet.binancefuture.com/"
echo "2. Edit .env file: nano .env"
echo "3. Review config: nano config_live.yaml"
echo "4. Start bot: python live_trader.py"
echo ""
echo "📚 Read README.md for detailed instructions"
echo ""
