#!/bin/bash
# PRODUCTION Trading Bot Launcher
#
# ⚠️⚠️⚠️ THIS IS FOR REAL MONEY TRADING ⚠️⚠️⚠️
#
# This script includes safety checks before starting real trading

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║                                                              ║${NC}"
echo -e "${RED}║        ⚠️  PRODUCTION TRADING BOT - REAL MONEY! ⚠️         ║${NC}"
echo -e "${RED}║                                                              ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ==================== SAFETY CHECKS ====================

echo -e "${YELLOW}Running safety checks...${NC}"
echo ""

# Check 1: .env.production exists
if [ ! -f .env.production ]; then
    echo -e "${RED}❌ ERROR: .env.production file not found!${NC}"
    echo ""
    echo "Please create .env.production with your REAL API keys:"
    echo "  cp .env.example .env.production"
    echo "  nano .env.production"
    echo ""
    exit 1
fi

# Check 2: API keys are configured
if grep -q "your_production_api_key_here" .env.production; then
    echo -e "${RED}❌ ERROR: API keys not configured!${NC}"
    echo ""
    echo "Please edit .env.production and add your REAL Binance API keys"
    echo "  nano .env.production"
    echo ""
    exit 1
fi

# Check 3: Pre-trained model exists
MODEL_PATH="../models/advanced_system_latest.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Pre-trained model not found!${NC}"
    echo ""
    echo "Recommended: Train model on full dataset first:"
    echo "  ./train_model.sh"
    echo ""
    read -p "Continue without pre-trained model? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check 4: Config file exists
if [ ! -f config_production.yaml ]; then
    echo -e "${RED}❌ ERROR: config_production.yaml not found!${NC}"
    exit 1
fi

# ==================== CONFIGURATION REVIEW ====================

echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║          PRODUCTION CONFIGURATION             ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Extract key settings from config
LEVERAGE=$(grep "leverage:" config_production.yaml | head -1 | awk '{print $2}')
POSITION_SIZE=$(grep "position_size_pct:" config_production.yaml | awk '{print $2}')
TESTNET=$(grep "testnet:" config_production.yaml | tail -1 | awk '{print $2}')
PAPER_TRADING=$(grep "paper_trading:" config_production.yaml | tail -1 | awk '{print $2}')

echo "Leverage: ${LEVERAGE}x"
echo "Position Size: ${POSITION_SIZE} (${POSITION_SIZE}% of balance per trade)"
echo "Testnet: $TESTNET"
echo "Paper Trading: $PAPER_TRADING"
echo ""

# ==================== FINAL WARNINGS ====================

echo -e "${RED}╔════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║              ⚠️  FINAL WARNINGS ⚠️             ║${NC}"
echo -e "${RED}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${RED}1. This will use REAL MONEY on Binance Futures${NC}"
echo -e "${RED}2. You can LOSE money - only risk what you can afford${NC}"
echo -e "${RED}3. Futures trading with leverage is HIGH RISK${NC}"
echo -e "${RED}4. Start with SMALL amounts (not your full capital!)${NC}"
echo -e "${RED}5. MONITOR the bot continuously for the first hours${NC}"
echo -e "${RED}6. Have stop-loss ready to kill the bot if needed${NC}"
echo ""

# ==================== CHECKLIST ====================

echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║              PRE-FLIGHT CHECKLIST              ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo "Before starting, confirm:"
echo ""
echo "  [ ] I tested the bot on TESTNET successfully"
echo "  [ ] I trained the model on full dataset (2018-2025)"
echo "  [ ] I understand the risks of futures trading"
echo "  [ ] I enabled IP whitelist on Binance API"
echo "  [ ] I enabled 2FA on my Binance account"
echo "  [ ] I gave ONLY 'Futures Trading' permission (NO withdrawal!)"
echo "  [ ] I am starting with a SMALL test amount"
echo "  [ ] I will monitor the bot for the first few hours"
echo "  [ ] I have emergency stop plan (Ctrl+C)"
echo "  [ ] I checked config_production.yaml settings"
echo ""

# ==================== FINAL CONFIRMATION ====================

echo -e "${RED}════════════════════════════════════════════════${NC}"
echo -e "${RED}     TYPE 'START PRODUCTION' TO CONTINUE        ${NC}"
echo -e "${RED}     OR PRESS Ctrl+C TO CANCEL                  ${NC}"
echo -e "${RED}════════════════════════════════════════════════${NC}"
echo ""
read -p "Confirmation: " CONFIRMATION

if [ "$CONFIRMATION" != "START PRODUCTION" ]; then
    echo ""
    echo -e "${YELLOW}Production start cancelled. Stay safe!${NC}"
    exit 1
fi

# ==================== START TRADING ====================

echo ""
echo -e "${GREEN}✅ Starting PRODUCTION trading bot...${NC}"
echo ""

# Create logs directory
mkdir -p logs
mkdir -p production_results

# Use .env.production
export ENV_FILE=".env.production"

# Start bot with production config and model
if [ -f "$MODEL_PATH" ]; then
    echo -e "${GREEN}Using pre-trained model: $MODEL_PATH${NC}"
    python3 live_trader.py \
        --config config_production.yaml \
        --model "$MODEL_PATH" \
        2>&1 | tee logs/production_trading.log
else
    echo -e "${YELLOW}Starting without pre-trained model (will train on API data)${NC}"
    python3 live_trader.py \
        --config config_production.yaml \
        2>&1 | tee logs/production_trading.log
fi

# ==================== CLEANUP ====================

echo ""
echo -e "${YELLOW}Production trading stopped.${NC}"
echo -e "${YELLOW}Check logs at: logs/production_trading.log${NC}"
echo -e "${YELLOW}Check results at: production_results/${NC}"
echo ""
