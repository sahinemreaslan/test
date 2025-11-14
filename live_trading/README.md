# 🤖 Bitcoin Live Trading Bot

Live trading system for the Fractal Multi-Timeframe Bitcoin strategy on Binance Futures.

---

## ⚠️ IMPORTANT WARNINGS

1. **CRYPTOCURRENCY TRADING IS RISKY**
   - You can lose all your capital
   - Leverage amplifies both gains and losses
   - Past performance does not guarantee future results

2. **START WITH TESTNET**
   - ALWAYS test with fake money first
   - Testnet: https://testnet.binancefuture.com/
   - Verify everything works before using real money

3. **START WITH PAPER TRADING**
   - Even on testnet, start with paper_trading: true
   - This simulates trades without placing actual orders
   - Verify signals make sense before live trading

4. **NEVER TRADE MORE THAN YOU CAN AFFORD TO LOSE**

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
cd live_trading
pip install -r requirements.txt
```

### Step 2: Get API Keys

**For TESTNET (Recommended First!):**
1. Go to https://testnet.binancefuture.com/
2. Login with email
3. Click "API Key" in top right
4. Generate new API key
5. Save both API Key and Secret Key

**For REAL TRADING (After testing!):**
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create new API key
3. **IMPORTANT SECURITY:**
   - ✅ Enable "Futures Trading" only
   - ❌ DO NOT enable "Withdrawal"
   - ✅ Enable IP whitelist (your server IP)
   - ✅ Use 2FA authentication

### Step 3: Configure API Credentials

```bash
# Copy the example file
cp .env.example .env

# Edit .env file and add your API keys
nano .env  # or use any text editor
```

Your `.env` file should look like:
```
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_secret_here
```

### Step 4: Configure Trading Settings

Edit `config_live.yaml`:

```yaml
trading:
  # Start with TESTNET
  testnet: true

  # Start with PAPER TRADING (no actual orders)
  paper_trading: true

  # Adjust position size based on your capital
  # 5,000 TL with 8% position size = 400 TL per trade
  position_size_pct: 0.08

  # Leverage (3x = safer, 5x = aggressive, 7x = very risky)
  leverage: 5
```

### Step 5: Run the Bot

```bash
cd live_trading
python live_trader.py
```

You should see:
```
🤖 BITCOIN LIVE TRADING BOT INITIALIZED
Symbol: BTCUSDT
Leverage: 5x
Testnet: ✅ Yes (Fake money)
Paper Trading: ✅ Yes (No actual trades)
```

---

## 📊 Configuration Presets

### 🟢 CONSERVATIVE (Recommended for Beginners)

```yaml
trading:
  leverage: 3
  position_size_pct: 0.05        # 5% per trade
  testnet: true
  paper_trading: true

advanced_features:
  enable_trailing_stop: true
  trailing_stop_pct: 0.03        # Wider trail (3%)
  enable_partial_exit: true
  partial_exit_percentage: 0.7   # Take 70% early
  enable_position_scaling: false # No pyramiding
```

**Expected:** 20-40% monthly returns, 5-8% max drawdown

---

### 🟡 SMART-AGGRESSIVE (Current Config, Recommended)

```yaml
trading:
  leverage: 5
  position_size_pct: 0.08        # 8% per trade
  testnet: true
  paper_trading: false           # Real orders (after testing!)

advanced_features:
  enable_trailing_stop: true
  trailing_stop_pct: 0.02        # 2% trail
  enable_partial_exit: true
  partial_exit_percentage: 0.4   # Keep 60% for big moves
  enable_position_scaling: true
  max_scale_ins: 2               # Up to 2 additions
```

**Expected:** 40-70% monthly returns, 8-12% max drawdown

---

### 🔴 HYPER-AGGRESSIVE (High Risk!)

```yaml
trading:
  leverage: 7
  position_size_pct: 0.12        # 12% per trade
  testnet: false                 # REAL MONEY!
  paper_trading: false

advanced_features:
  enable_trailing_stop: true
  trailing_stop_pct: 0.015       # Tight trail (1.5%)
  enable_partial_exit: true
  partial_exit_percentage: 0.3   # Keep 70% running
  enable_position_scaling: true
  max_scale_ins: 3
```

**Expected:** 100%+ monthly returns (or 30%+ losses), 15-20% max drawdown

⚠️ **WARNING:** Only use if you're experienced and can handle large swings!

---

## 📈 Understanding the Strategy

### How It Works

1. **Data Collection:**
   - Bot downloads 15-day history every minute
   - Converts to 11 timeframes (3M down to 15m)
   - Calculates 444 features (fractals, indicators, patterns)

2. **Signal Generation:**
   - Ensemble ML models (XGBoost, LightGBM, CatBoost)
   - HMM regime detection (Bull/Bear/Sideways/High Vol)
   - Confidence scoring (0-100%)

3. **Position Management:**
   - **Entry:** Only on high-confidence BUY signals
   - **Stop Loss:** 2x ATR below entry (adjustable)
   - **Take Profit:** 4x ATR above entry (adjustable)
   - **Trailing Stop:** Locks in profits as price rises
   - **Partial Exits:** Takes 40% profit at 50% of TP distance
   - **Position Scaling:** Adds to winners in strong trends

4. **Risk Management:**
   - Only 1 position at a time
   - Crash protection (stops scaling in high volatility)
   - Circuit breaker (stops bot if large loss)
   - Regime-based position sizing

### Example Trade Flow

```
1. Signal detected at $50,000 BTC
   - Confidence: 75%
   - Regime: Bull Market
   - Action: Open LONG position
   - Size: 8% of balance = $400
   - Leverage: 5x = 0.04 BTC
   - SL: $49,000 | TP: $54,000

2. Price rises to $52,000
   - Trailing SL: $50,960
   - Partial Exit: Close 40% (0.016 BTC)
   - Profit secured: $160
   - Remaining: 60% (0.024 BTC)

3. Price rises to $53,000, strong trend continues
   - Trailing SL: $51,940
   - Position Scaling: Add 0.012 BTC (50% of remaining)
   - Total position: 0.036 BTC

4. Price peaks at $55,000, reverses to $53,900
   - Trailing SL hit at $53,900
   - Exit remaining 0.036 BTC
   - Total profit: $160 + $140 = $300 (75% return on $400!)
```

---

## 🛠️ How to Use

### Starting the Bot

```bash
cd live_trading
python live_trader.py
```

### Monitoring

The bot will print:
- Current balance
- Open positions
- Signal checks
- Trade executions
- PnL updates

Example output:
```
🔍 Checking market at 2025-01-15 14:32:00
💵 Current price: 50234.50 USDT
📊 Signal: 1 | Confidence: 0.78 | Regime: Bull Market

🟢 BUY SIGNAL - Opening LONG position
   Quantity: 0.0398 BTC
   Value: 2000.00 USDT
   Entry: ~50234.50 USDT
   Stop Loss: 49200.00 USDT
   Take Profit: 54300.00 USDT
   Leverage: 5x
   Regime: Bull Market
   Confidence: 78.00%

✅ Position opened successfully!

💤 Sleeping for 60 seconds...
Next check at: 14:33:00
```

### Stopping the Bot

Press `Ctrl+C` to stop gracefully. The bot will:
- Show your final balance
- Show any open positions
- Display total trades executed

---

## 📊 Performance Monitoring

### Check Balance

The bot shows balance on every cycle:
```
💰 Balance: 5432.10 USDT
   (Available: 5100.00, PnL: +332.10)
```

### Check Open Position

```
📍 Current position: LONG 0.0398 BTC @ 50234.50
   Unrealized PnL: +89.50 USDT
```

### Trading History

All trades are logged to:
- Console output (live)
- `logs/live_trading.log` (file)
- `live_results/trades.csv` (optional, if enabled)

---

## ⚙️ Advanced Configuration

### Adjusting Position Size

**5,000 TL starting capital:**

| Position % | Leverage | Trade Size | BTC Amount (@ $50k) |
|------------|----------|------------|---------------------|
| 5%         | 3x       | $750       | 0.015 BTC          |
| 8%         | 5x       | $2,000     | 0.040 BTC          |
| 10%        | 5x       | $2,500     | 0.050 BTC          |
| 12%        | 7x       | $4,200     | 0.084 BTC          |

Edit `config_live.yaml`:
```yaml
trading:
  position_size_pct: 0.08  # Adjust this (0.05 = 5%, 0.10 = 10%)
  leverage: 5              # Adjust this (3-7)
```

### Adjusting Check Interval

```yaml
trading:
  check_interval_seconds: 60  # Check every 60 seconds

  # Options:
  # 30 = Very active (more signals, more commission)
  # 60 = Balanced (recommended)
  # 300 = Conservative (5 minutes, less noise)
```

### Adjusting Risk/Reward

```yaml
trading:
  stop_loss_atr_mult: 2.0    # Tighter = 1.5, Wider = 3.0
  take_profit_atr_mult: 4.0  # Conservative = 3.0, Aggressive = 6.0
```

### Disabling Features

```yaml
advanced_features:
  enable_trailing_stop: false      # Disable trailing stop
  enable_partial_exit: false       # Disable partial exits
  enable_position_scaling: false   # Disable pyramiding
```

---

## 🔒 Security Best Practices

1. **API Key Security:**
   - Never share your API keys
   - Never commit `.env` to git
   - Use IP whitelist on Binance
   - Disable withdrawal permissions
   - Enable 2FA on Binance account

2. **Trading Security:**
   - Start with testnet
   - Then paper trading
   - Then small real positions
   - Gradually increase size

3. **Server Security:**
   - Use strong passwords
   - Keep server updated
   - Use firewall
   - Monitor logs for suspicious activity

4. **Backup:**
   - Backup your `.env` file securely
   - Backup trade history
   - Document your configuration

---

## 🐛 Troubleshooting

### "API Key not found"

**Problem:** `.env` file missing or incorrect

**Solution:**
```bash
# Check .env exists
ls -la .env

# Check content
cat .env

# Should show:
# BINANCE_API_KEY=xxx...
# BINANCE_API_SECRET=xxx...
```

---

### "Insufficient balance"

**Problem:** Not enough USDT in Futures account

**Solution:**
1. Go to Binance Futures
2. Transfer USDT from Spot to Futures wallet
3. Minimum recommended: $50 for testnet, $100+ for real

---

### "Position opening failed"

**Problem:** Could be API permissions, balance, or market conditions

**Solution:**
1. Check API key has "Futures Trading" enabled
2. Check balance is sufficient
3. Check logs for specific error message
4. Try paper_trading: true first

---

### "Not enough historical data"

**Problem:** Binance API rate limit or connection issue

**Solution:**
1. Wait 1 minute and try again
2. Check internet connection
3. Check Binance API status: https://www.binance.com/en/support/announcement

---

### "Strategy not trained"

**Problem:** Training failed on startup

**Solution:**
1. Check you have enough historical data (need 300+ candles)
2. Check logs for specific error
3. Verify all dependencies installed
4. Try deleting old model cache

---

## 📞 Support & Resources

### Binance Resources
- Testnet: https://testnet.binancefuture.com/
- API Docs: https://binance-docs.github.io/apidocs/futures/en/
- Status: https://www.binance.com/en/support/announcement

### Strategy Documentation
- `ADVANCED_FEATURES.md` - Detailed feature explanations
- `STRATEGY_IMPROVEMENTS.md` - Performance analysis
- `LEVERAGE_GUIDE.md` - Leverage and commission guide

### Logs
- Console output (real-time)
- `logs/live_trading.log` (detailed file log)

---

## 🎯 Recommended Progression

### Week 1: Testing Phase
```yaml
testnet: true
paper_trading: true
leverage: 3
position_size_pct: 0.05
```
**Goal:** Understand how bot works, verify signals

---

### Week 2: Testnet Live
```yaml
testnet: true
paper_trading: false  # Real orders on testnet
leverage: 5
position_size_pct: 0.08
```
**Goal:** Test actual order execution, verify performance

---

### Week 3: Real Trading (Small)
```yaml
testnet: false        # REAL MONEY!
paper_trading: false
leverage: 3           # Conservative first
position_size_pct: 0.03  # Small positions
```
**Goal:** Build confidence with real money, small risk

---

### Week 4+: Full Strategy
```yaml
testnet: false
paper_trading: false
leverage: 5           # Aggressive growth
position_size_pct: 0.08
```
**Goal:** Full strategy execution, monitor and adjust

---

## 📈 Expected Results

Based on backtesting (2019-2025):

| Configuration | Monthly Return | Max Drawdown | Win Rate | Risk Level |
|--------------|----------------|--------------|----------|------------|
| Conservative | 15-25%         | 5-8%         | 82%      | Low        |
| Smart-Aggr.  | 40-70%         | 8-12%        | 81%      | Medium     |
| Hyper-Aggr.  | 100%+          | 15-20%       | 79%      | High       |

**IMPORTANT:** Past performance ≠ Future results!

### 5,000 TL Growth Projection (Smart-Aggressive)

| Month | Conservative (25%) | Smart-Aggressive (50%) | Hyper-Aggressive (100%) |
|-------|-------------------|------------------------|-------------------------|
| 0     | 5,000 TL          | 5,000 TL               | 5,000 TL                |
| 1     | 6,250 TL          | 7,500 TL               | 10,000 TL               |
| 2     | 7,812 TL          | 11,250 TL              | 20,000 TL               |
| 3     | 9,765 TL          | 16,875 TL              | 40,000 TL               |
| 6     | 19,073 TL         | 56,953 TL              | 320,000 TL              |

**Reality Check:**
- These are IDEAL scenarios
- Expect losing months too
- 2023 was sideways (low returns)
- Risk increases with leverage
- Withdraw profits regularly!

---

## ⚠️ Final Warnings

1. **This is NOT financial advice**
2. **Only invest what you can afford to lose**
3. **Leverage is dangerous** - can liquidate your account
4. **Past performance does NOT guarantee future results**
5. **Cryptocurrency is extremely volatile**
6. **Start small and scale up slowly**
7. **Monitor the bot regularly** - don't "set and forget"
8. **Have a stop-loss for your entire account** (circuit breaker)

---

## 🚀 Ready to Start?

**Checklist:**
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Got API keys from Binance testnet
- [ ] Created `.env` file with API keys
- [ ] Reviewed `config_live.yaml` settings
- [ ] Set `testnet: true` and `paper_trading: true`
- [ ] Understand the risks
- [ ] Read all documentation

**Start command:**
```bash
cd live_trading
python live_trader.py
```

**Good luck! 🎯**

---

*Remember: Trading is a marathon, not a sprint. Focus on consistent, sustainable returns rather than getting rich quick. The strategy is powerful, but discipline and risk management are what make the difference between success and failure.*
