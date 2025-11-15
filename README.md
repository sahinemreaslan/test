# 🚀 Bitcoin Fractal Trading System

**Professional algorithmic trading bot with ML, HMM regime detection, and Binance Futures live trading**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](.)

**Versiyon:** 4.0 | **Durum:** Production Ready ✅

---

## 📋 İçindekiler

1. [Hızlı Başlangıç](#-hızlı-başlangıç) - 5 dakikada başla
2. [Sistem Özellikleri](#-sistem-özellikleri) - Neler yapabilir?
3. [Kurulum](#-kurulum) - Adım adım kurulum
4. [Live Trading](#-live-trading) - Testnet & Production
5. [Dashboard Kullanımı](#-dashboard-kullanımı) - Web arayüzü
6. [Backtest](#-backtest) - Strateji testi
7. [Konfigürasyon](#️-konfigürasyon) - Ayarlar
8. [Güvenlik](#️-güvenlik) - Önemli!
9. [Sorun Giderme](#-sorun-giderme) - Hata çözümleri
10. [Proje Yapısı](#-proje-yapısı) - Kod organizasyonu

---

## ⚡ Hızlı Başlangıç

### 5 Dakikada Başla

```bash
# 1. Repo klonla
git clone <repo-url>
cd test

# 2. Tam kurulum (dependencies + model eğitimi)
./bot setup

# 3. Testnet API keylerini al ve .env'e ekle
# https://testnet.binancefuture.com/
nano live_trading/.env

# 4. Botu başlat
./bot testnet

# 5. Dashboard'ları aç (yeni terminal)
./bot dashboards
```

**✅ Hazır! Bot testnet'te çalışıyor.**

**Tüm komutlar:**
```bash
./bot help     # Yardım
./bot setup    # İlk kurulum
./bot testnet  # Testnet bot
./bot production  # Production bot (dikkatli!)
./bot dashboard   # Metrics dashboard
./bot chart       # Chart dashboard
./bot stop        # Durdur
```

---

## 🎯 Sistem Özellikleri

### ⭐ Ana Özellikler

- **11 Timeframe Analizi** - 3M, 1M, 1W, 1D, 12h, 8h, 4h, 2h, 1h, 30m, 15m
- **Fractal Pattern Detection** - HHHL, HLLH, INSIDE, OUTSIDE
- **Ensemble ML** - XGBoost + LightGBM + CatBoost
- **HMM Regime Detection** - Bull, Bear, Sideways, High Volatility
- **Advanced Position Management**
  - Trailing Stop Loss - Kar kilitleme
  - Partial Exits - Kademeli çıkış (%40-50)
  - Position Scaling - Kazanan pozisyonlara ekleme
- **Live Trading**
  - Binance Futures API entegrasyonu
  - Testnet support (sahte para)
  - Paper trading (simülasyon)
  - Smart candle synchronization (15m)
- **Web Dashboards**
  - Real-time metrics dashboard
  - Interactive candlestick charts
  - Balance tracking (PnL, ROI)
  - Trade history

### 📈 Backtest Performansı (2019-2025)

| Metrik | Değer |
|--------|-------|
| **Toplam Getiri** | +175% |
| **CAGR** | +28.3% |
| **Max Drawdown** | -14.09% |
| **Sharpe Ratio** | 2.65 |
| **Win Rate** | 82.1% |
| **Tüm Yıllar** | ✅ POZİTİF |

**2020 COVID Crash:** +147% (crash protection sayesinde!)

---

## 💻 Kurulum

### Tek Komutla Kurulum

```bash
# Repo klonla
git clone <repo-url>
cd test

# Otomatik kurulum (dependencies + model eğitimi)
./bot setup
```

Bu komut:
- ✅ Python dependencies yükler
- ✅ .env dosyası oluşturur
- ✅ 7 yıllık data ile model eğitir (10-30 dakika)
- ✅ models/advanced_system_latest.pkl oluşturur

### API Keyleri Al

**TESTNET (Önerilen - Sahte Para):**
1. https://testnet.binancefuture.com/ → GitHub ile giriş
2. API Management → Create API Key
3. Keyleri kopyala

**PRODUCTION (Gerçek Para - Dikkatli!):**
1. https://www.binance.com/en/my/settings/api-management
2. Create API → **Sadece** "Futures Trading" izni
3. **Withdrawal izni KAPALI** ⚠️
4. IP Whitelist ekle (zorunlu)
5. 2FA aktif et

### .env Dosyasını Doldur

```bash
nano live_trading/.env
```

**Testnet için:**
```bash
BINANCE_API_KEY=your_testnet_key_here
BINANCE_API_SECRET=your_testnet_secret_here
```

**Production için:**
```bash
# .env.production dosyasına
BINANCE_API_KEY=your_production_key_here
BINANCE_API_SECRET=your_production_secret_here
```

✅ Kurulum tamamlandı!

---

## 🤖 Live Trading

### Testnet vs Production - Farklar

| Özellik | Testnet (Önerilen) | Production (Gerçek Para) |
|---------|-------------------|--------------------------|
| **Para** | 🧪 Sahte USDT ($100k) | 💰 Gerçek USDT |
| **Risk** | ✅ Sıfır risk | ⚠️ Sermaye kaybı riski |
| **API Keys** | https://testnet.binancefuture.com/ | https://www.binance.com/ |
| **Binance Server** | Testnet sunucusu | Production sunucusu |
| **Config** | `config_live.yaml` (testnet: true) | `config_production.yaml` (testnet: false) |
| **ENV File** | `.env` | `.env.production` |
| **Başlatma** | `./bot testnet` | `./bot production` |
| **Amaç** | Test, öğrenme, deneme | Gerçek kazanç |
| **Öneri** | ✅ İlk 1-2 hafta burada | ⚠️ Sadece test sonrası |
| **IP Whitelist** | Opsiyonel | ✅ Zorunlu |
| **2FA** | Opsiyonel | ✅ Zorunlu |
| **Withdrawal** | Zaten sahte para | ❌ Mutlaka KAPALI |
| **İlk Sermaye** | Sınırsız (test) | 100-500 USDT (küçük başla) |

### Testnet ile Başlama (Önerilen)

**1. Config Kontrol Et**

`live_trading/config_live.yaml`:
```yaml
trading:
  testnet: true          # ✅ Testnet aktif
  paper_trading: true    # ✅ Simülasyon modu
  leverage: 5
  position_size_pct: 0.08
  check_interval_seconds: 900  # 15 dakika
```

**2. Botu Başlat**

```bash
./bot testnet
```

**Çıktı:**
```
🤖 BITCOIN LIVE TRADING BOT INITIALIZED
Symbol: BTCUSDT
Leverage: 5x
Testnet: ✅ Yes (Fake money)
Paper Trading: ✅ Yes (No actual trades)

⏰ Syncing with 15m candle close...
   Next candle closes at: 16:45:00
   Waiting 555 seconds...

✅ Candle closed! Starting checks...
💵 Current price: 96039.20 USDT
📊 Signal: BUY | Confidence: 0.78 | Regime: Bull Market
```

**3. Dashboard'ları Başlat (Yeni Terminal)**

```bash
# Seçenek 1: Her ikisi için talimat göster
./bot dashboards

# Seçenek 2: Metrics dashboard
./bot dashboard

# Seçenek 3: Chart dashboard
./bot chart
```

- **Dashboard (8501):** http://localhost:8501 - Metrikler, PnL, win rate
- **Chart Dashboard (8502):** http://localhost:8502 - Candlestick grafikler

### Production Trading (Gerçek Para)

⚠️ **ÖNCE TESTNET'TE EN AZ 1 HAFTA TEST ET!**

**1. Production API Keyleri**

`live_trading/.env.production` oluştur:
```bash
BINANCE_API_KEY=your_production_api_key
BINANCE_API_SECRET=your_production_secret
```

**2. Production Config Kontrol**

`live_trading/config_production.yaml`:
```yaml
trading:
  leverage: 3              # Muhafazakar başla
  position_size_pct: 0.03  # %3 (küçük başla!)
  testnet: false           # Gerçek para
  paper_trading: false     # Gerçek emirler

risk_management:
  max_daily_loss_pct: 0.03      # Günlük %3 limit
  circuit_breaker_loss_pct: 0.15  # Acil stop %15
```

**3. Güvenlik Checklist**

```bash
✅ Testnet'te 1+ hafta test edildi
✅ API key IP whitelist eklendi
✅ 2FA aktif
✅ Withdrawal izni KAPALI
✅ Sadece "Futures Trading" izni
✅ Küçük sermaye ile başlanıyor (100-500 USDT)
```

**4. Production Başlat**

```bash
./bot production
```

Script:
- ⚠️ Güvenlik uyarıları gösterir
- 📋 Checklist gösterir
- ✍️ "START PRODUCTION" yazmanı ister
- 🚀 Onaydan sonra başlar

### Bot Nasıl Çalışır?

**✅ Hem LONG hem SHORT pozisyon desteği var!**

```
Her 15 dakikada bir (candle close):
  1. Market data çek (500 candle)
  2. 11 timeframe'e dönüştür
  3. 445+ feature oluştur
  4. ML model sinyal üret
  5. Regime tespit et (Bull/Bear/Sideways/HighVol)

  Sinyal Mantığı:
  - Signal = 1 (BUY): LONG pozisyon aç
    • Pozisyon yoksa → LONG aç
    • SHORT pozisyon varsa → SHORT'u kapat, LONG aç

  - Signal = -1 (SELL): SHORT pozisyon aç
    • Pozisyon yoksa → SHORT aç
    • LONG pozisyon varsa → LONG'u kapat, SHORT aç

  - Signal = 0 (HOLD): Hiçbir şey yapma

  Pozisyon açıldıktan sonra:
    → Stop loss ve take profit otomatik yerleştirilir
    → Trailing stop aktif (karlı pozisyonları korur)
    → Partial exit (kısmi kar realizasyonu)
    → Position scaling (güçlü trendlerde ekleme)
```

**Not:** Bot her iki yönde de (LONG/SHORT) trade yapabilir. Sinyal hangi yönü gösterirse o yöne pozisyon açar.

---

## 📊 Dashboard Kullanımı

### Metrics Dashboard (Port 8501)

**Başlatma:**
```bash
./bot dashboard
```

**Açılır:** http://localhost:8501

**Bölümler:**

**1. Bot Status**
- Running / Stopped
- Current Price
- Current Signal (BUY/SELL/HOLD)
- Regime (Bull/Bear/Sideways/HighVol)
- Open Position details

**2. Balance**
- Start Balance
- Current Balance
- Total PnL
- ROI %

**3. Performance Metrics**
- Win Rate
- Total Trades (wins/losses)
- Sharpe Ratio
- Max Drawdown

**4. Interactive Charts**
- PnL over time (line chart)
- Signal distribution (bar chart)
- Win rate trend (line chart)
- PnL distribution (histogram)

**5. Trade History**
- All trades table
- Entry/exit prices
- PnL per trade
- Duration

**6. Signal History**
- All signals (not just trades)
- Signal strength
- Regime at signal time

**Auto-refresh:** 5 saniyede bir

### Chart Dashboard (Port 8502)

**Başlatma:**
```bash
./bot chart
```

**Açılır:** http://localhost:8502

**Özellikler:**

**1. Candlestick Chart**
- Real-time 15m candles
- Zoom & pan (interactive)
- Time range seçici (6h, 12h, 24h, 3d, 7d)

**2. Technical Indicators**
- Moving Averages (MA7, MA25, MA99)
- RSI (14 period)
- MACD (12, 26, 9)
- Bollinger Bands
- Volume bars

**3. Trade Markers**
- 🟢 Entry points (green triangles)
- 🔴 Exit points (red triangles)
- 💰 PnL labels

**4. Signal Overlays**
- 🔵 BUY signals (cyan circles)
- 🟠 SELL signals (orange circles)

**5. Latest Candles Table**
- Son 10 candle
- OHLCV data
- Color coded (green/red)

**Auto-refresh:** 30 saniyede bir

---

## 🧪 Backtest

**✅ Backtest hem LONG hem SHORT pozisyonları destekliyor!**

### Hızlı Backtest

```bash
# Basit test (son data)
python test_quick.py

# Tam backtest
python main.py

# Walk-forward analysis (önerilen - en gerçekçi)
./bot backtest
```

### Walk-Forward Analysis (Önerilen)

```bash
# Her yıl ayrı train/test
python run_walk_forward.py
# veya
./bot backtest
```

**Çıktı:**
```
WALK-FORWARD ANALYSIS RESULTS

ANNUAL PERFORMANCE:
┌──────┬─────────┬──────────────┬────────┬───────────┐
│ Year │ Return  │ Max Drawdown │ Sharpe │ Win Rate  │
├──────┼─────────┼──────────────┼────────┼───────────┤
│ 2019 │ +109.2% │     9.13%    │  2.45  │   82.3%   │
│ 2020 │ +147.8% │     8.88%    │  3.12  │   83.1%   │
│ 2021 │ +134.5% │     5.68%    │  4.21  │   84.2%   │
│ 2022 │ +108.3% │     8.76%    │  2.89  │   81.5%   │
│ 2023 │  +1.42% │    14.09%    │  0.23  │   79.8%   │
│ 2024 │ +137.1% │     6.89%    │  3.56  │   82.7%   │
│ 2025 │ +44.2%  │     6.33%    │  2.11  │   81.9%   │
└──────┴─────────┴──────────────┴────────┴───────────┘
```

### Backtest Config Değiştirme

`config.yaml`:
```yaml
backtesting:
  initial_capital: 10000
  leverage: 3              # 3x, 5x, 7x
  commission: 0.001        # 0.1%

  # Advanced features
  enable_trailing_stop: true
  trailing_stop_pct: 0.025  # 2.5%

  enable_partial_exit: true
  partial_exit_percentage: 0.5  # 50% erken çık

  enable_position_scaling: true
  max_scale_ins: 1         # Maks 1 ekleme
```

---

## ⚙️ Konfigürasyon

### Config Dosyaları

```
config.yaml                         # Backtest config
live_trading/config_live.yaml       # Testnet/live config
live_trading/config_production.yaml # Production config
```

### Önerilen Presetler

**Yeni Başlayan (Muhafazakar)**
```yaml
trading:
  leverage: 3
  position_size_pct: 0.03      # %3

advanced_features:
  trailing_stop_pct: 0.03      # %3 geniş
  partial_exit_percentage: 0.7  # %70 erken çık
  enable_position_scaling: false  # Scaling KAPALI

risk_management:
  max_daily_loss_pct: 0.03    # Günlük %3 limit
```

**Deneyimli (Smart-Aggressive)**
```yaml
trading:
  leverage: 5
  position_size_pct: 0.08      # %8

advanced_features:
  trailing_stop_pct: 0.02      # %2
  partial_exit_percentage: 0.4  # %40 erken çık
  enable_position_scaling: true
  max_scale_ins: 2             # Maks 2 ekleme

risk_management:
  max_daily_loss_pct: 0.05    # Günlük %5 limit
```

**Riskli (Hyper-Aggressive) ⚠️**
```yaml
trading:
  leverage: 7                  # ⚠️ Yüksek risk!
  position_size_pct: 0.12      # %12

advanced_features:
  trailing_stop_pct: 0.015     # %1.5 dar
  partial_exit_percentage: 0.3
  max_scale_ins: 3

risk_management:
  max_daily_loss_pct: 0.10    # Günlük %10 limit
```

### Önemli Parametreler

```yaml
# Position sizing
position_size_pct: 0.08  # Bakiyenin %8'i
leverage: 5              # 5x kaldıraç
# → Toplam exposure: 8% * 5 = 40% balance

# Stop loss
stop_loss_atr_mult: 2.0  # 2x ATR
# Eğer ATR = $1000 → SL = $2000 aşağıda

# Take profit
take_profit_atr_mult: 4.0  # 4x ATR
# Eğer ATR = $1000 → TP = $4000 yukarıda

# Trailing stop
trailing_stop_pct: 0.02  # %2
# Fiyat yükselince SL otomatik yukarı çekilir

# Check interval
check_interval_seconds: 900  # 15 dakika
# Her 15m candle close'da kontrol
```

---

## 🛡️ Güvenlik

### ⚠️ KRİTİK GÜVENLİK KURALLARI

```
❌ API keylerini ASLA paylaşma
❌ Withdrawal iznini ASLA açma
❌ .env dosyasını git'e gönderme
❌ Public sunucuda çalıştırma
❌ SSH key olmadan bağlanma

✅ Sadece "Futures Trading" izni ver
✅ IP whitelist kullan
✅ 2FA aktif et
✅ Güvenli sunucu kullan
✅ Regular backup yap
```

### API Key Oluşturma (Binance Production)

1. **Binance → API Management**
2. **Create API Key**
   - Label: "Trading Bot"
   - API restrictions: ✅ Enable Futures
   - Withdrawal: ❌ KAPALI
3. **IP Access Restriction**
   - Restrict to trusted IPs
   - Sunucunun IP'sini ekle
4. **2FA Confirm**
5. **Keyleri kopyala ve GÜVENLİ yere kaydet**

### .env Dosyası Güvenliği

```bash
# ✅ DOĞRU
.env                  # Gitignore'da
chmod 600 .env        # Sadece sen okuyabilirsin

# ❌ YANLIŞ
git add .env          # ASLA!
chmod 777 .env        # TEHLİKELİ!
```

### Production Sunucu Güvenliği

```bash
# Firewall
sudo ufw allow 22        # SSH
sudo ufw allow 8501      # Dashboard (opsiyonel)
sudo ufw allow 8502      # Chart dashboard (opsiyonel)
sudo ufw enable

# SSH key-only
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no

# Auto updates
sudo apt install unattended-upgrades
```

---

## 🔧 Sorun Giderme

### Bot Başlamıyor

**Hata:** `ModuleNotFoundError: No module named 'binance'`

**Çözüm:**
```bash
pip install -r requirements.txt
```

---

**Hata:** `FileNotFoundError: advanced_system_latest.pkl`

**Çözüm:**
```bash
# Modeli eğit
./train_offline.sh
```

---

**Hata:** `APIError: Invalid API-key`

**Çözüm:**
```bash
# .env dosyasını kontrol et
cat live_trading/.env

# Keylerin doğru olduğundan emin ol
# Testnet için: testnet.binancefuture.com'dan al
```

### Dashboard Açılmıyor

**Hata:** `Port 8501 already in use`

**Çözüm:**
```bash
# Eski dashboard'u öldür
pkill -f streamlit

# Yeniden başlat
./start_dashboard.sh
```

---

**Hata:** `use_container_width deprecated warning`

**Çözüm:**
```bash
# Kod güncellenmiş, son versiyonu pull et
git pull origin main
```

### Feature Mismatch Hatası

**Hata:** `ValueError: Feature names unseen at fit time`

**Çözüm:**
```bash
# Modeli yeniden eğit
./train_offline.sh

# Bot'u yeniden başlat
cd live_trading
./run_live.sh
```

### Candle Timing Sorunları

**Problem:** Bot her dakika kontrol ediyor (15m yerine)

**Çözüm:**
```yaml
# config_live.yaml
trading:
  check_interval_seconds: 900  # 15 dakika = 900 saniye
```

### Position Açılmıyor (Paper Trading)

**Problem:** "Paper position opened" ama dashboard'da görünmüyor

**Çözüm:** Dashboard'u yenile veya yeniden başlat
```bash
Ctrl+C  # Dashboard'u durdur
./start_dashboard.sh  # Yeniden başlat
```

### Testnet Bağlantı Hatası

**Hata:** `ConnectTimeout` veya `ReadTimeout`

**Çözüm:**
```bash
# 1. İnternet bağlantını kontrol et
ping testnet.binancefuture.com

# 2. Testnet sunucusu down olabilir
# Birkaç dakika bekle ve tekrar dene

# 3. API keylerinin testnet keyleri olduğundan emin ol
```

### Yüksek CPU Kullanımı

**Problem:** Bot %100 CPU kullanıyor

**Çözüm:**
```yaml
# config_live.yaml - Feature sayısını azalt (opsiyonel)
# Veya check interval'i artır
trading:
  check_interval_seconds: 1800  # 30 dakika
```

---

## 📁 Proje Yapısı

```
test/
├── README.md                       # 👈 Bu dosya (her şey burada!)
├── requirements.txt                # Python dependencies
├── config.yaml                     # Backtest configuration
│
├── main.py                         # Backtest runner
├── run_walk_forward.py             # Walk-forward analysis
├── train_offline.sh                # Model training script
├── test_quick.py                   # Quick backtest
│
├── live_trading/                   # 🚀 Live Trading
│   ├── run_live.sh                 # Bot launcher
│   ├── run_production.sh           # Production launcher
│   ├── live_trader.py              # Main bot
│   ├── binance_connector.py        # Binance API wrapper
│   ├── strategy_executor.py        # Signal generation
│   ├── dashboard_data.py           # Dashboard data manager
│   ├── dashboard.py                # Metrics dashboard
│   ├── chart_dashboard.py          # Chart dashboard
│   ├── start_dashboard.sh          # Dashboard launcher
│   ├── start_chart_dashboard.sh    # Chart launcher
│   ├── test_connection.py          # API connection test
│   │
│   ├── config_live.yaml            # Live/testnet config
│   ├── config_production.yaml      # Production config
│   ├── .env.example                # API key template
│   ├── .env.production             # Production API template
│   │
│   ├── PRODUCTION_GUIDE.md         # Production Turkish guide
│   ├── DASHBOARD_KULLANIM.md       # Dashboard Turkish guide
│   └── CHART_DASHBOARD_KULLANIM.md # Chart dashboard guide
│
├── src/                            # Core System
│   ├── data/
│   │   ├── data_loader.py          # Data loading
│   │   └── timeframe_converter.py  # Multi-timeframe conversion
│   │
│   ├── features/
│   │   ├── fractal_analysis.py     # Fractal pattern detection
│   │   ├── indicators.py           # Technical indicators
│   │   └── feature_engineering.py  # Feature pipeline (445+ features)
│   │
│   ├── models/
│   │   └── xgboost_model.py        # XGBoost model
│   │
│   ├── advanced/
│   │   ├── ensemble_models.py      # XGB + LGB + CatBoost
│   │   ├── market_regime.py        # HMM regime detection
│   │   └── integrated_system.py    # Complete system
│   │
│   ├── backtesting/
│   │   ├── backtester.py           # Backtest engine
│   │   └── metrics.py              # Performance metrics
│   │
│   └── utils/
│       └── helpers.py              # Helper functions
│
├── models/                         # Trained models (gitignored)
│   └── advanced_system_latest.pkl  # Pre-trained model
│
├── results/                        # Backtest results (gitignored)
├── logs/                           # Log files (gitignored)
└── data/                           # Runtime data (gitignored)
```

### Önemli Dosyalar

**Backtest:**
- `main.py` - Basit backtest
- `run_walk_forward.py` - Walk-forward analysis
- `config.yaml` - Backtest ayarları

**Live Trading:**
- `live_trading/live_trader.py` - Ana bot (lines 800+)
- `live_trading/binance_connector.py` - API wrapper
- `live_trading/strategy_executor.py` - Sinyal üretimi
- `live_trading/config_live.yaml` - Live ayarları

**Dashboards:**
- `live_trading/dashboard.py` - Metrics dashboard
- `live_trading/chart_dashboard.py` - Chart dashboard
- `live_trading/dashboard_data.py` - Data management

**Core System:**
- `src/features/feature_engineering.py` - Feature pipeline
- `src/advanced/integrated_system.py` - Complete strategy
- `src/backtesting/backtester.py` - Backtest engine

---

## 📚 Nasıl Çalışır?

### 1. Fractal Multi-Timeframe Analizi

Her candle bir önceki mumla 4 şekilde ilişki kurar:

```
HHHL - Higher High Higher Low → 🐂 Boğa gücü
HLLH - Lower High Lower Low   → 🐻 Ayı gücü
INSIDE - Inside Bar            → 📦 Konsolidasyon
OUTSIDE - Outside Bar          → 💥 Volatilite
```

11 farklı timeframe'de (3M → 15m) bu pattern'leri analiz eder.

### 2. Machine Learning Ensemble

3 güçlü model birleşimi:
- **XGBoost** - Gradient boosting
- **LightGBM** - Hızlı & verimli
- **CatBoost** - Robust

Her model tahmin yapar, optimal ağırlıklarla birleştirilir.

### 3. HMM Regime Detection

4 piyasa rejimi tespit edilir:

| Rejim | Position Size | Leverage | Stop Loss |
|-------|--------------|----------|-----------|
| Bull Market | 1.5x | 1.2x | 0.8x (dar) |
| Bear Market | 0.4x | 0.5x | 1.2x (geniş) |
| Sideways | 0.8x | 1.0x | 1.0x |
| High Vol | 0.25x | 0.6x | 1.5x (çok geniş) |

Bot otomatik olarak rejime adapte olur.

### 4. Advanced Position Management

**Trailing Stop:**
```
Entry: $50,000, SL: $49,000
Price → $52,000: SL → $50,960 (2% trail)
Price → $54,000: SL → $52,920
Price drops to $53,000: SL stays $52,920 ✅ Kar korundu!
```

**Partial Exit:**
```
Entry: $50,000, TP: $54,000
Price → $52,000 (50% yol):
  → 40% pozisyonu kapat (kar garantile)
  → 60% kalan (büyük hareket için koş)
```

**Position Scaling:**
```
İlk: 0.02 BTC @ $50,000
Price → $51,500, güçlü trend:
  → Ekle: 0.01 BTC (50% of initial)
Toplam: 0.03 BTC, trailing stop hepsini korur!
```

---

## ⚠️ Risk Uyarıları

### 🚨 ÇOK ÖNEMLİ

```
⚠️ Kripto trading SON DERECE RİSKLİDİR
⚠️ Tüm sermayenizi kaybedebilirsiniz
⚠️ Kaldıraç riski KATLAR (liquidation)
⚠️ Geçmiş performans gelecek GARANTİSİ DEĞİLDİR
⚠️ Bu yazılım EĞİTİM AMAÇLIDIR
⚠️ Gerçek para ile kullanımda TÜM RİSK SİZE AİTTİR
```

### 📋 Kullanım Öncesi Checklist

```
✅ Backtesti çalıştırdım ve anladım
✅ Testnet'te en az 1 hafta test ettim
✅ Paper trading ile simülasyon yaptım
✅ Stratejiyi ve riskleri anlıyorum
✅ Sadece KAYBEDERSEM SORUN OLMAZ parası kullanıyorum
✅ API güvenliği sağlandı (IP whitelist, 2FA)
✅ Withdrawal izni KAPALI
✅ İlk sermayeyi geri çıkardım
✅ Stop loss her zaman aktif
✅ Günlük/haftalık kontrol ediyorum
```

### 💡 Trading İpuçları

```
✅ Küçük başla (1-3% position size)
✅ Muhafazakar leverage (3x maks)
✅ Düzenli kar realizasyonu
✅ İlk sermayeni geri çek
✅ FOMO yapma, sisteme güven
✅ Disiplinli ol
✅ Her gün kontrol et
✅ Beklenmedik durumlar için hazır ol
```

---

## 📝 SSS (Sık Sorulan Sorular)

**Q: Testnet keyleri nerede alırım?**
A: https://testnet.binancefuture.com/ → GitHub ile giriş → API Management

**Q: Model ne kadar sürede eğitiliyor?**
A: 10-30 dakika arası (CPU'ya bağlı). GPU ile 5-10 dakika.

**Q: Her ne kadar kontrol ediyor?**
A: Her 15 dakikada bir (candle close). Config'den değiştirilebilir.

**Q: Paper trading nedir?**
A: Gerçek emir yerleştirmeden simülasyon. Test için ideal.

**Q: Testnet parası nereden geliyor?**
A: Binance testnet otomatik $100,000 sahte USDT veriyor.

**Q: Production'a geçmeden önce ne yapmalıyım?**
A: En az 1 hafta testnet + paper trading. Sonuçları analiz et.

**Q: Dashboard çalışmıyor?**
A: `pkill -f streamlit` sonra yeniden başlat.

**Q: Bot stop oluyor mu otomatik?**
A: Circuit breaker aktifse evet (%15-20 kayıpta otomatik dur).

**Q: Telegram bildirim var mı?**
A: Şu anda yok ama eklenebilir (.env'de TELEGRAM_BOT_TOKEN).

**Q: Multi-coin destekliyor mu?**
A: Şu anda sadece BTCUSDT. Multi-coin planlandı.

**Q: Leverage'ı kaç yapmalıyım?**
A: Yeni başlayan: 3x, Deneyimli: 5x, Uzman: 7x (dikkatli!)

**Q: Modeli ne sıklıkla eğitmeliyim?**
A: Ayda bir veya piyasa değiştiğinde (yeni trend, regime change).

---

## 🚀 Başla!

```bash
# 1. Kurulum (tek komut!)
git clone <repo-url> && cd test && ./bot setup

# 2. Testnet keyleri al
# https://testnet.binancefuture.com/

# 3. .env'e keyleri ekle
nano live_trading/.env

# 4. Botu başlat
./bot testnet

# 5. Dashboard'ları aç (yeni terminal)
./bot dashboard    # Terminal 2
./bot chart        # Terminal 3

# 6. Tarayıcıda aç
# http://localhost:8501 (Metrics)
# http://localhost:8502 (Charts)
```

**Tüm komutlar:**
```bash
./bot help        # Yardım
./bot setup       # Kurulum
./bot testnet     # Testnet bot
./bot production  # Production bot
./bot dashboard   # Metrics dashboard
./bot chart       # Chart dashboard
./bot stop        # Durdur
./bot status      # Durum
```

**✅ Hazırsın! İyi kazançlar! 💰**

---

## 📞 Destek

- **GitHub Issues** - Bug reports, feature requests
- **Binance API Docs** - https://binance-docs.github.io/apidocs/futures/en/
- **Testnet** - https://testnet.binancefuture.com/

---

## 📜 Lisans

MIT License - Eğitim amaçlıdır. Gerçek para ile kullanımda tüm sorumluluk size aittir.

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
USE AT YOUR OWN RISK.
```

---

**Son Güncelleme:** 2025-11-15
**Versiyon:** 4.0
**Durum:** Production Ready ✅
