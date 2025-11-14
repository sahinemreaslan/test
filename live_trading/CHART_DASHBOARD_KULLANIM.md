# 📈 Chart Dashboard Kullanım Kılavuzu

## 🎯 Ne İşe Yarar?

Chart Dashboard, Bitcoin'in 15 dakikalık (veya istediğin timeframe) mum grafiğini **gerçek zamanlı** olarak gösterir ve teknik analiz yapmanı sağlar.

### ✨ Özellikler

- **📊 Candlestick Chart**: Profesyonel mum grafiği
- **📈 Teknik İndikatörler**: MA, RSI, MACD, Bollinger Bands
- **🎯 Trade Markers**: Entry/exit noktaları grafikte işaretli
- **📡 Signal Overlay**: BUY/SELL sinyalleri grafikte
- **⚡ Real-time**: Binance'den canlı veri
- **🎨 İnteraktif**: Zoom, pan, hover detaylar
- **⏱️ Multi-timeframe**: 1m'den 1D'ye kadar

---

## 🚀 Hızlı Başlangıç

### 1. Dashboard'u Başlat

```bash
cd live_trading
./start_chart_dashboard.sh
```

Veya manuel:
```bash
streamlit run chart_dashboard.py --server.port 8502
```

### 2. Tarayıcıda Aç

```
http://localhost:8502
```

**Not**: Chart Dashboard port **8502** kullanır (Ana dashboard 8501, Chart dashboard 8502)

---

## 📊 Dashboard Layout

### Üst Kısım: Fiyat Bilgileri
```
┌─────────────────────────────────────────────────────────┐
│ Current Price │ Price Change │ 24h High  │ 24h Low     │
│   $94,534     │  +$450 (+2%) │  $95,200  │  $93,800    │
└─────────────────────────────────────────────────────────┘
```

### Ana Chart: Candlestick + İndikatörler
```
┌─────────────────────────────────────────────────────────┐
│                    BTCUSDT 15M Chart                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  [Candlestick grafiği + MA çizgileri]             │  │
│  │  🟢 Entry    🔴 Exit    🔵 BUY    🟠 SELL        │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌─ Volume ───────────────────────────────────────┐    │
│  │  [Volume barları]                              │    │
│  └────────────────────────────────────────────────┘    │
│  ┌─ RSI ──────────────────────────────────────────┐    │
│  │  [RSI çizgisi, 30-70 seviyeleri]              │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Alt Kısım: İstatistikler & Latest Candles

---

## ⚙️ Ayarlar (Sol Sidebar)

### Auto Refresh
- ✅ **Auto Refresh**: Grafik otomatik güncellenir
- **Interval**: 5-60 saniye (Önerilen: 15)

### Timeframe Seçimi
- **1m**: 1 dakikalık mumlar (scalping için)
- **5m**: 5 dakikalık mumlar
- **15m**: 15 dakikalık mumlar (varsayılan, bot bunu kullanıyor)
- **30m**: 30 dakikalık mumlar
- **1h**: 1 saatlik mumlar
- **4h**: 4 saatlik mumlar
- **1D**: Günlük mumlar (swing trading)

### Candle Count
- 50-500 arası mum sayısı
- Önerilen: 200 (15m'de ~2 gün)

### 📊 Technical Indicators

**Moving Averages (MA)**
- MA7: Kısa vadeli trend (mavi)
- MA25: Orta vadeli trend (turuncu)
- MA99: Uzun vadeli trend (mor)

**Bollinger Bands (BB)**
- Üst/orta/alt bantlar
- Volatilite göstergesi

**Volume**
- İşlem hacmi
- Yeşil: Alıcı baskısı
- Kırmızı: Satıcı baskısı

**RSI (Relative Strength Index)**
- 0-100 arası
- >70: Aşırı alım (overbought)
- <30: Aşırı satım (oversold)

**MACD**
- Trend gücü ve yön
- MACD çizgisi + Signal çizgisi + Histogram

### 🎯 Overlays

**Show Trades**
- 🟢 **Üçgen yukarı**: Entry (pozisyon açılış)
- 🔴 **Üçgen aşağı**: Exit (pozisyon kapanış)
- Hover ile detaylar (fiyat, miktar, PnL)

**Show Signals**
- 🔵 **Mavi daire**: BUY sinyali
- 🟠 **Turuncu daire**: SELL sinyali
- Hover ile confidence ve regime

---

## 🎮 Kullanım Senaryoları

### Senaryo 1: Bot İzleme (Ana Kullanım)

```bash
# Terminal 1: Bot'u çalıştır
./run.sh

# Terminal 2: Ana dashboard
./start_dashboard.sh
# http://localhost:8501

# Terminal 3: Chart dashboard
./start_chart_dashboard.sh
# http://localhost:8502
```

**Kullanım:**
1. Ana dashboard'da metrikleri izle
2. Chart dashboard'da grafiği izle
3. Bot sinyal verdiğinde:
   - Ana dashboard: Sinyal ve confidence
   - Chart dashboard: Grafikte nerede sinyal verdi

### Senaryo 2: Teknik Analiz

Chart Dashboard'u aç:
1. **Timeframe**: 15m seç
2. **Indicators**: MA, RSI, Volume aç
3. **Candles**: 200 mum
4. Analiz yap:
   - Fiyat MA7'nin üstünde mi? → Bullish
   - RSI >70 mi? → Aşırı alım, düşüş beklenir
   - Volume artıyor mu? → Güçlü hareket

### Senaryo 3: Entry/Exit Noktalarını İncele

Bot trade yaptıktan sonra:
1. Chart Dashboard'u aç
2. **Show Trades**: ON
3. **Show Signals**: ON
4. Grafikte gör:
   - 🔵 BUY sinyali nerede verildi?
   - 🟢 Entry nerede yapıldı?
   - 🔴 Exit nerede yapıldı?
   - İyi timing miydi?

### Senaryo 4: Backtest Sonuçlarını Görselleştir

Paper trading sonuçlarını grafikte gör:
1. Bot'u paper trading ile çalıştır (1 gün)
2. Chart Dashboard'u aç
3. Tüm entry/exit noktalarını gör
4. Analiz yap:
   - Hangi entry'ler kârlıydı?
   - Hangi exit'ler erken miydi?
   - Pattern var mı?

---

## 📊 Grafikleri Okuma Rehberi

### Candlestick (Mum) Grafiği

```
     ┌─┐  ← Fitil (wick): Yüksek/düşük fiyat
     │█│  ← Gövde (body): Açılış/kapanış
     └─┘

🟢 Yeşil mum: Kapanış > Açılış (yükseliş)
🔴 Kırmızı mum: Kapanış < Açılış (düşüş)
```

### Moving Averages (Hareketli Ortalamalar)

- **Fiyat > MA**: Bullish (yükseliş trendi)
- **Fiyat < MA**: Bearish (düşüş trendi)
- **MA7 > MA25 > MA99**: Güçlü yükseliş
- **MA7 < MA25 < MA99**: Güçlü düşüş
- **MA kesişimi**: Trend değişimi sinyali

### Bollinger Bands

```
   ──────── Üst band (aşırı alım)
   ~~~~~~~~ Orta band (ortalama)
   ──────── Alt band (aşırı satım)
```

- Fiyat üst banda yaklaşır → Aşırı alım, düşüş bekle
- Fiyat alt banda yaklaşır → Aşırı satım, yükseliş bekle
- Bantlar daralır → Volatilite düşük, büyük hareket yakın
- Bantlar genişler → Yüksek volatilite

### Volume (Hacim)

- **Yükseliş + Yüksek volume**: Güçlü alım
- **Düşüş + Yüksek volume**: Güçlü satım
- **Yükseliş + Düşük volume**: Zayıf hareket
- Volume artışı → Trend güçlenebilir

### RSI (0-100)

```
100 ─────────
 70 ───────── Aşırı alım (overbought)
 50 ───────── Nötr
 30 ───────── Aşırı satım (oversold)
  0 ─────────
```

- **RSI > 70**: Aşırı alım, düzeltme bekle
- **RSI < 30**: Aşırı satım, toparlanma bekle
- **RSI 40-60**: Yatay piyasa
- **Divergence**: Fiyat yükselir RSI düşer → Trend zayıflar

### MACD

- **MACD > Signal**: Bullish
- **MACD < Signal**: Bearish
- **Kesişim yukarı**: BUY sinyali
- **Kesişim aşağı**: SELL sinyali
- **Histogram büyür**: Trend güçlenir
- **Histogram küçülür**: Trend zayıflar

---

## 💡 Pro İpuçları

### İki Dashboard Birlikte Kullan

**Ana Dashboard (8501):**
- Performans metrikleri
- Win rate
- PnL takibi

**Chart Dashboard (8502):**
- Teknik analiz
- Entry/exit noktaları
- Trend analizi

### Multi-Monitor Setup

İdeal kurulum:
- **Monitör 1**: Terminal (bot çalışıyor)
- **Monitör 2**: Ana Dashboard (metrikler)
- **Monitör 3**: Chart Dashboard (grafikler)

Tek monitörse:
- Tarayıcıda iki tab aç (8501 ve 8502)

### Timeframe Stratejisi

Farklı timeframe'lerde analiz yap:

1. **1h Chart**: Genel trend nedir?
2. **15m Chart**: Entry/exit zamanlaması
3. **5m Chart**: Hassas giriş noktası

**Kural**: Büyük timeframe trend, küçük timeframe timing!

### İndikatör Kombinasyonları

**Trend Following:**
- MA7, MA25, MA99
- MACD
- Volume

**Range Trading:**
- Bollinger Bands
- RSI
- Volume

**Momentum:**
- RSI
- MACD
- Volume

### Pattern Recognition

Chart'ta dikkat et:
- **Support/Resistance**: Fiyat sıçrama noktaları
- **Trend Lines**: Yükseliş/düşüş kanalları
- **Chart Patterns**: Head & Shoulders, Double Top/Bottom
- **Candlestick Patterns**: Doji, Hammer, Engulfing

---

## 🔧 Sorun Giderme

### Chart Görünmüyor

```bash
# API keys kontrol et
cat .env

# Binance bağlantısı test et
cd live_trading
python test_connection.py
```

### Trade Markers Görünmüyor

- Bot henüz trade yapmadı
- `Show Trades` checkbox'ı ON mu?
- Timeframe içinde trade var mı? (zoom out dene)

### Grafikler Yavaş

- Candle count'u azalt (500 → 100)
- Auto refresh interval'i artır (15 → 30 saniye)
- Indicator sayısını azalt

### Port Hatası (8502 kullanımda)

```bash
# Farklı port kullan
streamlit run chart_dashboard.py --server.port 8503
```

---

## 📈 Trading Stratejileri (Grafikten)

### Trend Following

Grafik analizi:
1. MA7 > MA25 > MA99 → Yükseliş trendi
2. Fiyat MA7'ye yaklaşır (pullback)
3. RSI 40-60 arası (aşırı alım değil)
4. Volume artıyor
5. **Action**: BUY sinyali bekle

### Mean Reversion

Grafik analizi:
1. Fiyat Bollinger alt banda yakın
2. RSI < 30 (aşırı satım)
3. Volume azalıyor (panik satış bitti)
4. MA99 yatay (range piyasa)
5. **Action**: Toparlanma için BUY

### Breakout Trading

Grafik analizi:
1. Bollinger Bands daralıyor
2. Fiyat MA'ların arasında sıkışmış
3. Volume düşük
4. Ani volume artışı + mum kırılması
5. **Action**: Kırılım yönünde trade

---

## 🎨 Özelleştirme

### Farklı Renkler

`chart_dashboard.py` dosyasında:

```python
# Candlestick renkleri
increasing_line_color='#00ff00',  # Yeşil
decreasing_line_color='#ff0000',  # Kırmızı
```

### Yeni İndikatör Ekle

Örnek: EMA eklemek için:

```python
# Calculate EMA
df['ema12'] = df['close'].ewm(span=12).mean()

# Add to chart
fig.add_trace(
    go.Scatter(
        x=df['timestamp'],
        y=df['ema12'],
        name='EMA12',
        line=dict(color='yellow', width=1)
    ),
    row=1, col=1
)
```

### Farklı Timeframe Default

```python
# Sidebar'da
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1D"],
    index=2  # 0: 1m, 1: 5m, 2: 15m (default)
)
```

---

## 📱 Mobil Erişim

Chart Dashboard'a telefondan da bakabilirsin:

### Local Network

```bash
# Bilgisayarın IP'sini öğren
hostname -I

# Chart Dashboard başlat
./start_chart_dashboard.sh

# Telefondan aç:
http://192.168.1.XXX:8502
```

**Not**: Grafik interaktif, mobilde de zoom/pan çalışır!

---

## 🎯 Kullanım Örnekleri

### Örnek 1: Sabah Analizi

08:00 - Chart Dashboard'u aç:
1. Timeframe: 4h
2. MA + RSI + Volume göster
3. Genel trend nedir?
   - Bullish → Uzun pozisyon ara
   - Bearish → Kısa pozisyon ara
4. Timeframe: 15m
5. Bot'un bugün nerede entry yapacağını tahmin et

### Örnek 2: Trade Sonrası Analiz

Bot trade kapattı:
1. Chart Dashboard'u aç
2. Trade markers'ı göster
3. Analiz:
   - Entry iyi miydi? (MA'lara göre)
   - Exit erken mi? (RSI'a göre)
   - Daha iyi timing mümkün müydü?
4. Not al, bir sonraki trade için

### Örnek 3: Market Crash Takibi

Piyasa düşüyor:
1. Chart Dashboard: 15m
2. RSI < 30 mı? → Aşırı satım
3. Volume'ı kontrol et:
   - Volume düşüyor → Panik bitti, toparlanma yakın
   - Volume artıyor → Henüz erken
4. Bot ne yapacak? Bekle ve gör

---

## 📊 Veri Kaynağı

Chart Dashboard şu verileri kullanır:

**Binance API:**
- Real-time OHLCV data (candlestick)
- Volume data
- Current price

**Dashboard Data Manager:**
- Trade history (entry/exit points)
- Signal history (BUY/SELL)

**Calculated:**
- Moving Averages
- RSI, MACD
- Bollinger Bands

---

## 🚀 İleri Seviye

### Multi-Timeframe Analysis

Farklı tarayıcı tablarında:
- Tab 1: 1h chart (trend)
- Tab 2: 15m chart (entry)
- Tab 3: 5m chart (precise timing)

### Correlation Analysis

Başka chart dashboard instance'ı:
```bash
# Port 8503'te ETH chart
streamlit run chart_dashboard.py --server.port 8503
# Config'de symbol değiştir: ETHUSDT
```

### Custom Indicators

Kendi indikatörünü ekle:
```python
# Ichimoku, Fibonacci, etc.
```

---

## ✅ Checklist

Chart Dashboard kullanmaya başlamadan:

- [ ] Ana dashboard çalışıyor (8501)
- [ ] Chart dashboard başlattım (8502)
- [ ] Bot çalışıyor
- [ ] API keys doğru
- [ ] Timeframe seçtim (15m önerilen)
- [ ] İndikatörleri seçtim
- [ ] Trade markers aktif
- [ ] Auto refresh ON

---

## 🎓 Öğrenme Kaynakları

### Candlestick Patterns
- Doji, Hammer, Shooting Star
- Engulfing, Harami
- Morning/Evening Star

### Technical Analysis
- Support/Resistance
- Trend Lines
- Fibonacci Retracement

### Risk Management
- Stop loss belirleme
- Position sizing
- Risk/reward ratio

---

**Chart Dashboard ile profesyonel teknik analiz yap! 📈🚀**

*Bot'un ne yaptığını grafikte görmek bambaşka bir deneyim!*
