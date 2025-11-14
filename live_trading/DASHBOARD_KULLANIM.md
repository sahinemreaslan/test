# 📊 Dashboard Kullanım Kılavuzu

## 🎯 Ne İşe Yarar?

Live Trading Dashboard, bot'unuzun performansını gerçek zamanlı olarak tarayıcıdan takip etmenizi sağlar.

### ✨ Özellikler

- **📈 Gerçek Zamanlı Grafikler**: Fiyat, sinyaller, PnL, win rate
- **🤖 Bot Durumu**: Anlık durum, fiyat, rejim, son sinyal
- **💰 Performans Metrikleri**: Total PnL, win rate, Sharpe ratio, drawdown
- **📋 Trade Geçmişi**: Tüm açılan/kapanan pozisyonlar
- **📡 Sinyal Geçmişi**: Tüm BUY/SELL sinyalleri
- **🎨 Interaktif Arayüz**: Zoom, filter, export
- **🔄 Otomatik Güncelleme**: 5 saniyede bir refresh

---

## 🚀 Hızlı Başlangıç

### 1. Paketleri Kur

```bash
cd live_trading
pip install -r requirements.txt
```

Veya sadece dashboard için:
```bash
pip install streamlit plotly
```

### 2. Dashboard'u Başlat

```bash
./start_dashboard.sh
```

Veya manuel:
```bash
streamlit run dashboard.py
```

### 3. Tarayıcıda Aç

Dashboard otomatik olarak açılır:
```
http://localhost:8501
```

---

## 📖 Kullanım

### Dashboard Layout

#### 🤖 Bot Status (Üst Kısım)
- **Status**: Bot çalışıyor mu? (🟢 Running / 🔴 Stopped)
- **Current Price**: Anlık BTC fiyatı
- **Market Regime**: Piyasa durumu (🟢 Bull / 🟡 Sideways / 🔴 Bear)
- **Last Signal**: Son sinyal (🟢 BUY / 🔴 SELL / ⚪ HOLD)

#### 📈 Performance Metrics
- **Total PnL**: Toplam kar/zarar
- **Win Rate**: Kazanma yüzdesi
- **Total Trades**: Toplam trade sayısı (✅ kazanan | ❌ kaybeden)
- **Sharpe Ratio**: Risk-adjusted return (>2 iyi)
- **Max Drawdown**: En büyük düşüş

#### 📍 Open Position (Varsa)
- Açık pozisyon detayları
- Entry price, quantity, unrealized PnL

#### 📊 Charts (4 Tab)

**1. 💰 PnL Chart**
- Kümülatif kar/zarar grafiği
- Zaman içinde performans
- Trend analizi

**2. 📈 Signals**
- Fiyat grafiği üzerinde BUY/SELL işaretleri
- 🟢 Üçgen yukarı = BUY
- 🔴 Üçgen aşağı = SELL
- Confidence over time grafiği

**3. 🎯 Win Rate**
- Regime'lere göre win rate
- Hangi piyasa koşulunda daha başarılı?
- Tablo ve bar chart

**4. 📊 Trade Distribution**
- PnL dağılımı (histogram)
- Kaç trade kazandı/kaybetti?
- Ortalama kar/zarar

#### 📋 Recent Trades
- Son 20 trade
- 🟢 Yeşil = Kar
- 🔴 Kırmızı = Zarar
- Timestamp, type, side, PnL

#### 📡 Recent Signals
- Son 20 sinyal
- BUY/SELL/HOLD
- Confidence seviyeleri
- Regime bilgisi

---

## ⚙️ Ayarlar (Sol Sidebar)

### Auto Refresh
- ✅ Aktif: Dashboard otomatik güncellenir
- Interval: 1-60 saniye arası ayarlanabilir
- Önerilen: 5 saniye

### Time Range Filter
- **Last Hour**: Son 1 saat
- **Last 6 Hours**: Son 6 saat
- **Last 24 Hours**: Son 1 gün
- **Last Week**: Son 1 hafta
- **All Time**: Tüm veri

### Export Data
- 📥 **Export Data** butonu
- CSV formatında export eder
- `exports/` klasörüne kaydeder

### Clear All Data
- 🗑️ **Clear All Data** butonu
- **TEHLİKELİ!** Tüm veriyi siler
- Test için kullanılır

---

## 🎮 Kullanım Senaryoları

### Senaryo 1: Bot İzleme (24/7)

```bash
# Terminal 1: Bot'u başlat
./run.sh

# Terminal 2: Dashboard'u başlat
./start_dashboard.sh
```

Dashboard'dan:
- Bot durumunu izle
- Sinyalleri takip et
- Performansı gör
- Gerekirse bot'u durdur (Terminal 1'de Ctrl+C)

### Senaryo 2: Geçmiş Analizi

Bot durduktan sonra:
```bash
./start_dashboard.sh
```

- Time Range: "All Time" seç
- Tüm trade'leri incele
- Win rate analizi yap
- Hangi regime'de iyi performans göstermiş?

### Senaryo 3: Canlı Takip

Bot çalışırken:
1. Dashboard'u aç
2. Auto Refresh: ON
3. Interval: 5 saniye
4. Charts → Signals tab
5. Gerçek zamanlı fiyat + sinyaller

### Senaryo 4: Performans Raporu

```bash
# Dashboard'u aç
./start_dashboard.sh

# Export Data butonuna tık
# exports/ klasörüne CSV olarak kaydedilir

# CSV'leri Excel'de aç ve analiz yap
```

---

## 📊 Grafikleri Anlama

### PnL Chart (Kümülatif)
- **Yukarı trend**: Kar ediyor ✅
- **Aşağı trend**: Zarar ediyor ❌
- **Yatay**: Sideways, kar/zarar dengede
- **Sıçramalar**: Büyük trade'ler

### Signals Chart
- **Fiyat çizgisi**: BTC fiyat hareketi
- **🟢 Üçgen yukarı**: BUY sinyali
- **🔴 Üçgen aşağı**: SELL sinyali
- Fiyat yükselirken BUY = iyi timing
- Fiyat düşerken SELL = iyi timing

### Win Rate by Regime
- **Bull Market**: Yükseliş piyasası performansı
- **Sideways**: Yatay piyasa performansı
- **Bear Market**: Düşüş piyasası performansı
- **High Volatility**: Volatil piyasa performansı

En yüksek win rate hangi regime'de?
→ O piyasa koşulunda bot daha başarılı!

### PnL Distribution
- **Sıfırın sağında**: Kazanan trade'ler
- **Sıfırın solunda**: Kaybeden trade'ler
- Dağılım geniş mi? → Yüksek volatilite
- Dağılım dar mı? → Düşük volatilite

---

## 🔧 Sorun Giderme

### Dashboard Açılmıyor

```bash
# Streamlit kurulu mu kontrol et
pip install streamlit plotly

# Manuel başlat
cd live_trading
streamlit run dashboard.py
```

### Veri Gösterilmiyor

1. Bot çalışıyor mu kontrol et
2. `data/` klasörü var mı?
   ```bash
   ls -la data/
   ```
3. JSON dosyaları var mı?
   - `trades.json`
   - `signals.json`
   - `performance.json`
   - `bot_status.json`

### Grafikler Boş

- Bot henüz trade yapmadı
- Time range'i değiştir: "All Time"
- Bot'u biraz beklet, sinyal gelmesini sağla

### Port 8501 Kullanımda

```bash
# Farklı port kullan
streamlit run dashboard.py --server.port 8502

# Tarayıcıda aç:
http://localhost:8502
```

---

## 💡 İpuçları

### En İyi Kullanım

1. **İki Monitör**:
   - Bir monitörde bot terminal
   - Diğer monitörde dashboard

2. **Mobil Takip**:
   - Tarayıcıdan mobil cihazla da açılabilir
   - Local network IP ile: `http://192.168.x.x:8501`

3. **Periyodik Kontrol**:
   - Sabah: Win rate'e bak
   - Öğle: PnL grafiğini kontrol et
   - Akşam: Trade geçmişini incele

4. **Analiz**:
   - Hangi saatlerde daha çok trade var?
   - Hangi regime'de win rate yüksek?
   - Average PnL pozitif mi?

### Performance Hedefleri

İyi performans göstergeleri:
- ✅ Win Rate > 55%
- ✅ Sharpe Ratio > 1.5
- ✅ Max Drawdown < %20
- ✅ Total PnL pozitif ve artıyor
- ✅ Her regime'de >50% win rate

---

## 🎨 Özelleştirme

Dashboard kodunu istediğin gibi düzenleyebilirsin:

**Renk değiştir**:
```python
# dashboard.py dosyasında
st.markdown("""
<style>
    .positive { color: #00ff00; }  # Yeşil
    .negative { color: #ff0000; }  # Kırmızı
</style>
""", unsafe_allow_html=True)
```

**Yeni metrik ekle**:
```python
# dashboard.py'de
col6 = st.columns(1)
with col6:
    avg_trade_duration = calculate_avg_duration()
    st.metric("Avg Trade Duration", f"{avg_trade_duration} hours")
```

**Yeni chart ekle**:
```python
# dashboard.py'de
with tab5:
    st.subheader("Hourly Performance")
    # Saatlik kar/zarar grafiği
```

---

## 📱 Remote Access (İsteğe Bağlı)

Dashboard'a başka cihazlardan erişmek için:

### Local Network

```bash
# IP adresini öğren
hostname -I

# Dashboard'u başlat
streamlit run dashboard.py --server.address 0.0.0.0

# Diğer cihazdan aç:
http://192.168.1.XXX:8501
```

### İnternet Üzerinden (ngrok)

```bash
# ngrok kur
pip install pyngrok

# Tunnel oluştur
ngrok http 8501

# Verilen URL'i kullan
https://xxxx-xx-xxx-xxx-xx.ngrok.io
```

**⚠️ GÜVENLİK UYARISI**: İnternet'e açarken dikkatli ol!

---

## 📊 Veri Yapısı

Dashboard şu dosyaları kullanır:

### `data/trades.json`
```json
[
  {
    "timestamp": "2025-11-15T00:03:44",
    "type": "OPEN",
    "side": "LONG",
    "entry_price": 94534.30,
    "quantity": 0.016925,
    "regime": "Sideways",
    "confidence": 0.65
  },
  {
    "timestamp": "2025-11-15T02:15:30",
    "type": "CLOSE",
    "side": "LONG",
    "entry_price": 94534.30,
    "exit_price": 95234.50,
    "quantity": 0.016925,
    "pnl": 118.50,
    "pnl_pct": 0.74,
    "regime": "Bull Market",
    "confidence": 0.72
  }
]
```

### `data/signals.json`
```json
[
  {
    "timestamp": "2025-11-15T00:03:44",
    "signal": 1,
    "signal_name": "BUY",
    "price": 94534.30,
    "confidence": 0.65,
    "regime": "Sideways"
  }
]
```

### `data/performance.json`
```json
{
  "total_trades": 10,
  "winning_trades": 6,
  "losing_trades": 4,
  "total_pnl": 450.25,
  "win_rate": 60.0,
  "sharpe_ratio": 2.1,
  "max_drawdown": 125.50,
  "start_balance": 5000.0,
  "current_balance": 5450.25
}
```

---

## 🎯 Sonraki Adımlar

1. **Bot'u çalıştır**: `./run.sh`
2. **Dashboard'u aç**: `./start_dashboard.sh`
3. **İzle ve öğren**: 24 saat bekle, sonuçları analiz et
4. **Optimize et**: Hangi ayarlar daha iyi çalışıyor?
5. **Gerçek paraya geç**: Test sonuçları iyiyse!

---

**Dashboard ile happy trading! 📊🚀**

*Sorularınız için: Dashboard'da sağ üstteki menüden "Report a bug" seçeneğini kullanabilirsiniz.*
