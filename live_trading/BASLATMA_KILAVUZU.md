# 🚀 Hızlı Başlangıç Kılavuzu

Bitcoin Live Trading Bot - Türkçe Başlangıç Rehberi

---

## ⚡ 5 Dakikada Başlat

### 1. Kurulum

```bash
cd live_trading
pip install -r requirements.txt
```

### 2. API Anahtarlarını Al

**TESTNET (Önce bununla başla!):**
1. https://testnet.binancefuture.com/ adresine git
2. Email ile giriş yap
3. Sağ üstteki "API Key" butonuna tık
4. Yeni API key oluştur
5. API Key ve Secret'i kaydet

### 3. .env Dosyasını Oluştur

```bash
cp .env.example .env
nano .env  # veya herhangi bir editör
```

`.env` dosyasına API anahtarlarını ekle:
```
BINANCE_API_KEY=senin_api_key_buraya
BINANCE_API_SECRET=senin_secret_buraya
```

### 4. Bağlantıyı Test Et

```bash
python test_connection.py
```

Herşey OK görünüyorsa:

### 5. Botu Başlat!

```bash
python live_trader.py
```

---

## 🎓 Model Eğitimi: İki Yöntem

### ⚠️ ÖNEMLİ: Eğitim Verisi Farkı

Bot iki şekilde çalışabilir:

**Yöntem 1: Canlı Eğitim** (Varsayılan)
- Binance API'den son 1500 mum çeker (~15 gün)
- Her başlatmada yeniden eğitir
- Hızlı başlangıç ama sınırlı veri

**Yöntem 2: Önceden Eğitilmiş Model** (ÖNERİLEN!)
- 2018-2025 arası TÜM veriyle eğitilmiş (7 yıl!)
- Backtest ile aynı model
- Tutarlı sonuçlar, güçlü performans

### 🎯 Önerilen Yol: Önceden Eğitilmiş Model

**Adım 1: Modeli Eğit (Bir kerelik)**

```bash
cd live_trading
python train_offline.py --csv ../btc_15m_data_2018_to_2025.csv
```

Çıktı:
```
🎓 OFFLINE MODEL TRAINING
📊 Loading historical data...
✅ Loaded 245678 candles (2018-01-01 to 2025-11-14)
⏱️ Converting to multiple timeframes...
🔬 Processing indicators...
🧬 Creating features...
📚 Preparing ML dataset...
🎓 Training advanced system...
💾 Saving models...
✅ Saved: ../models/advanced_system_latest.pkl
✅ TRAINING COMPLETE!
```

Model şuraya kaydedilir:
- `../models/advanced_system_latest.pkl` (her zaman son model)
- `../models/advanced_system_YYYYMMDD_HHMMSS.pkl` (yedek)

**Adım 2: Modeli Kullan**

```bash
python live_trader.py --model ../models/advanced_system_latest.pkl
```

Çıktı:
```
🤖 BITCOIN LIVE TRADING BOT INITIALIZED
🚀 INITIALIZING TRADING BOT
📦 Loading pre-trained model from: ../models/advanced_system_latest.pkl
✅ Pre-trained model loaded successfully!
✅ INITIALIZATION COMPLETE!
```

### 📊 Karşılaştırma

| Özellik | Canlı Eğitim | Önceden Eğitilmiş |
|---------|--------------|-------------------|
| **Veri** | 15 gün (1500 mum) | 7 yıl (245K+ mum) |
| **Başlangıç** | python live_trader.py | python live_trader.py --model ../models/advanced_system_latest.pkl |
| **Eğitim süresi** | 2-5 dakika her başlatmada | Bir kez 10-20 dakika |
| **Backtest tutarlılığı** | ❌ Farklı | ✅ Aynı |
| **Güçlü performans** | ⚠️ Sınırlı | ✅ Çok güçlü |
| **Ne zaman kullan** | Hızlı test | Gerçek trading |

### 💡 Öneri

1. **İlk test için:** Canlı eğitim (varsayılan) kullan, sistemi tanı
2. **Gerçek trading için:** Önceden eğitilmiş model kullan
3. **Model güncelleme:** Ayda bir yeniden eğit (yeni verilerle)

---

## 🎯 5,000 TL ile Başlangıç (Senin Planın)

### Önerilen Ayarlar

`config_live.yaml` dosyasını aç:

```yaml
trading:
  # Kaldıraç (3x güvenli, 5x agresif, 7x çok riskli)
  leverage: 5

  # Pozisyon büyüklüğü (5,000 TL'nin %8'i = 400 TL)
  position_size_pct: 0.08

  # İLK BAŞTA MUTLAKA TRUE YAPPPPP!
  testnet: true          # Sahte parayla test
  paper_trading: true    # Emir yerleştirmeden simülasyon
```

### Aşamalı İlerleme

**Hafta 1: Testnet + Paper Trading**
```yaml
testnet: true
paper_trading: true
leverage: 3
```
→ Botu tanı, sinyalleri izle, riski anla

**Hafta 2: Testnet + Gerçek Emirler**
```yaml
testnet: true
paper_trading: false    # Testnet'te gerçek emir
leverage: 5
```
→ Emir sistemini test et, performansı gör

**Hafta 3: Gerçek Para (Küçük)**
```yaml
testnet: false          # GERÇEK PARA!
paper_trading: false
leverage: 3             # İlk başta küçük
position_size_pct: 0.03 # %3'le başla
```
→ Gerçek parayla güven kazan

**Hafta 4+: Tam Strateji**
```yaml
testnet: false
paper_trading: false
leverage: 5             # Agresif büyüme
position_size_pct: 0.08 # %8 pozisyon
```
→ Hızlı büyüme moduna geç!

---

## 💰 Beklenen Kazanç (5,000 TL Başlangıç)

### Smart-Aggressive Ayarlar (Önerilen)

Leverage: 5x, Position: %8

| Ay | Muhafazakar (%25) | Agresif (%50) | Hiper-Agresif (%100) |
|----|-------------------|---------------|----------------------|
| 0  | 5,000 TL          | 5,000 TL      | 5,000 TL             |
| 1  | 6,250 TL          | 7,500 TL      | 10,000 TL            |
| 2  | 7,812 TL          | 11,250 TL     | 20,000 TL            |
| 3  | 9,765 TL          | 16,875 TL     | 40,000 TL            |
| 6  | 19,073 TL         | 56,953 TL     | 320,000 TL           |

**GERÇEKÇI OL:**
- Bunlar İDEAL senaryolar
- Kaybettiğin aylar da olacak
- 2023 gibi yatay piyasalarda düşük getiri
- Kaldıraç = Risk!
- Düzenli kar çek

---

## 📊 Ayarları Özelleştirme

### Daha Güvenli (Risk düşür)

```yaml
trading:
  leverage: 3              # Düşük kaldıraç
  position_size_pct: 0.05  # Küçük pozisyon (%5)

advanced_features:
  enable_trailing_stop: true
  trailing_stop_pct: 0.03  # Geniş trailing (%3)
  enable_partial_exit: true
  partial_exit_percentage: 0.7  # Erkenden %70 kapat
  enable_position_scaling: false # Pyramiding kapalı
```

### Daha Agresif (Hızlı büyüme)

```yaml
trading:
  leverage: 7              # Yüksek kaldıraç
  position_size_pct: 0.12  # Büyük pozisyon (%12)

advanced_features:
  enable_trailing_stop: true
  trailing_stop_pct: 0.015 # Dar trailing (%1.5)
  enable_partial_exit: true
  partial_exit_percentage: 0.3  # Sadece %30 kapat
  enable_position_scaling: true
  max_scale_ins: 3         # 3'e kadar ekle
```

⚠️ **UYARI:** Agresif ayarlar = Yüksek risk!

---

## 🔒 Güvenlik

### API Anahtarı Güvenliği

1. **Binance'de ayarlar:**
   - ✅ Sadece "Futures Trading" izni ver
   - ❌ "Withdrawal" iznini ASLA açma
   - ✅ IP whitelist kullan
   - ✅ 2FA aç

2. **Dosya güvenliği:**
   - .env dosyasını kimseyle paylaşma
   - .env dosyasını git'e gönderme
   - API keylerini ekran görüntüsünde gösterme

### Para Yönetimi

1. **Küçük başla, yavaş büyüt**
2. **İlk kazançları çek** (sermayeni geri al)
3. **Stop loss ayarla** (circuit breaker)
4. **Kaybedebileceğinden fazlasını yatırma**

---

## 🐛 Sorun Giderme

### "API Key bulunamadı"

```bash
# .env dosyasını kontrol et
cat .env

# Şöyle görünmeli:
BINANCE_API_KEY=xxx...
BINANCE_API_SECRET=xxx...
```

### "Yetersiz bakiye"

1. Binance Futures'a git
2. Spot'tan Futures'a USDT transfer et
3. Minimum: Testnet için $50, gerçek için $100+

### "Pozisyon açılamadı"

1. API key izinlerini kontrol et
2. Bakiye yeterli mi kontrol et
3. Log dosyasına bak: `logs/live_trading.log`
4. İlk başta `paper_trading: true` dene

---

## ⚙️ Önemli Komutlar

```bash
# Kurulum
./setup.sh

# Bağlantı testi
python test_connection.py

# MODEL EĞİTİMİ (Önerilen - bir kerelik)
python train_offline.py --csv ../btc_15m_data_2018_to_2025.csv

# Botu başlat (canlı eğitim - varsayılan)
python live_trader.py

# Botu başlat (önceden eğitilmiş model - önerilen)
python live_trader.py --model ../models/advanced_system_latest.pkl

# Botu durdur
Ctrl+C

# Log'ları izle
tail -f logs/live_trading.log

# Gerçek zamanlı takip
watch -n 5 'tail -20 logs/live_trading.log'
```

---

## 📞 Yardım

### Dökümanlar
- `README.md` - Detaylı İngilizce kılavuz
- `ADVANCED_FEATURES.md` - Özellik açıklamaları
- `STRATEGY_IMPROVEMENTS.md` - Performans analizi

### Binance
- Testnet: https://testnet.binancefuture.com/
- API Docs: https://binance-docs.github.io/apidocs/futures/en/
- Durum: https://www.binance.com/en/support/announcement

### Loglar
- Konsol çıktısı (anlık)
- `logs/live_trading.log` (detaylı)

---

## ✅ Kontrol Listesi

Başlamadan önce:

- [ ] Gereksinimleri yükledim (`pip install -r requirements.txt`)
- [ ] Testnet'ten API key aldım
- [ ] `.env` dosyasını oluşturdum
- [ ] API keylerimi `.env`'e ekledim
- [ ] `test_connection.py` çalıştırdım (BAŞARILI)
- [ ] `config_live.yaml`'da `testnet: true` ve `paper_trading: true` yaptım
- [ ] **(Opsiyonel ama önerilen)** Modeli offline eğittim (`python train_offline.py --csv ../btc_15m_data_2018_to_2025.csv`)
- [ ] Riskleri anladım
- [ ] Kaybedebileceğimden fazlasını yatırmayacağım

**Başlatma komutları:**

Hızlı test (canlı eğitim):
```bash
python live_trader.py
```

Gerçek trading (önceden eğitilmiş model):
```bash
python live_trader.py --model ../models/advanced_system_latest.pkl
```

---

## ⚠️ SON UYARILAR

1. **Kripto trading risklidir**
2. **Sadece kaybedebileceğin kadar yatır**
3. **Kaldıraç tehlikelidir** - hesabını sıfırlayabilir
4. **Geçmiş performans gelecek getiriyi garanti etmez**
5. **İlk başta küçük başla**
6. **Botu düzenli takip et** - "kur unut" yapma

---

## 🎯 Başarı İçin İpuçları

1. **Sabırlı ol** - Zengin olmak bir süreç
2. **Disiplinli ol** - Kurallara uy
3. **Duygusal olma** - Sisteme güven
4. **Risk yönet** - Her zaman stop loss kullan
5. **Düzenli kar çek** - Kazandığında çek, birikim yap
6. **Öğrenmeye devam et** - Piyasayı takip et

---

**Bol kazançlar! 🚀💰**

*Unutma: Trading bir maraton, sprint değil. Hızlı ve sürdürülebilir büyümeye odaklan, "hemen zengin ol" değil. Strateji güçlü ama başarı = Disiplin + Risk yönetimi.*
