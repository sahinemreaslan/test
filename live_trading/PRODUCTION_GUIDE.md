# ⚠️ PRODUCTION TRADING GUIDE - GERÇEK PARA

## 🚨 KRİTİK GÜVENLİK UYARILARI

### PAYLAŞTIĞIN API KEY'İ HEMEN SİL!

Sohbette paylaştığın API key'i artık **PUBLIC** durumda! Hemen:

1. **Binance'e git**: https://www.binance.com/en/my/settings/api-management
2. **O API key'i SİL**
3. **YENİ bir API key oluştur**
4. **ASLA paylaşma** (chat, email, screenshot, etc.)

---

## 📋 PRODUCTION'A GEÇMeden ÖNCE MUTLAKA YAP

### 1. Testnet'te Test Et (ZORUNLU!)

```bash
# Önce testnet'te başarılı olmalısın
cd live_trading
./run.sh  # Testnet ile

# En az 24-48 saat çalıştır
# Sonuçları gözlemle
# Win rate, PnL, davranış kontrolü
```

**Testnet başarılı değilse → Production'a geçme!**

### 2. Modeli Tam Veri ile Eğit (ZORUNLU!)

```bash
# 2018-2025 verisiyle eğit
./train_model.sh

# Model kaydedildi mi kontrol et
ls -lh ../models/advanced_system_latest.pkl
```

**Model yoksa → 15 günlük veriye dayanır (KÖTÜ!)**

### 3. Binance Güvenlik Ayarları (ZORUNLU!)

#### a) Yeni API Key Oluştur

1. https://www.binance.com/en/my/settings/api-management
2. **Create API**
3. **Label**: "Bot Trading" (tanımlayıcı bir isim)
4. **Permissions**:
   - ✅ **Enable Futures** (SADECE BU!)
   - ❌ **Enable Spot** (KAPALI!)
   - ❌ **Enable Withdrawals** (KAPALI! ÇOK ÖNEMLİ!)
   - ❌ **Enable Reading** (Opsiyonel, ama yeterli değil)

#### b) IP Whitelist Ekle (ÇOK ÖNEMLİ!)

```bash
# Senin sunucu IP'ni öğren
curl ifconfig.me
```

Binance API ayarlarında:
- **Restrict access to trusted IPs only**: AÇIK
- IP ekle: [Senin IP adresin]

**IP whitelist yoksa → API key çalınırsa her yerden kullanılabilir!**

#### c) 2FA Aktif Et

- Binance hesabında **2FA** mutlaka olsun
- Google Authenticator veya SMS

### 4. .env.production Dosyasını Oluştur

```bash
cd live_trading

# Template'den kopyala
cp .env.production .env.production.backup

# Düzenle
nano .env.production
```

İçine YENİ API keys'lerini ekle:

```bash
BINANCE_API_KEY=yeni_api_key_buraya
BINANCE_API_SECRET=yeni_api_secret_buraya
```

**ASLA:**
- Git'e ekleme
- Paylaşma
- Screenshot alma
- Chat'e yazma

### 5. Config Kontrolü

`config_production.yaml` dosyasını kontrol et:

```yaml
trading:
  leverage: 3                    # ✅ 3x başla (güvenli)
  position_size_pct: 0.03        # ✅ %3 başla (muhafazakar)
  testnet: false                 # ⚠️ GERÇEK PARA!
  paper_trading: false           # ⚠️ GERÇEK EMİRLER!

risk_management:
  max_daily_loss_pct: 0.03       # ✅ Günlük %3 kayıp → dur
  circuit_breaker_loss_pct: 0.15 # ✅ %15 kayıp → otomatik dur
```

**İlk hafta için önerilen:**
- Leverage: 3x (5x değil!)
- Position size: %3 (%8 değil!)
- Daily loss limit: %3

---

## 🚀 PRODUCTION BAŞLATMA

### Adım 1: Tüm Hazırlıkları Kontrol Et

```bash
cd live_trading

# Model var mı?
ls -lh ../models/advanced_system_latest.pkl

# Config doğru mu?
cat config_production.yaml | grep -E "testnet|paper_trading|leverage|position_size"

# .env.production var mı?
ls -lh .env.production

# API keys doğru mu? (placeholderlar değişmiş mi?)
head -5 .env.production
```

### Adım 2: Güvenlik Teyidi

```bash
# Binance'de kontrol et:
# 1. API key oluşturuldu mu? ✓
# 2. Sadece "Futures" izni var mı? ✓
# 3. IP whitelist aktif mi? ✓
# 4. 2FA açık mı? ✓
```

### Adım 3: İLK KÜÇÜK TEST

**ÇOK ÖNEMLİ**: İlk başta KÜÇÜK miktar!

```yaml
# config_production.yaml'de
trading:
  position_size_pct: 0.01  # Sadece %1 ile başla!
```

Örnek: 5000 TL bakiye
- %1 pozisyon = 50 TL
- 3x leverage = 150 TL pozisyon değeri
- İlk trade'de maksimum kayıp: ~15-30 TL (SL'ye göre)

**İlk trade başarılı → sonra artır**

### Adım 4: Production Bot'u Başlat

```bash
./run_production.sh
```

Script sana soracak:
1. Safety checks geçiyor mu?
2. Konfigürasyon doğru mu?
3. Checklist tamamlandı mı?
4. **"START PRODUCTION" yaz** → Başlar

### Adım 5: Dashboard'ları Başlat

**Terminal 1**: Production bot (zaten çalışıyor)

**Terminal 2**: Ana Dashboard
```bash
./start_dashboard.sh
# http://localhost:8501
```

**Terminal 3**: Chart Dashboard
```bash
./start_chart_dashboard.sh
# http://localhost:8502
```

---

## 📊 İLK SAATLER - YAKIN TAKİP

### İlk Trade Geldiğinde

Bot trade açtığında:

1. **Dashboard'u kontrol et**:
   - Pozisyon doğru mu? (LONG/SHORT)
   - Miktar beklendiği gibi mi?
   - SL/TP yerleştirildi mi?

2. **Binance'i kontrol et**:
   - https://www.binance.com/en/futures/BTCUSDT
   - Position tab'ına bak
   - Emir gerçekten açıldı mı?
   - SL ve TP emirleri var mı?

3. **İlk 15 dakika sık kontrol**:
   - Her 5 dakikada bir bak
   - Beklenmedik bir şey var mı?

### İlk Gün

- **Sürekli monitör et** (laptop başında ol)
- Her trade'i izle
- Dashboard'dan PnL takip et
- Anormal bir şey varsa **HEMEN DURDUR** (Ctrl+C)

### İlk Hafta

Günde 3-4 kez kontrol et:
- Sabah: Gece ne olmuş?
- Öğle: Güncel durum?
- Akşam: Gün sonu özeti?

---

## 🛑 EMERGENCY STOP (ACİL DURDURMA)

### Bot'u Durdur

**Terminal'de**: `Ctrl+C`

Bot şunu yapacak:
1. Mevcut döngüyü bitir
2. Açık pozisyonu KAPAT (opsiyonel)
3. Güvenli şekilde kapat

### Manuel Pozisyon Kapatma

Eğer bot dondu/çöktü ve pozisyon hala açık:

1. **Binance web'e git**
2. Futures → Positions
3. **Close Position** → Confirm

### Circuit Breaker

Bot otomatik duracak eğer:
- Günlük kayıp > %3
- Toplam kayıp > %15

Log'da göreceksin:
```
🚨 CIRCUIT BREAKER ACTIVATED!
🛑 Maximum loss reached: -15.2%
🛑 Stopping bot for safety
```

---

## 📈 SONUÇLARI DEĞERLENDİR

### İlk Hafta Sonunda

Dashboard'dan kontrol et:
- **Win Rate**: %55+ ise iyi
- **Total PnL**: Pozitif mi?
- **Max Drawdown**: Kontrolde mi? (<%10)
- **Sharpe Ratio**: >1.5 ise iyi

### Backtest ile Karşılaştır

Production sonuçları backtest'e benziyor mu?

**Benzer ise** ✅:
- Win rate ±5% fark
- Drawdown benzer
- Trade sıklığı benzer
→ **Güvenle devam et**

**Çok farklı ise** ❌:
- Win rate çok düşük
- Drawdown çok yüksek
- Beklenmedik kayıplar
→ **DURDUR ve araştır**

---

## ⚙️ AYARLARI OPTİMİZE ET

### İlk Hafta Başarılı İse

Kademeli artır:

**1. Pozisyon boyutunu artır**:
```yaml
# Hafta 1: %1
position_size_pct: 0.01

# Hafta 2: %3
position_size_pct: 0.03

# Hafta 3-4: %5
position_size_pct: 0.05

# Ay 2: %8 (max)
position_size_pct: 0.08
```

**2. Leverage'ı artır** (opsiyonel):
```yaml
# Ay 1: 3x
leverage: 3

# Ay 2-3: 5x (eğer çok başarılıysa)
leverage: 5
```

**ASLA**:
- Bir anda büyük artış yapma
- %10+ pozisyon kullanma
- 10x+ leverage kullanma

---

## 🔍 SORUN GİDERME

### "API Error 403"

**Sebep**: IP whitelist
**Çözüm**: Binance'de IP'ni ekle

### "Insufficient Balance"

**Sebep**: Yeterli USDT yok
**Çözüm**: Spot → Futures transfer yap

### "Position Not Opening"

**Kontroller**:
1. API key'de "Futures" izni var mı?
2. Bakiye yeterli mi?
3. Minimum trade size üstünde mi? (Binance minimum ~10 USDT)

### "Bot Donuyor"

**İlk yardım**:
```bash
# Ctrl+C ile durdur
# Tekrar başlat
./run_production.sh
```

**Kalıcı sorun**:
- Log'ları incele: `tail -100 logs/production_trading.log`
- Hata mesajlarını bul

---

## 💰 KARLILLIK BEKLENTİLERİ

### Gerçekçi Hedefler

**İyi senaryoda** (backtest ile uyumlu):
- **Aylık**: %20-40
- **Haftalık**: %5-10
- **Günlük**: %0.5-2

**Kötü senaryoda**:
- **Aylık**: %0-10
- Bazı haftalar eksi

**Unutma**:
- Geçmiş performans gelecek garantisi değil
- Piyasa koşulları değişir
- Bazen zarar edersin (normal)

### Drawdown Yönetimi

**Normal drawdown**: %5-10
**Endişe verici**: %10-15
**Tehlikeli**: >%15 (circuit breaker)

---

## 📝 GÜNLÜK RUTİN

### Sabah (09:00)
- Dashboard'a bak
- Gece ne olmuş?
- Açık pozisyon var mı?
- Win rate / PnL kontrolü

### Öğle (14:00)
- Hızlı kontrol
- Anormal bir şey var mı?

### Akşam (21:00)
- Detaylı analiz
- Günlük özet
- Dashboard'u export et

### Hafta Sonu
- Haftalık rapor
- Backtest ile karşılaştır
- Ayarları gözden geçir

---

## 🎯 BAŞARI KONTROL LİSTESİ

Production'da başarılı olmak için:

- [ ] Testnet'te en az 48 saat başarıyla çalıştı
- [ ] Model 2018-2025 verisiyle eğitildi
- [ ] Yeni API key oluşturuldu (eskisi silindi!)
- [ ] IP whitelist aktif
- [ ] Sadece "Futures" izni verildi (withdrawal KAPALI!)
- [ ] 2FA aktif
- [ ] .env.production dosyası oluşturuldu
- [ ] Küçük pozisyon ile başladım (%1-3)
- [ ] Düşük leverage ile başladım (3x)
- [ ] İlk saatlerde sürekli izliyorum
- [ ] Dashboard'ları kurdum
- [ ] Acil durdurma planım var
- [ ] Sadece kaybedebileceğim kadar para kullanıyorum

---

## ⚠️ SON UYARILAR

1. **Kripto futures trading çok riskli**
2. **Kaldıraç tüm paranı sıfırlayabilir**
3. **Bot %100 garantili değil**
4. **Piyasa koşulları değişir**
5. **Testnet başarısı ≠ Production başarısı garanti etmez**
6. **Sadece kaybedebileceğin kadar yatır**
7. **Düzenli takip et, "kur unut" yapma**
8. **İlk kayıp sinyalinde stratejiyi gözden geçir**

---

## 📞 DESTEK

### Sorun mu var?

1. **Log'ları kontrol et**: `logs/production_trading.log`
2. **Binance'i kontrol et**: Pozisyonlar, emirler
3. **Dashboard'a bak**: Hata var mı?

### Bot'u durdur ve araştır eğer:

- Beklenmedik kayıplar
- Çok fazla trade (her 15 dakikada olmamalı!)
- SL/TP çalışmıyor
- Drawdown >%15

---

**BAŞARILAR VE GÜVENLİ TRADİNGLER! 🚀**

*Unutma: Bu bir robot, sen kontroldesin. Her zaman sen karar veriyorsun.*
