# 💰 Kaldıraç (Leverage) ve Komisyon Sistemi

## 🎯 Özellikler

Sistem artık gerçekçi trading koşullarını simüle ediyor:

✅ **Kaldıraç Desteği:** 1x - 10x arası ayarlanabilir
✅ **Komisyon:** %0.1 (Binance ortalaması)
✅ **Slippage:** %0.05 (market impact)
✅ **Liquidation Takibi:** Otomatik liquidation fiyatı hesaplama
✅ **Margin Yönetimi:** İlk margin ve bakım marjı

---

## ⚙️ Konfigürasyon

`config.yaml` dosyasında ayarlar:

```yaml
backtesting:
  initial_capital: 10000

  # Trading costs
  commission: 0.001  # 0.1% (maker/taker average)
  slippage: 0.0005   # 0.05% (market impact)

  # Leverage settings
  leverage: 1        # BURADAN DEĞİŞTİR: 1x, 2x, 3x, 5x, 10x
  max_leverage: 10

  # Margin requirements
  maintenance_margin: 0.05  # 5% (liquidation seviyesi)
  initial_margin_ratio: 0.10  # 10% (minimum gerekli)
```

---

## 📊 Kaldıraç Nasıl Çalışır?

### **Örnek 1: Leverage 1x (Yok)**

```
Capital: $10,000
Position Size: 5% = $500
Leverage: 1x
Gerçek Position: $500
Margin Used: $500
Commission: $500 * 0.1% = $0.50
```

**Sonuç:**
- %10 kazanç → $50 kar (sermayenin %0.5'i)
- %10 kayıp → $50 zarar (sermayenin %0.5'i)

### **Örnek 2: Leverage 5x**

```
Capital: $10,000
Position Size: 5% = $500 (margin)
Leverage: 5x
Gerçek Position: $500 * 5 = $2,500
Margin Used: $500
Commission: $2,500 * 0.1% = $2.50
```

**Sonuç:**
- %10 kazanç → $250 kar (sermayenin %2.5'i) 🚀
- %10 kayıp → $250 zarar (sermayenin %2.5'i) ⚠️
- %20 kayıp → **LİKİDASYON** 💀

### **Örnek 3: Leverage 10x**

```
Capital: $10,000
Position Size: 5% = $500 (margin)
Leverage: 10x
Gerçek Position: $500 * 10 = $5,000
Margin Used: $500
Commission: $5,000 * 0.1% = $5.00
```

**Sonuç:**
- %10 kazanç → $500 kar (sermayenin %5'i) 🚀🚀
- %5 kayıp → $250 zarar (sermayenin %2.5'i) ⚠️
- %10 kayıp → **LİKİDASYON** 💀💀

---

## ⚠️ Liquidation (Tasfiye) Fiyatı

### **Long Position:**
```
Entry: $50,000
Leverage: 5x
Liquidation = $50,000 * (1 - (1/5 - 0.05))
            = $50,000 * (1 - 0.15)
            = $42,500 (-15%)
```

### **Short Position:**
```
Entry: $50,000
Leverage: 5x
Liquidation = $50,000 * (1 + (1/5 - 0.05))
            = $50,000 * (1 + 0.15)
            = $57,500 (+15%)
```

---

## 🧪 Test Senaryoları

### **Senaryo 1: Leverage 1x (Mevcut)**

```bash
# config.yaml'de leverage: 1
python main.py --use-advanced
```

**Beklenen Sonuçlar:**
- Return: ~20-25% yıllık
- Max DD: ~1-2%
- Liquidation: 0
- **Güvenli, düşük risk**

### **Senaryo 2: Leverage 2x (Orta Risk)**

```bash
# config.yaml'de leverage: 2
python main.py --use-advanced
```

**Beklenen Sonuçlar:**
- Return: ~40-50% yıllık (2x)
- Max DD: ~3-5%
- Liquidation: 0-2
- **Orta risk, iyi potansiyel**

### **Senaryo 3: Leverage 5x (Yüksek Risk)**

```bash
# config.yaml'de leverage: 5
python main.py --use-advanced
```

**Beklenen Sonuçlar:**
- Return: ~100-125% yıllık (5x)
- Max DD: ~10-15%
- Liquidation: 5-15
- **Yüksek risk, yüksek getiri**

### **Senaryo 4: Leverage 10x (Çok Yüksek Risk)**

```bash
# config.yaml'de leverage: 10
python main.py --use-advanced
```

**Beklenen Sonuçlar:**
- Return: ~200% YA DA **-100%** (10x)
- Max DD: ~25-50%
- Liquidation: 15-50
- **ÇOK RİSKLİ, sadece test için!**

---

## 📈 Karşılaştırmalı Analiz

Farklı leverage seviyelerini karşılaştırmak için:

```bash
# 1x
sed -i 's/leverage: .*/leverage: 1/' config.yaml
python walk_forward_analysis.py --use-advanced --train-test

# 2x
sed -i 's/leverage: .*/leverage: 2/' config.yaml
python walk_forward_analysis.py --use-advanced --train-test

# 5x
sed -i 's/leverage: .*/leverage: 5/' config.yaml
python walk_forward_analysis.py --use-advanced --train-test
```

---

## 💡 Öneriler

### **Leverage 1x - Konservatif**
✅ Sermaye koruma
✅ Düşük risk
✅ Tutarlı getiri
❌ Düşük kar potansiyeli

**Kimler için:** Risk-averse yatırımcılar, başlangıç

### **Leverage 2-3x - Dengeli**
✅ İyi risk/getiri dengesi
✅ Kabul edilebilir risk
✅ Orta-iyi getiri
⚠️ Bazı liquidation riski

**Kimler için:** Deneyimli trader'lar, orta risk toleransı

### **Leverage 5-10x - Agresif**
🚀 Çok yüksek getiri potansiyeli
💀 Çok yüksek liquidation riski
❌ Volatilite'de hızlı kayıplar

**Kimler için:** Çok deneyimli trader'lar, yüksek risk toleransı

---

## 🎯 Komisyon ve Maliyet Etkisi

### **Komisyon Hesabı:**

```python
# Her trade için:
Entry commission = Position Value * 0.1%
Exit commission = Position Value * 0.1%
Total commission per trade = Position Value * 0.2%

# Leverage ile:
1x: $500 position → $1 toplam komisyon
5x: $2,500 position → $5 toplam komisyon
10x: $5,000 position → $10 toplam komisyon
```

### **Yıllık Komisyon Maliyeti:**

```
2000 trades/year:
- 1x leverage: ~$2,000 komisyon
- 5x leverage: ~$10,000 komisyon
- 10x leverage: ~$20,000 komisyon
```

**ÖNEMLİ:** Leverage arttıkça komisyon maliyeti de artar!

---

## 📊 Beklenen Sonuç Değişiklikleri

### **Leverage 1x → 5x Değişimi:**

| Metrik | 1x | 5x | Değişim |
|--------|----|----|---------|
| Return | 20% | ~80-100% | +4-5x |
| Sharpe | 15 | ~5-10 | -50% |
| Max DD | 1% | ~10% | +10x |
| Liquidations | 0 | 10-20 | +∞ |
| Win Rate | 82% | ~65-75% | -10-15% |

**Analiz:**
- Getiri artıyor ✅
- Risk çok artıyor ⚠️
- Risk-adjusted return düşüyor ⚠️

---

## 🚨 Uyarılar

1. **Yüksek leverage = Yüksek risk**
   - 5x leverage ile %20 düşüş = Liquidation
   - 10x leverage ile %10 düşüş = Liquidation

2. **Komisyon Etkisi:**
   - Leverage arttıkça komisyon da artar
   - Çok işlem yapan stratejilerde ciddi maliyet

3. **Backtesting vs Real Trading:**
   - Backtestte perfect execution
   - Gerçekte slippage daha yüksek olabilir
   - Liquidation gerçekte daha hızlı olur

4. **Bear Market'te:**
   - Leverage kullanımı çok riskli
   - 2022'de 5x leverage ile ciddi kayıplar
   - Sideways market'te de riskli

---

## 📚 Sonuç

**Leverage bir çoklayıcıdır - hem kar hem zararı büyütür!**

### **Tavsiye Edilen:**
- Başlangıç: **1x** (leverage yok)
- Deneyimli: **2-3x** (dengeli)
- Uzman: **Maksimum 5x** (dikkatli)
- **10x:** Sadece test/eğitim amaçlı

### **Gerçek Trading İçin:**
- Paper trading ile başla
- Düşük leverage kullan (1-2x)
- Liquidation mesafesini izle
- Stop loss koy
- Sermaye yönetimi uygula

---

**Unutma:** Leverage ateşle oynamak gibidir. Kontrollü kullanılmazsa yakar! 🔥
