# 🚗 Çoklu Doğrusal Regresyon (MLR) ile Araç Fiyatı Tahmini

Bu proje, kullanılmış araçların çeşitli özelliklerini (Yaş, KM, Yakıt Tipi vb.) kullanarak piyasa değerini tahmin eden bir makine öğrenmesi uygulamasıdır. 

## 📝 Proje Bilgileri
* **Ad Soyad:** [Gamze Büyüktorun]
* **Numara:** [2212721033]
* **Ders:** [Makine Öğrenmesi]

## 🛠️ Kullanılan Teknolojiler
* **Python** (Veri işleme ve modelleme)
* **Google Colab** (Model eğitimi ve Backward Elimination)
* **Scikit-Learn** (MLR Modeli)
* **Statsmodels** (İstatistiksel analiz - p-value)
* **Flask** (Web Arayüzü)
* **Pandas & Numpy** (Veri analizi)

## 📊 Veri Seti ve Ön İşleme
Projede Kaggle'dan alınan "Car Dekho" veri seti kullanılmıştır. Uygulanan adımlar:
1. **Veri Temizleme:** Eksik veriler kontrol edildi.
2. **Feature Engineering:** `Year` sütunu kullanılarak `Age` (Yaş) özelliği türetildi.
3. **Encoding:** Kategorik değişkenler (Yakıt, Vites, Satıcı Tipi) **One-Hot Encoding** yöntemiyle sayısal hale getirildi.
4. **Scaling:** Sayısal veriler **StandardScaler** ile ölçeklendirildi.

## 📉 Modelleme: Backward Elimination
Modelin başarısını artırmak için **Geriye Doğru Eleme (Backward Elimination)** yöntemi kullanılmıştır. 
* Başlangıçta tüm öznitelikler modele dahil edildi.
* **P-value > 0.05** olan anlamlılık düzeyi düşük öznitelikler (`Fuel_Type_Petrol` ve `Owner`) elendi.
* Final model, sadece istatistiksel olarak anlamlı 6 öznitelik ile eğitildi.

## 🏆 Model Performansı
* **R² (R Kare):** 0.8543 (%85 Başarı)
* **MAE (Ortalama Mutlak Hata):** 1.1860

## 🚀 Uygulamayı Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Gerekli kütüphaneleri kurun:
   ```bash
   pip install flask pandas numpy scikit-learn joblib
2. Terminalden uygulamayı başlatın:
   python app.py

 

