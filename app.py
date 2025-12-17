import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# 1. Kayıtlı Nesneleri Yükleme
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    final_features_names = joblib.load('final_features.pkl')
    
    # 'const' varsa kaldır
    if 'const' in final_features_names:
        final_features_names = [f for f in final_features_names if f != 'const']
    
    print("=" * 70)
    print("✅ Modeller başarıyla yüklendi.")
    print(f"📋 Final features ({len(final_features_names)} adet):")
    for i, f in enumerate(final_features_names, 1):
        print(f"   {i}. {f}")
    
    # UYARI: Owner kontrolü
    if 'Owner' not in final_features_names:
        print("\n⚠️  UYARI: 'Owner' sütunu final_features'da YOK!")
        print("   Model eğitimi sırasında Backward Elimination tarafından elendi.")
        print("   Owner değeri tahminleri ETKİLEMEYECEK.")
    
    print("=" * 70)
except FileNotFoundError as e:
    print(f"❌ HATA: {e} dosyası bulunamadı.")
    exit()

app = Flask(__name__)

# Kullanıcıdan alınacak orijinal özellikler
ORIGINAL_FEATURES = ['Present_Price_Lakh', 'Kms', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Age']
CATEGORICAL_FEATURES = ['Fuel_Type', 'Seller_Type', 'Transmission']
NUMERIC_FEATURES_TO_SCALE = ['Present_Price_Lakh', 'Kms', 'Age']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = request.form
        
        print("\n" + "=" * 70)
        print("🔍 YENİ TAHMİN İSTEĞİ")
        print("=" * 70)
        
        # 1. Ham Veriyi Çekme
        raw_data = {}
        for feature in ORIGINAL_FEATURES:
            value = form_values.get(feature)
            
            if value is None or value == '':
                raise ValueError(f"'{feature}' alanı formdan eksik gelmiştir.")

            if feature not in CATEGORICAL_FEATURES:
                raw_data[feature] = float(value)
            else:
                raw_data[feature] = value
        
        print("\n1️⃣ HAM VERİ (Formdan Gelen):")
        for key, val in raw_data.items():
            print(f"   {key}: {val}")
        
        # 2. Manuel One-Hot Encoding (Daha güvenilir)
        encoded_data = {
            'Present_Price_Lakh': raw_data['Present_Price_Lakh'],
            'Kms': raw_data['Kms'],
            'Age': raw_data['Age'],
        }
        
        # Fuel_Type encoding
        fuel_type = raw_data['Fuel_Type']
        encoded_data['Fuel_Type_Diesel'] = 1 if fuel_type == 'Diesel' else 0
        encoded_data['Fuel_Type_Petrol'] = 1 if fuel_type == 'Petrol' else 0
        encoded_data['Fuel_Type_CNG'] = 1 if fuel_type == 'CNG' else 0
        
        # Seller_Type encoding
        seller_type = raw_data['Seller_Type']
        encoded_data['Seller_Type_Dealer'] = 1 if seller_type == 'Dealer' else 0
        encoded_data['Seller_Type_Individual'] = 1 if seller_type == 'Individual' else 0
        
        # Transmission encoding
        transmission = raw_data['Transmission']
        encoded_data['Transmission_Manual'] = 1 if transmission == 'Manual' else 0
        encoded_data['Transmission_Automatic'] = 1 if transmission == 'Automatic' else 0
        
        print("\n2️⃣ ONE-HOT ENCODING SONRASI:")
        for key, val in encoded_data.items():
            print(f"   {key}: {val}")
        
        # 3. Final Features'a Göre Düzenleme
        processed_df = pd.DataFrame()
        
        for feature in final_features_names:
            if feature in encoded_data:
                processed_df[feature] = [encoded_data[feature]]
            else:
                # Model bu özelliği bekliyor ama elimizde yok -> 0 ata
                processed_df[feature] = [0]
        
        print("\n3️⃣ FINAL FEATURES'A GÖRE DÜZENLENMIŞ VERİ:")
        print(processed_df.to_string())
        
        # 4. Sayısal Verileri Ölçekleme
        cols_to_scale = [col for col in NUMERIC_FEATURES_TO_SCALE if col in processed_df.columns]
        
        print(f"\n4️⃣ ÖLÇEKLEME:")
        print(f"   Ölçeklenecek sütunlar: {cols_to_scale}")
        
        if cols_to_scale:
            print(f"   Ölçekleme öncesi değerler:")
            for col in cols_to_scale:
                print(f"      {col}: {processed_df[col].iloc[0]}")
            
            processed_df[cols_to_scale] = scaler.transform(processed_df[cols_to_scale])
            
            print(f"   Ölçekleme sonrası değerler:")
            for col in cols_to_scale:
                print(f"      {col}: {processed_df[col].iloc[0]:.6f}")
        
        # 5. Sütun Sırasını Düzenle ve Tahmin Yap
        X_predict = processed_df[final_features_names]
        
        print("\n5️⃣ MODELE GİDEN FINAL VERİ:")
        print(X_predict.to_string())
        
        # 6. Tahmin
        prediction = model.predict(X_predict)
        
        # TL'ye çevirme
        output = f"{prediction[0] * 100000:,.2f} TL"
        
        print(f"\n6️⃣ TAHMİN SONUCU:")
        print(f"   Lakh: {prediction[0]:.4f}")
        print(f"   TL: {output}")
        print("=" * 70 + "\n")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("\n" + "=" * 70)
        print("❌ HATA OLUŞTU:")
        print(error_msg)
        print("=" * 70 + "\n")
        return render_template('index.html', 
                             prediction_text=f"Tahmin Hatası: {str(e)}",
                             form_data=request.form)

    return render_template('index.html', 
                         prediction_text=f'Tahmin Edilen Satış Fiyatı: {output}',
                         form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True)