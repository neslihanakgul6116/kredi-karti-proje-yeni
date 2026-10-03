import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Kurumsal Kredi Kartı Dolandırıcılık Tespiti", 
    page_icon="💳", 
    layout="wide"
)

# Kurumsal Başlık ve Açıklama
st.markdown("## 🚨 Kurumsal Kredi Kartı Dolandırıcılık Tespiti & Risk Yönetim Paneli")
st.markdown("Bu uygulama; makine öğrenmesi algoritmaları, risk skoru hesaplaması ve canlı simülasyonlar ile entegre profesyonel bir sahtekarlık tespit sistemidir.")

@st.cache_data
def veri_yukle():
    try:
        df = pd.read_csv("creditcard.csv")
        return df
    except FileNotFoundError:
        return None

with st.spinner("Veri yükleniyor, lütfen bekleyin..."):
    df = veri_yukle()

if df is not None:
    # Üst Kurumsal Metrik Kartları
    toplam_islem = len(df)
    fraud_vaka = int(df['Class'].sum())
    riskli_tutar = df[df['Class'] == 1]['Amount'].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam İşlem Hacmi", f"{toplam_islem:,}".replace(",", "."))
    with col2:
        st.metric("Gerçek Dolandırıcılık Vakası", f"{fraud_vaka:,}".replace(",", "."))
    with col3:
        st.metric("Potansiyel Riskli Tutar ($)", f"{riskli_tutar:,.2f} ABD doları")

    st.markdown("---")

    # İlk 5 satır önizlemesi
    if st.checkbox("Veri setinin ilk 5 satırını göster"):
        st.dataframe(df.head(), use_container_width=True)

    # İki sütun oluşturarak grafikleri yan yana yerleştiriyoruz
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📊 İşlem Sınıf Dağılımı")
        fraud_count = df['Class'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(['Normal (0)', 'Fraud (1)'], fraud_count, color=['#4CAF50', '#F44336'])
        ax.set_ylabel("İşlem Sayısı")
        st.pyplot(fig)

    # Modeli hafızada tutarak hızlı eğitmek için fonksiyon
    @st.cache_resource
    def model_egit(data):
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, X_test, y_test, X.columns

    # Model Eğitimi Butonu
    if st.button("🚀 Modeli Eğit ve Performansı Göster"):
        with st.spinner("Model eğitiliyor ve analiz ediliyor..."):
            model, X_test, y_test, feature_names = model_egit(df)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            st.subheader("🎯 Model Performans Raporu")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Dolandırıcılık Yakalama (Recall)", f"%{report['1']['recall']*100:.1f}")
            with mcol2:
                st.metric("Kesinlik (Precision)", f"%{report['1']['precision']*100:.1f}")
            with mcol3:
                st.metric("F1-Skoru", f"{report['1']['f1-score']:.2f}")
            
            # Gelişmiş Özellik Önem Düzeyi Grafiği (Daha kompakt boyutta)
            with col_g2:
                st.subheader("📈 En Önemli 5 Değişken")
                importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(5)
                fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
                importances.plot(kind='barh', ax=ax_imp, color='teal')
                ax_imp.set_xlabel("Önem Derecesi")
                st.pyplot(fig_imp)
            
            st.success("Model eğitimi ve analiz başarıyla tamamlandı!")

    # --- İNTERAKTİF TEST PANELİ ---
    st.markdown("---")
    st.subheader("🔍 Canlı İşlem Test Etme Paneli")
    st.write("Veri setinden rastgele bir işlem seçip yapay zekanın doğru tahmin yapıp yapamadığını test edin.")

    islem_turu = st.selectbox(
        "Test etmek istediğiniz işlem türünü seçin:",
        ("Normal İşlem (0)", "Dolandırıcılık İşlemi (1)")
    )

    if st.button("İşlemi Test Et"):
        model, X_test, y_test, _ = model_egit(df)
        
        if "Normal" in islem_turu:
            ornek = df[df['Class'] == 0].sample(1)
        else:
            ornek = df[df['Class'] == 1].sample(1)
            
        gercek_durum = ornek['Class'].values[0]
        girdi_verisi = ornek.drop('Class', axis=1)
        
        tahmin = model.predict(girdi_verisi)[0]
        
        st.write(f"**Seçilen İşlemin Tutarı:** {ornek['Amount'].values[0]} TL/Dolar")
        
        if tahmin == 1:
            st.error("🚨 Yapay Zeka Tahmini: Bu işlem **DOLANDIRICILIK (Fraud)**!")
        else:
            st.success("✅ Yapay Zeka Tahmini: Bu işlem **NORMAL**.")
            
        if gercek_durum == tahmin:
            st.info("🎯 Sonuç: Yapay zeka bu işlemi **DOĞRU** bildi!")
        else:
            st.warning("⚠️ Sonuç: Yapay zeka bu işlemde **yanıldı**.")
else:
    st.warning("⚠️ 'creditcard.csv' dosyası proje klasöründe bulunamadı! Lütfen dosyanın bu script ile aynı klasörde olduğundan emin olun.")