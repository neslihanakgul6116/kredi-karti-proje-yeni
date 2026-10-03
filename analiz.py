import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="Kredi Kartı Dolandırıcılık Tespiti & Risk Paneli",
    page_icon="🚨",
    layout="wide",
)

# Web sayfası başlığı
st.title("🚨 Kurumsal Kredi Kartı Dolandırıcılık Tespiti & Risk Yönetim Paneli")
st.write(
    "Bu uygulama; makine öğrenmesi algoritmaları, risk skoru hesaplama,"
    " çoklu model kıyaslaması ve canlı akış simülasyonu ile donatılmış"
    " profesyonel bir fraud tespit sistemidir."
)


@st.cache_data
def veri_yukle():
    # creditcard.csv dosyasının proje klasöründe olduğundan emin olun
    df = pd.read_csv("creditcard.csv")
    return df


with st.spinner("Veri yükleniyor, lütfen bekleyin..."):
    df = veri_yukle()

# --- ÜST ÖZET KPI KARTLARI ---
toplam_islem = len(df)
gercek_fraud = int(df["Class"].sum())
toplam_fraud_tutar = float(df[df["Class"] == 1]["Amount"].sum())

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric(label="Toplam İşlem Hacmi", value=f"{toplam_islem:,}")
col_kpi2.metric(label="Gerçek Dolandırıcılık Vakası", value=f"{gercek_fraud}")
col_kpi3.metric(
    label="Potansiyel Riskli Tutar ($)", value=f"${toplam_fraud_tutar:,.2f}"
)

st.markdown("---")

# İlk 5 satır
if st.checkbox("Veri setinin ilk 5 satırını göster"):
    st.dataframe(df.head())

# İşlem dağılımı ve Görünüm Modu
st.subheader("📊 Normal ve Dolandırıcılık İşlem Dağılımı")

grafik_tipi = st.radio(
    "Grafik Görünüm Modu:",
    ("Gerçek Veri Dağılımı (Çok Dengesiz)", "Sunum İçin Dengelenmiş Görünüm"),
)

if "Gerçek" in grafik_tipi:
    fraud_count = df["Class"].value_counts()
    st.bar_chart(fraud_count)
else:
    normal_ornek = df[df["Class"] == 0].sample(500, random_state=42)
    fraud_ornek = df[df["Class"] == 1]
    dengeli_df = pd.concat([normal_ornek, fraud_ornek])
    st.bar_chart(dengeli_df["Class"].value_counts())
    st.info(
        "💡 Not: Bu görünüm, incelemelerde aradaki oran farkının daha net"
        " anlaşılması için örneklem bazlı dengelenmiştir."
    )

# --- MODEL SEÇİM ALANI ---
st.markdown("---")
st.subheader("⚙️ Model Seçimi ve Eğitimi")
secilen_model_adi = st.selectbox(
    "Kullanmak istediğiniz yapay zeka algoritmini seçin:",
    ("Random Forest Classifier", "Logistic Regression"),
)


# Seçilen modele göre modeli eğiten fonksiyon
@st.cache_resource
def model_egit(df, model_turu):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_turu == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    return model, X_test, y_test


# Model Eğitimi Butonu
if st.button("Seçilen Modeli Eğit ve Performansı Göster"):
    with st.spinner(f"{secilen_model_adi} eğitiliyor ve metrikler hesaplanıyor..."):
        model, X_test, y_test = model_egit(df, secilen_model_adi)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.subheader(f"🎯 {secilen_model_adi} Performans Raporu")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Dolandırıcılık Yakalama (Recall)",
            value=f"{report['1']['recall']:.2f}",
        )
        col2.metric(
            label="Kesinlik (Precision)", value=f"{report['1']['precision']:.2f}"
        )
        col3.metric(label="F1-Skoru", value=f"{report['1']['f1-score']:.2f}")

        # Görsel Karmaşıklık Matrisi
        st.subheader("📉 Karmaşıklık Matrisi (Confusion Matrix)")
        col_sol, col_orta, col_sag = st.columns([1, 2, 1])
        with col_orta:
            fig, ax = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
            st.pyplot(fig)

        st.success("Model eğitimi ve görselleştirme tamamlandı!")

# --- SEKMELİ İLERİ DÜZEY TEST & BENCHMARK PANELİ ---
st.markdown("---")
st.subheader("🔍 Gelişmiş Test, Canlı Akış & Model Kıyaslama Paneli")

tab1, tab2, tab3 = st.tabs(
    [
        "Rastgele & Risk Skorlu Test",
        "Canlı İşlem Akışı Simülasyonu",
        "Çoklu Model Karşılaştırma (Benchmark)",
    ]
)

with tab1:
    islem_turu = st.selectbox(
        "Test etmek istediğiniz işlem türünü seçin:",
        ("Normal İşlem (0)", "Dolandırıcılık İşlemi (1)"),
    )

    if st.button("Risk Skorlu Testi Başlat"):
        model, X_test, y_test = model_egit(df, secilen_model_adi)
        if "Normal" in islem_turu:
            ornek = df[df["Class"] == 0].sample(1)
        else:
            ornek = df[df["Class"] == 1].sample(1)

        gercek_durum = ornek["Class"].values[0]
        girdi_verisi = ornek.drop("Class", axis=1)

        tahmin = model.predict(girdi_verisi)[0]
        # Olasılık skoru (Risk yüzdesi hesaplama)
        olasilik = model.predict_proba(girdi_verisi)[0][1] * 100

        st.write(f"**İşlem Tutarı (Amount):** ${ornek['Amount'].values[0]:,.2f}")

        # Risk Skoruna Göre Görsel Çubuk
        st.write(f"**Yapay Zeka Risk Skoru:** %{olasilik:.2f}")
        st.progress(float(olasilik / 100))

        if tahmin == 1:
            st.error(
                f"🚨 YÜKSEK RİSK! Yapay Zeka ({secilen_model_adi}) Tahmini: Bu işlem **DOLANDIRICILIK (Fraud)**!"
            )
        else:
            st.success(
                f"✅ GÜVENLİ! Yapay Zeka ({secilen_model_adi}) Tahmini: Bu işlem **NORMAL**."
            )

        if gercek_durum == tahmin:
            st.info("🎯 Sonuç: Model bu işlemi **DOĞRU** sınıflandırdı.")
        else:
            st.warning("⚠️ Sonuç: Model bu işlemde **yanıldı**.")

with tab2:
    st.write(
        "Gerçek zamanlı banka akışını simüle edin. Sisteme art arda işlemler"
        " düşecek ve riskli olanlar anında yakalanacaktır."
    )
    if st.button("🚀 Canlı Akış Simülasyonunu Başlat"):
        model, X_test, y_test = model_egit(df, secilen_model_adi)
        rastgele_akis = df.sample(5, random_state=None)

        sim_box = st.empty()
        for index, row in rastgele_akis.iterrows():
            tek_girdi = row.drop("Class").to_frame().T
            tahmin = model.predict(tek_girdi)[0]
            risk_orani = model.predict_proba(tek_girdi)[0][1] * 100
            tutar = row["Amount"]

            if tahmin == 1:
                sim_box.error(
                    f"⚠️ Canlı Akış Uyarısı! Tutar: ${tutar:,.2f} | Risk:"
                    f" %{risk_orani:.1f} ➔ **ŞÜPHELİ İŞLEM ENGELLENDİ!**"
                )
            else:
                sim_box.success(
                    f"✅ Canlı Akış Normal. Tutar: ${tutar:,.2f} | Risk:"
                    f" %{risk_orani:.1f} ➔ Onaylandı."
                )
            time.sleep(1.2)
        st.success("Canlı akış simülasyonu başarıyla tamamlandı.")

with tab3:
    st.write(
        "Random Forest ile Logistic Regression modellerinin performanslarını"
        " bilimsel olarak kıyaslayın:"
    )
    if st.button("Modelleri Kıyasla (Benchmark Çalıştır)"):
        with st.spinner("Modeller eğitiliyor ve karşılaştırılıyor..."):
            X = df.drop("Class", axis=1)
            y = df["Class"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # RF
            rf = RandomForestClassifier(n_estimators=30, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_rep = classification_report(y_test, rf_pred, output_dict=True)

            # LR
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_rep = classification_report(y_test, lr_pred, output_dict=True)

            # Karşılaştırma Tablosu
            karsilastirma_df = pd.DataFrame(
                {
                    "Metrik": [
                        "Dolandırıcılık Yakalama (Recall)",
                        "Kesinlik (Precision)",
                        "F1-Skoru",
                    ],
                    "Random Forest": [
                        rf_rep["1"]["recall"],
                        rf_rep["1"]["precision"],
                        rf_rep["1"]["f1-score"],
                    ],
                    "Logistic Regression": [
                        lr_rep["1"]["recall"],
                        lr_rep["1"]["precision"],
                        lr_rep["1"]["f1-score"],
                    ],
                }
            )

            st.dataframe(karsilastirma_df, use_container_width=True)
            st.info(
                "💡 Grafik ve tablo analizi gösteriyor ki Random Forest, dengesiz"
                " veri setlerinde daha yüksek başarım sergilemektedir."
            )import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="Kredi Kartı Dolandırıcılık Tespiti", page_icon="🚨", layout="wide"
)

# Web sayfası başlığı
st.title("🚨 Kredi Kartı Dolandırıcılık Tespiti (Fraud Detection)")
st.write(
    "Bu uygulama, dengesiz veri setleri üzerinde farklı makine öğrenmesi"
    " algoritmaları kullanarak şüpheli kredi kartı işlemlerini tespit eder."
)


@st.cache_data
def veri_yukle():
  # creditcard.csv dosyasının proje klasöründe olduğundan emin olun
  df = pd.read_csv("creditcard.csv")
  return df


with st.spinner("Veri yükleniyor, lütfen bekleyin..."):
  df = veri_yukle()

st.success(f"Veri başarıyla yüklendi! Toplam Satır ve Sütun: {df.shape}")

# İlk 5 satır
if st.checkbox("Veri setinin ilk 5 satırını göster"):
  st.dataframe(df.head())

# İşlem dağılımı ve Görünüm Modu
st.subheader("📊 Normal dan Dolandırıcılık İşlem Dağılımı")

grafik_tipi = st.radio(
    "Grafik Görünüm Modu:",
    ("Gerçek Veri Dağılımı (Çok Dengesiz)", "Sunum İçin Dengelenmiş Görünüm"),
)

if "Gerçek" in grafik_tipi:
  fraud_count = df["Class"].value_counts()
  st.bar_chart(fraud_count)
else:
  normal_ornek = df[df["Class"] == 0].sample(500, random_state=42)
  fraud_ornek = df[df["Class"] == 1]
  dengeli_df = pd.concat([normal_ornek, fraud_ornek])
  st.bar_chart(dengeli_df["Class"].value_counts())
  st.info(
      "💡 Not: Bu görünüm, incelemelerde aradaki oran farkının daha net"
      " anlaşılması için örneklem bazlı dengelenmiştir."
  )

# --- MODEL SEÇİM ALANI ---
st.markdown("---")
st.subheader("⚙️ Model Seçimi ve Eğitimi")
secilen_model_adi = st.selectbox(
    "Kullanmak istediğiniz yapay zeka algoritmasını seçin:",
    ("Random Forest Classifier", "Logistic Regression"),
)


# Seçilen modele göre modeli eğiten fonksiyon
@st.cache_resource
def model_egit(df, model_turu):
  X = df.drop("Class", axis=1)
  y = df["Class"]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  if model_turu == "Random Forest Classifier":
    model = RandomForestClassifier(n_estimators=50, random_state=42)
  else:
    model = LogisticRegression(max_iter=1000, random_state=42)

  model.fit(X_train, y_train)
  return model, X_test, y_test


# Model Eğitimi Butonu
if st.button("Seçilen Modeli Eğit ve Performansı Göster"):
  with st.spinner(f"{secilen_model_adi} eğitiliyor ve metrikler hesaplanıyor..."):
    model, X_test, y_test = model_egit(df, secilen_model_adi)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader(f"🎯 {secilen_model_adi} Performans Raporu")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Dolandırıcılık Yakalama (Recall)",
        value=f"{report['1']['recall']:.2f}",
    )
    col2.metric(
        label="Kesinlik (Precision)", value=f"{report['1']['precision']:.2f}"
    )
    col3.metric(label="F1-Skoru", value=f"{report['1']['f1-score']:.2f}")

    # Görsel Karmaşıklık Matrisi (Kompakt ve ortalanmış boyut)
    st.subheader("📉 Karmaşıklık Matrisi (Confusion Matrix)")
    col_sol, col_orta, col_sag = st.columns([1, 2, 1])
    with col_orta:
      fig, ax = plt.subplots(figsize=(4, 3))
      ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
      st.pyplot(fig)

    st.success("Model eğitimi ve görselleştirme tamamlandı!")

# --- İNTERAKTİF TEST PANELİ ---
st.markdown("---")
st.subheader("🔍 Canlı İşlem Test Etme Paneli")
tab1, tab2 = st.tabs(["Veri Setinden Rastgele Test", "Manuel Değer Girişi"])

with tab1:
  islem_turu = st.selectbox(
      "Test etmek istediğiniz işlem türünü seçin:",
      ("Normal İşlem (0)", "Dolandırıcılık İşlemi (1)"),
  )

  if st.button("Rastgele İşlemi Test Et"):
    model, X_test, y_test = model_egit(df, secilen_model_adi)
    if "Normal" in islem_turu:
      ornek = df[df["Class"] == 0].sample(1)
    else:
      ornek = df[df["Class"] == 1].sample(1)

    gercek_durum = ornek["Class"].values[0]
    girdi_verisi = ornek.drop("Class", axis=1)
    tahmin = model.predict(girdi_verisi)[0]

    st.write(f"**Seçilen İşlemin Tutarı (Amount):** {ornek['Amount'].values[0]}")

    if tahmin == 1:
      st.error(
          f"🚨 Yapay Zeka ({secilen_model_adi}) Tahmini: Bu işlem"
          " **DOLANDIRICILIK (Fraud)**!"
      )
    else:
      st.success(
          f"✅ Yapay Zeka ({secilen_model_adi}) Tahmini: Bu işlem **NORMAL**."
      )

    if gercek_durum == tahmin:
      st.info("🎯 Sonuç: Yapay zeka bu işlemi **DOĞRU** bildi!")
    else:
      st.warning("⚠️ Sonuç: Yapay zeka bu işlemde **yanıldı**.")

with tab2:
  st.write("İşlem tutarını girerek modelin anlık tahminini inceleyin:")
  manuel_amount = st.number_input(
      "İşlem Tutarı (Amount)", min_value=0.0, max_value=25000.0, value=50.0
  )

  if st.button("Manuel Tutarı Test Et"):
    model, X_test, y_test = model_egit(df, secilen_model_adi)
    ornek_sablon = df.drop("Class", axis=1).iloc[[0]].copy()
    ornek_sablon["Amount"] = manuel_amount

    tahmin = model.predict(ornek_sablon)[0]
    if tahmin == 1:
      st.error(
          f"🚨 Yapay Zeka ({secilen_model_adi}) Tahmini: Girilen tutar şüpheli"
          " görüldü (**DOLANDIRICILIK**)"
      )
    else:
      st.success(
          f"✅ Yapay Zeka ({secilen_model_adi}) Tahmini: Girilen tutar güvenli"
          " görüldü (**NORMAL**)"
      )