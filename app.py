import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Anemia Detection System",
    page_icon="🩸",
    layout="wide",   # <--- ini yang bikin full screen
)

hide_scrollbar = """
    <style>
    ::-webkit-scrollbar {
        display: none;
    }
    </style>
"""
st.markdown(hide_scrollbar, unsafe_allow_html=True)

# --------------------
# Load model & tools
# --------------------
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------------------
# Streamlit UI
# --------------------
st.title("🩸 Anemia Detection System (Random Forest + SMOTE)")
st.markdown("Masukkan hasil **Complete Blood Count (CBC)** untuk memprediksi jenis anemia.")

# Fitur input (14 parameter CBC sesuai dataset)
features = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH",
    "MCHC", "PLT", "PDW", "PCT"
]

# Input form
# --------------------
# Layout 2 kolom utama: kiri input, kanan output
# --------------------
col1, col2 = st.columns([2, 1])  # kiri lebih lebar untuk form

with col1:
    st.subheader("📝 Input Data CBC")
    user_input = []
    col1a, col1b = st.columns(2)  # bagi 2 kolom lagi di dalam form kiri

    for i, feat in enumerate(features):
        if i % 2 == 0:
            val = col1a.number_input(f"{feat}", value=0.0, format="%.2f")
        else:
            val = col1b.number_input(f"{feat}", value=0.0, format="%.2f")
        user_input.append(val)

    user_array = np.array(user_input).reshape(1, -1)

with col2:
    st.subheader("📊 Hasil Prediksi")
    if st.button("Prediksi Jenis Anemia"):
        # Scaling input
        user_scaled = scaler.transform(user_array)

        # Prediksi
        prediction = rf_model.predict(user_scaled)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        # Probabilitas per kelas
        proba = rf_model.predict_proba(user_scaled)[0]

        st.success(f"**Diagnosis: {prediction_label}**")

        st.markdown("### Probabilitas Tiap Kelas")
        for cls, p in zip(label_encoder.classes_, proba):
            st.write(f"{cls}: {p:.4f}")
