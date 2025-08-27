import streamlit as st
import numpy as np
import joblib

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
user_input = []
col1, col2 = st.columns(2)

for i, feat in enumerate(features):
    if i % 2 == 0:
        val = col1.number_input(f"{feat}", value=0.0, format="%.2f")
    else:
        val = col2.number_input(f"{feat}", value=0.0, format="%.2f")
    user_input.append(val)

# Convert ke numpy array
user_array = np.array(user_input).reshape(1, -1)

# --------------------
# Prediction
# --------------------
if st.button("Prediksi Jenis Anemia"):
    # Scaling input
    user_scaled = scaler.transform(user_array)

    # Prediksi
    prediction = rf_model.predict(user_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    # Probabilitas per kelas
    proba = rf_model.predict_proba(user_scaled)[0]

    st.subheader("📊 Hasil Prediksi")
    st.write(f"**Diagnosis: {prediction_label}**")

    # Tampilkan probabilitas tiap kelas
    st.subheader("Probabilitas Tiap Kelas")
    for cls, p in zip(label_encoder.classes_, proba):
        st.write(f"{cls}: {p:.4f}")
