import streamlit as st
import pandas as pd
import joblib

# ====================
# LOAD MODEL DAN DATA
# ====================
model = joblib.load("catboost_model.pkl")         # Model CatBoost hasil training
df = joblib.load("cleaned_df.pkl")                # Dataset hasil cleaning

# Ambil daftar lokasi unik (untuk dropdown)
locations = sorted(df["location"].unique())

# ====================
# HALAMAN UTAMA
# ====================
st.title("🏠 Prediksi Harga Rumah di Bandung")
st.markdown("Masukkan detail properti untuk memprediksi harga menggunakan model CatBoost.")

# ====================
# INPUT USER
# ====================
st.header("📌 Masukkan Informasi Properti")

location = st.selectbox("Lokasi", locations)
bedroom = st.slider("Jumlah Kamar Tidur", min_value=2, max_value=5, value=3)
bathroom = st.slider("Jumlah Kamar Mandi", min_value=1, max_value=4, value=2)
carport = st.slider("Jumlah Carport", min_value=0, max_value=5, value=1)
land_area = st.number_input("Luas Tanah (m²)", min_value=60, max_value=490, value=150)
building_area = st.number_input("Luas Bangunan (m²)", min_value=33, max_value=465, value=90)

# ====================
# PREDIKSI
# ====================
if st.button("🔮 Prediksi Harga"):
    input_data = pd.DataFrame([{
        "location": location,
        "bedroom_count": bedroom,
        "bathroom_count": bathroom,
        "carport_count": carport,
        "land_area": land_area,
        "building_area (m2)": building_area
    }])

    # Prediksi harga
    prediction = model.predict(input_data)[0]
    harga = f"Rp {int(prediction):,}".replace(",", ".")

    st.success(f"💰 Estimasi Harga Rumah: **{harga}**")