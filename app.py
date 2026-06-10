import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD MODEL DAN ENCODER
# ==========================
try:
    model = joblib.load("model_knn.pkl")
    encoders = joblib.load("encoders.pkl")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================
# HEADER
# ==========================
st.set_page_config(
    page_title="Prediksi Prestasi Siswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Prestasi Siswa")
st.write("Klasifikasi Grade Siswa Menggunakan KNN")

# ==========================
# INPUT USER
# ==========================

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

education = st.selectbox(
    "Pendidikan Orang Tua",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox(
    "Lunch",
    ["standard", "free/reduced"]
)

prep = st.selectbox(
    "Test Preparation",
    ["none", "completed"]
)

math = st.slider(
    "Math Score",
    0, 100, 70
)

reading = st.slider(
    "Reading Score",
    0, 100, 70
)

writing = st.slider(
    "Writing Score",
    0, 100, 70
)

# ==========================
# PREDIKSI
# ==========================

if st.button("Prediksi"):

    try:

        input_data = pd.DataFrame({
            'gender': [encoders['gender'].transform([gender])[0]],
            'race/ethnicity': [encoders['race/ethnicity'].transform([race])[0]],
            'parental level of education': [encoders['parental level of education'].transform([education])[0]],
            'lunch': [encoders['lunch'].transform([lunch])[0]],
            'test preparation course': [encoders['test preparation course'].transform([prep])[0]],
            'math score': [math],
            'reading score': [reading],
            'writing score': [writing]
        })

        st.subheader("Data Uji")
        st.dataframe(input_data)

        prediksi = model.predict(input_data)

        total = math + reading + writing
        persen = total / 300 * 100

        st.success(
            f"Prediksi Grade Siswa : {prediksi[0]}"
        )

        st.metric(
            "Persentase Nilai",
            f"{persen:.2f}%"
        )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# ==========================
# DATA UJI OTOMATIS
# ==========================

st.markdown("---")

if st.button("Coba Data Uji Otomatis"):

    try:

        test_data = pd.DataFrame({
            'gender': [0],
            'race/ethnicity': [2],
            'parental level of education': [3],
            'lunch': [1],
            'test preparation course': [0],
            'math score': [85],
            'reading score': [88],
            'writing score': [90]
        })

        st.subheader("Data Uji Otomatis")
        st.dataframe(test_data)

        hasil = model.predict(test_data)

        st.success(
            f"Hasil Prediksi : {hasil[0]}"
        )

    except Exception as e:
        st.error(f"Error Data Uji: {e}")

# ==========================
# DEBUG ENCODER
# ==========================

with st.expander("Debug Encoder"):

    st.write("Tipe encoders:")
    st.write(type(encoders))

    st.write("Isi encoders:")
    st.write(encoders)
