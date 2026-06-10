import streamlit as st
import pandas as pd
import joblib

# =====================================
# LOAD MODEL
# =====================================

try:
    model = joblib.load("model_knn.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================
# HEADER
# =====================================

st.set_page_config(
    page_title="Prediksi Prestasi Siswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Prestasi Siswa")
st.write("Klasifikasi Grade Siswa Menggunakan KNN")

# =====================================
# INPUT USER
# =====================================

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

race = st.selectbox(
    "Race / Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

education = st.selectbox(
    "Pendidikan Orang Tua",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

lunch = st.selectbox(
    "Lunch",
    ["free/reduced", "standard"]
)

prep = st.selectbox(
    "Test Preparation",
    ["completed", "none"]
)

math_score = st.slider(
    "Math Score",
    0, 100, 70
)

reading_score = st.slider(
    "Reading Score",
    0, 100, 70
)

writing_score = st.slider(
    "Writing Score",
    0, 100, 70
)

# =====================================
# MAPPING MANUAL
# =====================================

gender_map = {
    "female": 0,
    "male": 1
}

race_map = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

education_map = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "high school": 2,
    "master's degree": 3,
    "some college": 4,
    "some high school": 5
}

lunch_map = {
    "free/reduced": 0,
    "standard": 1
}

prep_map = {
    "completed": 0,
    "none": 1
}

# =====================================
# PREDIKSI
# =====================================

if st.button("Prediksi"):

    try:

        input_data = pd.DataFrame({
            'gender': [gender_map[gender]],
            'race/ethnicity': [race_map[race]],
            'parental level of education': [education_map[education]],
            'lunch': [lunch_map[lunch]],
            'test preparation course': [prep_map[prep]],
            'math score': [math_score],
            'reading score': [reading_score],
            'writing score': [writing_score]
        })

        st.subheader("Data Uji")

        st.dataframe(input_data)

        prediction = model.predict(input_data)

        total = (
            math_score +
            reading_score +
            writing_score
        )

        percentage = total / 300 * 100

        st.success(
            f"Prediksi Grade : {prediction[0]}"
        )

        st.metric(
            "Persentase Nilai",
            f"{percentage:.2f}%"
        )

    except Exception as e:
        st.error(f"Error Prediksi: {e}")

# =====================================
# DATA UJI OTOMATIS
# =====================================

st.markdown("---")

if st.button("Coba Data Uji Otomatis"):

    try:

        test_data = pd.DataFrame({
            'gender': [0],
            'race/ethnicity': [2],
            'parental level of education': [4],
            'lunch': [1],
            'test preparation course': [0],
            'math score': [85],
            'reading score': [90],
            'writing score': [88]
        })

        st.subheader("Contoh Data Uji")

        st.dataframe(test_data)

        hasil = model.predict(test_data)

        st.success(
            f"Hasil Prediksi : {hasil[0]}"
        )

    except Exception as e:
        st.error(f"Error: {e}")
