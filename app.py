import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("model_knn.pkl")
feature_columns = joblib.load("feature_columns.pkl")

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
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
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
    "Jenis Lunch",
    [
        "free/reduced",
        "standard"
    ]
)

prep = st.selectbox(
    "Test Preparation",
    [
        "none",
        "completed"
    ]
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

# ==========================
# PREDIKSI
# ==========================

if st.button("Prediksi Grade"):

    input_dict = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": prep,
        "math score": math_score,
        "reading score": reading_score,
        "writing score": writing_score
    }

    input_df = pd.DataFrame([input_dict])

    # One Hot Encoding sama seperti training
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    # Samakan kolom dengan model training
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    total = (
        math_score +
        reading_score +
        writing_score
    )

    percentage = total / 300 * 100

    st.success(
        f"Prediksi Grade : {prediction}"
    )

    st.metric(
        "Persentase Nilai",
        f"{percentage:.2f}%"
    )

    st.write("### Data Input")

    st.dataframe(input_df)
