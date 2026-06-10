import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_knn.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(
    page_title="Prediksi Prestasi Siswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Prestasi Siswa")
st.write("Klasifikasi Grade Menggunakan KNN")

# Input User
gender = st.selectbox(
    "Gender",
    ["female","male"]
)

race = st.selectbox(
    "Race/Ethnicity",
    ["group A","group B","group C","group D","group E"]
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
    ["standard","free/reduced"]
)

prep = st.selectbox(
    "Test Preparation",
    ["none","completed"]
)

math = st.slider(
    "Math Score",
    0,100,70
)

reading = st.slider(
    "Reading Score",
    0,100,70
)

writing = st.slider(
    "Writing Score",
    0,100,70
)

if st.button("Prediksi"):

    input_data = pd.DataFrame({
        'gender':[encoders['gender'].transform([gender])[0]],
        'race/ethnicity':[encoders['race/ethnicity'].transform([race])[0]],
        'parental level of education':[encoders['parental level of education'].transform([education])[0]],
        'lunch':[encoders['lunch'].transform([lunch])[0]],
        'test preparation course':[encoders['test preparation course'].transform([prep])[0]],
        'math score':[math],
        'reading score':[reading],
        'writing score':[writing]
    })

    prediksi = model.predict(input_data)

    st.success(
        f"Prediksi Grade Siswa : {prediksi[0]}"
    )

    total = math + reading + writing
    persen = total/300*100

    st.metric(
        "Persentase Nilai",
        f"{persen:.2f}%"
    )
