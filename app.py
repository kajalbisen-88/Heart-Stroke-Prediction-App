import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- Load Model & Files ----
model = joblib.load("KNN_heart.pkl")      # FIXED
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction by Kajal ❤️")
st.markdown("Provide the following details:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])

max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---- Prediction Logic ----
if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts HIGH RISK of heart disease. Please consult a doctor.")
    else:
        st.success("The model predicts LOW RISK of heart disease. Stay healthy!")


