import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os

st.title("❤️ Heart Disease Prediction by Kajal ❤️")
st.markdown("Provide the following details:")

# ---- Load model, scaler, and columns safely ----
try:
    with open("KNN_heart.pkl", "rb") as f:
        model = cloudpickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = cloudpickle.load(f)
    with open("columns.pkl", "rb") as f:
        expected_columns = cloudpickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Make sure 'KNN_heart.pkl', 'scaler.pkl', and 'columns.pkl' are in the repo.")
    st.stop()

# ---- User Inputs ----
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

# ---- Prediction ----
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

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("High risk of heart disease. Please consult a doctor.")
        else:
            st.success("Low risk of heart disease. Keep a healthy lifestyle!")
    except Exception as e:
        st.error(f"Prediction error: {e}")




