import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model("heart_disease_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("💓 Heart Disease Prediction App")

# User input form (13 features as per UCI Heart Dataset)
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 250)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of the ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Combine inputs into a single array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    result = "⚠️ Likely Heart Disease" if prediction[0][0] > 0.5 else "✅ No Heart Disease"
    st.subheader("Prediction Result:")
    st.success(result) if prediction[0][0] <= 0.5 else st.error(result)