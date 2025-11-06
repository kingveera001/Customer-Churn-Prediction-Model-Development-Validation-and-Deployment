import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sys, numpy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Patch for old NumPy pickle structure
sys.modules["numpy._core"] = numpy.core

# --- Load model safely ---
model_path = r"C:\Users\preja\Downloads\churn_model_logistic.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("📞 Customer Churn Prediction App")
st.write("This app predicts whether a telecom customer is likely to churn.")

# --- Input fields for all 19 features ---
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# --- Create DataFrame ---
user_input = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# --- Encode categorical variables (one-hot) to match training format ---
user_input_encoded = pd.get_dummies(user_input)

# Align with model's expected columns
expected_features = model.feature_names_in_
for col in expected_features:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0
user_input_encoded = user_input_encoded[expected_features]

# --- Prediction ---
if st.button("Predict Churn"):
    prediction = model.predict(user_input_encoded)[0]
    if prediction == 1:
        st.error("⚠️ This customer is **likely to churn!** Take retention actions.")
    else:
        st.success("✅ This customer is **not likely to churn.**")
