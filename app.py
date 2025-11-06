import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sys, numpy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Patch for old NumPy pickle structure
sys.modules["numpy._core"] = numpy.core

# --- Load model ---
model_path = r"C:\Users\preja\Downloads\churn_model_logistic.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# --- Custom Page Config ---
st.set_page_config(
    page_title="Customer Churn Predictor 💡",
    page_icon="📞",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Add Custom CSS for unique look ---
st.markdown("""
<style>
body {
    background-color: #0f2027;
    background-image: linear-gradient(315deg, #2c5364 0%, #203a43 50%, #0f2027 100%);
    color: white;
}
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
div.stButton > button:first-child:hover {
    background-color: #ff6f61;
    color: black;
}
h1, h2, h3 {
    color: #00d4ff;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("📞 Customer Churn Prediction Dashboard")
st.markdown("### 🚀 Predict whether a telecom customer is likely to churn.")
st.write("Adjust the parameters below and click **Predict Churn** to get instant results!")

# --- Sidebar for Inputs ---
st.sidebar.header("📋 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# --- Data Preparation ---
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

# One-hot encoding
user_input_encoded = pd.get_dummies(user_input)
expected_features = model.feature_names_in_
for col in expected_features:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0
user_input_encoded = user_input_encoded[expected_features]

# --- Prediction ---
if st.button("🔍 Predict Churn"):
    prediction = model.predict(user_input_encoded)[0]
    prob = model.predict_proba(user_input_encoded)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ **Customer likely to churn!** Probability: {prob:.2f}%")
        st.progress(int(prob))
    else:
        st.success(f"✅ **Customer not likely to churn.** Probability: {prob:.2f}%")
        st.progress(int(prob))
