# Customer-Churn-Prediction-Model-Development-Validation-and-Deployment

# 📞 Customer Churn Prediction App

A machine learning–based web application that predicts whether a telecom customer is likely to **churn** (leave the service) or **stay**, based on their usage behavior and demographic information.  
This project helps telecom companies take **data-driven actions** to retain valuable customers.

---

## 🧠 Project Overview

Customer churn is a major problem in the telecom industry — retaining an existing customer is much cheaper than acquiring a new one.  
This project uses a **Logistic Regression** model to analyze customer data and predict churn probability.

A **Streamlit web app** provides an interactive interface for users to input customer details and instantly view predictions.

---

## 🎯 Objective

To develop and deploy a predictive model that determines whether a customer is at risk of churning based on their service usage patterns, contract type, and account details.

---

## ⚙️ How It Works

### 1. **Data Collection**
- Uses telecom customer data (similar to the IBM Telco Customer Churn dataset).  
- Key features include:
  - Gender, SeniorCitizen, Partner, Dependents  
  - Tenure, PhoneService, InternetService, MultipleLines  
  - Contract type, PaymentMethod, PaperlessBilling  
  - MonthlyCharges, TotalCharges  
  - Target variable: `Churn` (Yes/No)

### 2. **Data Preprocessing**
- Handled missing values and encoded categorical variables using **One-Hot Encoding**.
- Scaled numeric features for model consistency.

### 3. **Model Training**
- Trained a **Logistic Regression** model using `scikit-learn`.
- Evaluated model performance using metrics like **accuracy**, **precision**, and **recall**.
- Saved the trained model as a `.pkl` file using `pickle`.

### 4. **Model Deployment**
- Developed an interactive web interface using **Streamlit**.
- Users can enter customer information.
- The model predicts whether the customer is likely to churn.

---

## 🧩 Tech Stack

| Component | Tool / Library |
|------------|----------------|
| **Language** | Python |
| **Frontend Interface** | Streamlit |
| **Data Handling** | pandas, numpy |
| **Machine Learning Model** | scikit-learn (Logistic Regression) |
| **Serialization** | pickle |
| **Visualization / Deployment** | Streamlit |

---

## 🚀 Features

- 🧮 Predicts customer churn in real-time.  
- 💻 Interactive web-based user interface.  
- 📊 Uses logistic regression for interpretable predictions.  
- 🔒 Lightweight and easy to deploy.  
- 🧠 Designed for business decision-making and customer retention analysis.

---

## 🧰 Installation and Setup

### 1. Clone this repository:
```bash
git clone https://github.com/prejan/customer-churn-prediction.git
cd customer-churn-prediction
📂 Project Structure
├── app.py                        # Streamlit web application
├── churn_model_logistic.pkl      # Trained Logistic Regression model
├── requirements.txt              # Required dependencies
└── README.md                     # Project documentation
```
🧪 Example Input (via Streamlit UI)

Gender: Female

Senior Citizen: 0

Tenure: 24 months

Internet Service: Fiber optic

Monthly Charges: 85.5

Contract: Month-to-month

→ Output: ⚠️ “This customer is likely to churn!”

🧭 Applications

Telecom customer retention

Subscription-based services

Financial and SaaS churn analysis

Customer relationship management (CRM) analytics

📈 Future Improvements

Add more ML models (Random Forest, XGBoost).

Include feature importance visualization.

Deploy to Streamlit Cloud or Hugging Face Spaces.

Integrate with a live database for real-time churn monitoring.
<img width="409" height="870" alt="image" src="https://github.com/user-attachments/assets/3a20327b-4efc-4a28-9139-90cbcf4586dd" />

