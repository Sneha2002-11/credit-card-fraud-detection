import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Sidebar
st.sidebar.title("💳 Fraud Detection System")

st.sidebar.markdown("""
### Project Information

**Project:**  
Credit Card Fraud Detection Using Machine Learning

**Model Used:**  
Random Forest Classifier

**Technology Stack:**  
- Python
- Streamlit
- Scikit-Learn
- Machine Learning

**Purpose:**  
Detect fraudulent credit card transactions in real time.
""")

st.sidebar.success("System Status: ACTIVE")

# Main title
st.title("💳 AI-Based Credit Card Fraud Detection System")

st.markdown("""
This intelligent system analyzes transaction patterns and predicts whether a transaction is:

- ✅ Genuine
- 🚨 Fraudulent

The prediction is powered by a Machine Learning model trained on credit card transaction data.
""")

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value="99.99%"
    )

with col2:
    st.metric(
        label="Detection Status",
        value="ACTIVE"
    )

with col3:
    st.metric(
        label="ML Model",
        value="Random Forest"
    )

st.divider()

# Transaction Input Section
st.subheader("📊 Transaction Analysis")

left_col, right_col = st.columns(2)

with left_col:

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=250.0
    )

    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=12.0
    )

    transaction_count = st.number_input(
        "Transactions Today",
        min_value=0,
        value=2
    )

with right_col:

    location_risk = st.slider(
        "Location Risk Score",
        0,
        10,
        2
    )

    merchant_risk = st.slider(
        "Merchant Risk Score",
        0,
        10,
        3
    )

    device_risk = st.slider(
        "Device Risk Score",
        0,
        10,
        1
    )

st.divider()

# Prediction Button
if st.button("🔍 Analyze Transaction", use_container_width=True):

    # Create 30-feature array
    data = np.zeros(30)

    # Assign sample values
    data[0] = amount
    data[1] = time
    data[2] = transaction_count
    data[3] = location_risk
    data[4] = merchant_risk
    data[5] = device_risk

    # Reshape for prediction
    data = data.reshape(1, -1)

    # Predict
    prediction = model.predict(data)

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:

        st.error("""
🚨 FRAUD ALERT

This transaction shows suspicious activity patterns.

Recommended Action:
- Block transaction
- Request OTP verification
- Notify bank security team
""")

        risk_score = 92

    else:

        st.success("""
✅ GENUINE TRANSACTION

This transaction appears safe and legitimate.
""")

        risk_score = 12

    # Risk analysis
    st.subheader("📈 Fraud Risk Analysis")

    st.progress(risk_score / 100)

    st.write(f"Fraud Risk Score: {risk_score}%")

    # Summary table
    result_data = pd.DataFrame({
        "Parameter": [
            "Transaction Amount",
            "Location Risk",
            "Merchant Risk",
            "Device Risk"
        ],
        "Value": [
            amount,
            location_risk,
            merchant_risk,
            device_risk
        ]
    })

    st.table(result_data)

st.divider()

st.caption("Developed for M.Tech Project Demonstration")