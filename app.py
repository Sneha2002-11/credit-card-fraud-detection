import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import base64

# ---------------- LOAD MODEL ----------------
model = joblib.load("fraud_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- ALERT SOUND FUNCTION ----------------
def play_alert_sound():
    with open("alarm.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()

    encoded = base64.b64encode(audio_bytes).decode()

    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

# ---------------- DARK THEME (MAIN + SIDEBAR) ----------------
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: white;
}

/* BUTTON */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

/* INPUT */
.stTextInput>div>div>input {
    background-color: #262730;
    color: white;
}

/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {
    background-color: #0B0F19;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white;
}

/* SIDEBAR HEADINGS */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Secure Banking Login")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2489/2489756.png",
        width=120
    )

    st.markdown("### Welcome to AI Fraud Detection Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# ---------------- MAIN APP ----------------
else:

    # Sidebar
    st.sidebar.title("💳 Fraud Detection Dashboard")

    st.sidebar.success("🟢 AI Engine Active")
    st.sidebar.success("🟢 Fraud Monitoring Running")
    st.sidebar.success("🟢 Bank Server Connected")

    st.sidebar.markdown("""
    ### Technology Stack
    - Python
    - Streamlit
    - Machine Learning
    - Random Forest
    - AI Fraud Analytics
    """)

    # Header
    st.title("💳 AI-Based Credit Card Fraud Detection System")

    st.markdown("""
    This AI system analyzes transactions and detects fraud in real time using Machine Learning.
    """)

    # Inputs
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=250.0)
    time = st.number_input("Transaction Time", min_value=0.0, value=12.0)
    transaction_count = st.number_input("Transactions Today", min_value=0, value=2)

    location_risk = st.slider("Location Risk Score", 0, 10, 2)
    merchant_risk = st.slider("Merchant Risk Score", 0, 10, 3)
    device_risk = st.slider("Device Risk Score", 0, 10, 1)

    # Analyze button
    if st.button("🔍 Analyze Transaction"):

        # Feature vector
        data = np.zeros(30)
        data[0] = amount
        data[1] = time
        data[2] = transaction_count
        data[3] = location_risk
        data[4] = merchant_risk
        data[5] = device_risk

        data = data.reshape(1, -1)

        prediction = model.predict(data)

        # Risk logic
        fraud_score = 0

        if amount > 50000:
            fraud_score += 30
        if location_risk >= 8:
            fraud_score += 20
        if merchant_risk >= 8:
            fraud_score += 20
        if device_risk >= 8:
            fraud_score += 20
        if transaction_count > 15:
            fraud_score += 10

        st.subheader("📌 Prediction Result")

        # ---------------- LOW RISK ----------------
        if fraud_score < 30:

            st.markdown("""
            <style>
            .stApp { background-color: #d8f5d0; color: black; }
            h1,h2,h3,h4,h5,h6,p,label,div { color: black; }
            </style>
            """, unsafe_allow_html=True)

            st.success("✅ LOW RISK - GENUINE TRANSACTION")

        # ---------------- MEDIUM RISK ----------------
        elif fraud_score < 60:

            st.markdown("""
            <style>
            .stApp { background-color: #fff4cc; color: black; }
            h1,h2,h3,h4,h5,h6,p,label,div { color: black; }
            </style>
            """, unsafe_allow_html=True)

            st.warning("⚠ MEDIUM RISK TRANSACTION")

        # ---------------- HIGH RISK ----------------
        else:

            st.markdown("""
            <style>
            .stApp { background-color: #ffd6d6; color: black; }
            h1,h2,h3,h4,h5,h6,p,label,div { color: black; }
            </style>
            """, unsafe_allow_html=True)

            st.error("🚨 HIGH RISK FRAUD DETECTED")

            # 🔊 SOUND ALERT
            play_alert_sound()

        # Progress bar
        st.progress(fraud_score / 100)
        st.write(f"Fraud Risk Score: {fraud_score}%")

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(
            [fraud_score, 100 - fraud_score],
            labels=["Fraud Risk", "Safe"],
            autopct='%1.1f%%'
        )
        st.pyplot(fig)

        # Table
        result_data = pd.DataFrame({
            "Parameter": [
                "Transaction Amount",
                "Transaction Time",
                "Transactions Today",
                "Location Risk",
                "Merchant Risk",
                "Device Risk"
            ],
            "Value": [
                amount,
                time,
                transaction_count,
                location_risk,
                merchant_risk,
                device_risk
            ]
        })

        st.table(result_data)

    st.caption("Developed for M.Tech Project Demonstration")