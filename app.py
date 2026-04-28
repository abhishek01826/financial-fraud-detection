import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= BASE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD DATA =================
credit_df = pd.read_csv(os.path.join(BASE_DIR, "data", "credit_data.csv"))
loan_df = pd.read_csv(os.path.join(BASE_DIR, "data", "loan_data.csv"))

# ================= LOAD MODELS =================
credit_model = joblib.load(os.path.join(BASE_DIR, "models", "credit_model.pkl"))
loan_model = joblib.load(os.path.join(BASE_DIR, "models", "loan_model.pkl"))
risk_model = joblib.load(os.path.join(BASE_DIR, "models", "risk_model.pkl"))
anomaly_model = joblib.load(os.path.join(BASE_DIR, "models", "anomaly_model.pkl"))
cluster_model = joblib.load(os.path.join(BASE_DIR, "models", "cluster_model.pkl"))

anomaly_scaler = joblib.load(os.path.join(BASE_DIR, "models", "anomaly_scaler.pkl"))
cluster_scaler = joblib.load(os.path.join(BASE_DIR, "models", "cluster_scaler.pkl"))

# ================= PAGE =================
st.set_page_config(page_title="AI Fraud Intelligence", layout="wide")

st.title("🚀 AI Fraud Intelligence Dashboard")

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Select Module", [
    "Credit Fraud Detection",
    "Loan Default Prediction",
    "Customer Risk Score",
    "Anomaly Detection",
    "Customer Segmentation",
    "Analytics Dashboard"
])

# ================= CREDIT FRAUD =================
if menu == "Credit Fraud Detection":
    st.header("💳 Credit Fraud Detection")

    col1, col2 = st.columns(2)

    amount = col1.number_input("Amount", 10.0, 5000.0, 100.0)
    time = col2.number_input("Time", 0.0, 24.0, 12.0)
    location = col1.number_input("Location", 0.0, 50.0, 10.0)
    device = col2.number_input("Device Score", 0.0, 1.0, 0.5)

    if st.button("Predict Fraud"):
        vals = [[amount, time, location, device]]
        prob = credit_model.predict_proba(vals)[0][1]
        st.success(f"Fraud Risk: {round(prob*100,2)}%")

# ================= LOAN =================
elif menu == "Loan Default Prediction":
    st.header("🏦 Loan Default Prediction")

    col1, col2 = st.columns(2)

    income = col1.number_input("Income", 20000.0, 150000.0, 50000.0)
    loan = col2.number_input("Loan Amount", 5000.0, 50000.0, 20000.0)
    score = col1.number_input("Credit Score", 300.0, 850.0, 600.0)
    years = col2.number_input("Employment Years", 0.0, 20.0, 5.0)

    if st.button("Predict Default"):
        vals = [[income, loan, score, years]]
        prob = loan_model.predict_proba(vals)[0][1]
        st.warning(f"Default Risk: {round(prob*100,2)}%")

# ================= RISK =================
elif menu == "Customer Risk Score":
    st.header("📊 Risk Scoring")

    col1, col2 = st.columns(2)

    income = col1.number_input("Income", 20000.0, 150000.0, 50000.0)
    loan = col2.number_input("Loan Amount", 5000.0, 50000.0, 20000.0)
    score = col1.number_input("Credit Score", 300.0, 850.0, 600.0)
    years = col2.number_input("Employment Years", 0.0, 20.0, 5.0)

    if st.button("Calculate Risk"):
        vals = [[income, loan, score, years]]
        prob = risk_model.predict_proba(vals)[0][1]
        st.info(f"Risk Score: {round(prob*100,2)}%")

# ================= ANOMALY =================
elif menu == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection")

    col1, col2 = st.columns(2)

    amount = col1.number_input("Amount", 10.0, 5000.0, 100.0)
    time = col2.number_input("Time", 0.0, 24.0, 12.0)
    location = col1.number_input("Location", 0.0, 50.0, 10.0)
    device = col2.number_input("Device Score", 0.0, 1.0, 0.5)

    if st.button("Check Anomaly"):
        vals = [[amount, time, location, device]]
        vals_scaled = anomaly_scaler.transform(vals)
        pred = anomaly_model.predict(vals_scaled)[0]

        if pred == -1:
            st.error("⚠️ Anomaly Detected")
        else:
            st.success("✅ Normal Transaction")

# ================= CLUSTER =================
elif menu == "Customer Segmentation":
    st.header("📊 Customer Segmentation")

    col1, col2 = st.columns(2)

    amount = col1.number_input("Amount", 10.0, 5000.0, 100.0)
    time = col2.number_input("Time", 0.0, 24.0, 12.0)
    location = col1.number_input("Location", 0.0, 50.0, 10.0)
    device = col2.number_input("Device Score", 0.0, 1.0, 0.5)

    if st.button("Find Segment"):
        vals = [[amount, time, location, device]]
        vals_scaled = cluster_scaler.transform(vals)
        cluster = cluster_model.predict(vals_scaled)[0]

        labels = ["Low Risk", "Regular", "High Spender", "Suspicious"]
        st.success(f"Segment: {labels[cluster]}")

# ================= ANALYTICS =================
elif menu == "Analytics Dashboard":
    st.header("📈 Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Risk Distribution
    loan_df["risk_score"] = loan_df["loan_amount"] / loan_df["income"]

    fig1, ax1 = plt.subplots()
    sns.histplot(loan_df["risk_score"], bins=30, ax=ax1)
    ax1.set_title("Risk Distribution")
    col1.pyplot(fig1)

    # Fraud vs Amount
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="is_fraud", y="amount", data=credit_df, ax=ax2)
    ax2.set_title("Fraud vs Amount")
    col2.pyplot(fig2)

    # Loan Pattern
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="credit_score", y="loan_amount", hue="default", data=loan_df, ax=ax3)
    st.pyplot(fig3)

    # Clustering
    sample = credit_df.sample(500)
    X = sample.drop("is_fraud", axis=1)
    X_scaled = cluster_scaler.transform(X)
    sample["cluster"] = cluster_model.predict(X_scaled)

    fig4, ax4 = plt.subplots()
    sns.scatterplot(x="amount", y="device_score", hue="cluster", data=sample, ax=ax4)
    st.pyplot(fig4)