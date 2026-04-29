import tkinter as tk
from tkinter import ttk
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# ================= BASE =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# ================= UI HELPERS =================
def create_entry(frame, placeholder):
    entry = tk.Entry(frame, font=("Arial", 13), width=28, justify="center", bg="#1e1e2f", fg="white")
    entry.insert(0, placeholder)

    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)

    def on_focus_out(event):
        if entry.get() == "":
            entry.insert(0, placeholder)

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

    entry.pack(pady=6)
    return entry

def safe_float(val):
    try:
        return float(val)
    except:
        return None

# ================= MAIN DASHBOARD =================
def run_dashboard():
    root = tk.Tk()
    root.title("Financial Froud Detection System")
    root.geometry("1200x750")
    root.configure(bg="#1e1e2f")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # ================= CREDIT FRAUD =================
    frame1 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame1, text="💳 Credit Fraud")

    tk.Label(frame1, text="Transaction Analysis", bg="#2c2f4a", fg="cyan", font=("Arial", 16)).pack(pady=10)

    entries = []
    for ph in ["Amount", "Time", "Location", "Device Score"]:
        entries.append(create_entry(frame1, ph))

    def predict_credit():
        vals = [safe_float(e.get()) for e in entries]
        if None in vals:
            result_label.config(text="Invalid Input", fg="red")
            return

        prob = credit_model.predict_proba([vals])[0][1]
        result_label.config(text=f"Fraud Risk: {round(prob*100,2)}%", fg="cyan")

    tk.Button(frame1, text="Analyze Fraud", command=predict_credit).pack(pady=10)
    result_label = tk.Label(frame1, text="", bg="#2c2f4a", fg="white", font=("Arial", 14))
    result_label.pack()

    # ================= LOAN =================
    frame2 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame2, text="🏦 Loan Default")

    tk.Label(frame2, text="Loan Risk Analysis", bg="#2c2f4a", fg="orange", font=("Arial", 16)).pack(pady=10)

    loan_entries = []
    for ph in ["Income", "Loan Amount", "Credit Score", "Years"]:
        loan_entries.append(create_entry(frame2, ph))

    def predict_loan():
        vals = [safe_float(e.get()) for e in loan_entries]
        if None in vals:
            loan_label.config(text="Invalid Input", fg="red")
            return

        prob = loan_model.predict_proba([vals])[0][1]
        loan_label.config(text=f"Default Risk: {round(prob*100,2)}%", fg="orange")

    tk.Button(frame2, text="Analyze Loan", command=predict_loan).pack(pady=10)
    loan_label = tk.Label(frame2, bg="#2c2f4a", fg="white", font=("Arial", 14))
    loan_label.pack()

    # ================= RISK =================
    frame3 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame3, text="📊 Risk Score")

    tk.Label(frame3, text="Customer Risk Engine", bg="#2c2f4a", fg="yellow", font=("Arial", 16)).pack(pady=10)

    def predict_risk():
        vals = [safe_float(e.get()) for e in loan_entries]
        if None in vals:
            risk_label.config(text="Invalid Input", fg="red")
            return

        prob = risk_model.predict_proba([vals])[0][1]
        risk_label.config(text=f"Risk Score: {round(prob*100,2)}%", fg="yellow")

    tk.Button(frame3, text="Calculate Risk", command=predict_risk).pack(pady=10)
    risk_label = tk.Label(frame3, bg="#2c2f4a", fg="white", font=("Arial", 14))
    risk_label.pack()

    # ================= ANOMALY =================
    frame4 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame4, text="⚠️ Anomaly Detection")

    tk.Label(frame4, text="Real-Time Anomaly Engine", bg="#2c2f4a", fg="red", font=("Arial", 16)).pack(pady=10)

    def detect_anomaly():
        vals = [safe_float(e.get()) for e in entries]
        if None in vals:
            anomaly_label.config(text="Invalid Input", fg="red")
            return

        vals_scaled = anomaly_scaler.transform([vals])
        pred = anomaly_model.predict(vals_scaled)[0]

        anomaly_label.config(
            text="⚠️ Suspicious Transaction" if pred == -1 else "✅ Normal Transaction",
            fg="red" if pred == -1 else "green"
        )

    tk.Button(frame4, text="Scan Transaction", command=detect_anomaly).pack(pady=10)
    anomaly_label = tk.Label(frame4, bg="#2c2f4a", fg="white", font=("Arial", 14))
    anomaly_label.pack()

    # ================= CLUSTER =================
    frame5 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame5, text="📊 Segmentation")

    tk.Label(frame5, text="Customer Intelligence", bg="#2c2f4a", fg="cyan", font=("Arial", 16)).pack(pady=10)

    def cluster_user():
        vals = [safe_float(e.get()) for e in entries]
        if None in vals:
            cluster_label.config(text="Invalid Input", fg="red")
            return

        vals_scaled = cluster_scaler.transform([vals])
        cluster = cluster_model.predict(vals_scaled)[0]

        labels = ["Low Risk", "Regular", "High Spender", "Suspicious"]
        cluster_label.config(text=f"Segment: {labels[cluster]}", fg="cyan")

    tk.Button(frame5, text="Analyze Segment", command=cluster_user).pack(pady=10)
    cluster_label = tk.Label(frame5, bg="#2c2f4a", fg="white", font=("Arial", 14))
    cluster_label.pack()

    # ================= ANALYTICS =================
    frame6 = tk.Frame(notebook, bg="#2c2f4a")
    notebook.add(frame6, text="📈 Analytics")

    fig, axs = plt.subplots(2, 2, figsize=(9, 7))

    # Risk distribution
    loan_df["risk_score"] = loan_df["loan_amount"] / loan_df["income"]
    sns.histplot(loan_df["risk_score"], bins=30, ax=axs[0, 0])
    axs[0, 0].set_title("Risk Distribution")

    # Fraud vs amount
    sns.boxplot(x="is_fraud", y="amount", data=credit_df, ax=axs[0, 1])
    axs[0, 1].set_title("Fraud vs Amount")

    # Loan pattern
    sns.scatterplot(x="credit_score", y="loan_amount", hue="default",
                    data=loan_df, ax=axs[1, 0])
    axs[1, 0].set_title("Loan Risk Pattern")

    # Clustering
    sample = credit_df.sample(500)
    X = sample.drop("is_fraud", axis=1)
    X_scaled = cluster_scaler.transform(X)
    sample["cluster"] = cluster_model.predict(X_scaled)

    sns.scatterplot(x="amount", y="device_score", hue="cluster",
                    data=sample, ax=axs[1, 1])
    axs[1, 1].set_title("Customer Segments")

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame6)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # ================= STATUS BAR =================
    status = tk.Label(root, text="System Ready | All Models Loaded", bd=1,
                      relief=tk.SUNKEN, anchor=tk.W, bg="#1e1e2f", fg="white")
    status.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()
