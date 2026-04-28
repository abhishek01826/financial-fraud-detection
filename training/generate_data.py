import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# ✅ Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)

# ---------- CREDIT CARD DATA ----------
n = 5000

credit_data = pd.DataFrame({
    "amount": np.random.uniform(10, 5000, n),
    "time": np.random.uniform(0, 24, n),
    "location": np.random.randint(0, 50, n),
    "device_score": np.random.uniform(0, 1, n),
})

# ✅ Create FRAUD PATTERN (NOT RANDOM)
credit_data["is_fraud"] = (
    (credit_data["amount"] > 3000) &               # high amount
    (credit_data["device_score"] < 0.3) &          # suspicious device
    ((credit_data["time"] < 5) | (credit_data["time"] > 22))  # odd hours
).astype(int)

# Add small noise (realistic)
noise_idx = np.random.choice(n, size=int(0.02*n), replace=False)
credit_data.loc[noise_idx, "is_fraud"] = 1 - credit_data.loc[noise_idx, "is_fraud"]

credit_path = data_dir / "credit_data.csv"
credit_data.to_csv(credit_path, index=False)

# ---------- LOAN DATA ----------
loan_data = pd.DataFrame({
    "income": np.random.uniform(20000, 150000, n),
    "loan_amount": np.random.uniform(5000, 50000, n),
    "credit_score": np.random.uniform(300, 850, n),
    "employment_years": np.random.randint(0, 20, n),
})

# ✅ Create DEFAULT PATTERN
loan_data["default"] = (
    (loan_data["credit_score"] < 500) &            # low credit score
    (loan_data["loan_amount"] > 30000) &           # high loan
    (loan_data["income"] < 60000)                  # low income
).astype(int)

# Add noise
noise_idx = np.random.choice(n, size=int(0.03*n), replace=False)
loan_data.loc[noise_idx, "default"] = 1 - loan_data.loc[noise_idx, "default"]

loan_path = data_dir / "loan_data.csv"
loan_data.to_csv(loan_path, index=False)

# ---------- SUMMARY ----------
print("✅ Data generated successfully!")
print("📁 Credit data:", credit_path)
print("📁 Loan data:", loan_path)

print("\n📊 Fraud Distribution:")
print(credit_data["is_fraud"].value_counts())

print("\n📊 Loan Default Distribution:")
print(loan_data["default"].value_counts())