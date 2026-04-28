import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

# ✅ Absolute base path (VERY IMPORTANT)
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ Correct paths
data_path = BASE_DIR / "data" / "loan_data.csv"
model_path = BASE_DIR / "models" / "risk_model.pkl"

# ✅ Ensure models folder exists
model_path.parent.mkdir(parents=True, exist_ok=True)

# ✅ Debug print
print("📂 Loading data from:", data_path)

if not data_path.exists():
    raise FileNotFoundError(f"❌ Missing input data at: {data_path}")

# Load data
df = pd.read_csv(data_path)

# ✅ Feature engineering (risk ratio)
df["risk_score"] = df["loan_amount"] / (df["income"] + 1)  # +1 to avoid divide-by-zero

# Features & target
X = df[["income", "loan_amount", "credit_score", "employment_years"]]
y = df["default"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, model_path)

print("✅ Risk model trained successfully!")
print("📁 Saved at:", model_path)