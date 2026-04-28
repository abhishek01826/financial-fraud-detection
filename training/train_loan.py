import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from pathlib import Path

# ✅ Get absolute base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ Correct paths
data_path = BASE_DIR / "data" / "loan_data.csv"
model_path = BASE_DIR / "models" / "loan_model.pkl"

# ✅ Ensure models folder exists
model_path.parent.mkdir(parents=True, exist_ok=True)

# ✅ Debug print
print("📂 Loading data from:", data_path)

if not data_path.exists():
    raise FileNotFoundError(f"❌ Missing input data at: {data_path}")

# Load data
df = pd.read_csv(data_path)

# Split
X = df.drop("default", axis=1)
y = df["default"]

# Train model
model = GradientBoostingClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, model_path)

print("✅ Loan model trained successfully!")
print("📁 Saved at:", model_path)