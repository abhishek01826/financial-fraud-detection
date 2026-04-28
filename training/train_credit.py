import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# ✅ Get absolute project root
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ Correct paths
data_path = BASE_DIR / "data" / "credit_data.csv"
model_dir = BASE_DIR / "models"

# ✅ Ensure models folder exists
model_dir.mkdir(parents=True, exist_ok=True)

# ✅ Debug print (very useful)
print("Looking for data at:", data_path)

if not data_path.exists():
    raise FileNotFoundError(f"❌ Missing input data at: {data_path}")

# Load data
df = pd.read_csv(data_path)

# Split
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
model_path = model_dir / "credit_model.pkl"
joblib.dump(model, model_path)

print("✅ Credit model trained!")
print("📁 Saved at:", model_path)