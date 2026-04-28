import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "credit_data.csv"
model_path = BASE_DIR / "models" / "anomaly_model.pkl"
scaler_path = BASE_DIR / "models" / "anomaly_scaler.pkl"

model_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
X = df.drop("is_fraud", axis=1)

# ✅ Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Less aggressive anomaly detection
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_scaled)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Improved Anomaly model saved!")