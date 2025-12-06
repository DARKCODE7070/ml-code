import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("synthetic_power_anomalies_1000.csv")

print(df.head())

features = ["voltage", "current", "power", "energy_Wh"]
X = df[features]
y_true = df["label"]  # 0 = normal, 1 = anomaly

model = IsolationForest(
    n_estimators=200,
    contamination=df["label"].mean(),  # use actual anomaly percentage
    random_state=42,
    bootstrap=True
)

model.fit(X)
y_pred_raw = model.predict(X)
y_pred = np.where(y_pred_raw == -1, 1, 0)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
joblib.dump(model, "isolation_forest_new.pkl")
print("Model saved as isolation_forest_new.pkl")

