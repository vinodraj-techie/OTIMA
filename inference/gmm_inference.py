import sys
import pandas as pd
import joblib
import os

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

model = joblib.load("models/gmm2/gmm_model.pkl")
model.eval()

features = pd.read_csv(f"{dataset_path}/gmm_features.csv")

X = features.drop(columns=["sku_id"])

clusters = model.predict(X)
confidence = model.predict_proba(X).max(axis=1)

output = pd.DataFrame({
    "sku_id": features["sku_id"],
    "cluster": clusters,
    "confidence": confidence
})

output.to_csv(f"{result_path}/gmm_clusters.csv", index=False)

print("GMM inference completed")